#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/check.hpp"
#include "common/tensor.hpp"
#include "fastbev/fastbev.hpp"

namespace py = pybind11;

namespace {

constexpr int kNumCameras = 6;
constexpr int kImageHeight = 900;
constexpr int kImageWidth = 1600;
constexpr int kImageChannels = 3;
constexpr int kValidPoints = 160000;

std::string join_path(const std::string& base, const std::string& child) {
  if (base.empty()) return child;
  const char last = base[base.size() - 1];
  if (last == '/' || last == '\\') return base + child;
  return base + "/" + child;
}

void require_file(const std::string& path) {
  std::ifstream file(path.c_str(), std::ios::binary);
  if (!file.good()) {
    throw std::runtime_error("Missing required FastBEV artifact: " + path);
  }
}

void require_shape(const py::buffer_info& info, const std::vector<py::ssize_t>& expected,
                   const char* name) {
  if (info.ndim != static_cast<py::ssize_t>(expected.size())) {
    std::ostringstream oss;
    oss << name << " must have " << expected.size() << " dimensions, got " << info.ndim;
    throw std::invalid_argument(oss.str());
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    if (info.shape[i] != expected[i]) {
      std::ostringstream oss;
      oss << name << " dimension " << i << " must be " << expected[i] << ", got " << info.shape[i];
      throw std::invalid_argument(oss.str());
    }
  }
}

fastbev::CoreParameter make_core_parameter(const std::string& model_dir, const std::string& model_name) {
  fastbev::pre::NormalizationParameter normalization;
  normalization.image_width = kImageWidth;
  normalization.image_height = kImageHeight;
  normalization.output_width = 704;
  normalization.output_height = 256;
  normalization.num_camera = kNumCameras;
  normalization.resize_lim = 0.44f;
  normalization.interpolation = fastbev::pre::Interpolation::Nearest;

  float mean[3] = {123.675f, 116.28f, 103.53f};
  float std[3] = {58.395f, 57.12f, 57.375f};
  normalization.method = fastbev::pre::NormMethod::mean_std(mean, std, 1.0f, 0.0f);

  fastbev::pre::GeometryParameter geometry;
  geometry.feat_height = 64;
  geometry.feat_width = 176;
  geometry.num_camera = kNumCameras;
  geometry.valid_points = kValidPoints;
  geometry.volum_x = 200;
  geometry.volum_y = 200;
  geometry.volum_z = 4;

  const std::string model_root = join_path(model_dir, model_name);
  fastbev::CoreParameter param;
  param.pre_model = join_path(join_path(model_root, "build"), "fastbev_pre_trt.plan");
  param.post_model = join_path(join_path(model_root, "build"), "fastbev_post_trt_decode.plan");
  param.normalize = normalization;
  param.geo_param = geometry;
  return param;
}

}  // namespace

class FastBEVRuntime {
 public:
  FastBEVRuntime(const std::string& model_dir, const std::string& model_name, const std::string& precision,
                 int device_id)
      : model_dir_(model_dir), model_name_(model_name), precision_(precision), device_id_(device_id) {
    checkRuntime(cudaSetDevice(device_id_));
    checkRuntime(cudaStreamCreate(&stream_));

    auto param = make_core_parameter(model_dir_, model_name_);
    require_file(param.pre_model);
    require_file(param.post_model);

    core_ = fastbev::create_core(param);
    if (core_ == nullptr) {
      throw std::runtime_error("Failed to initialize FastBEV core");
    }
    core_->set_timer(false);
  }

  ~FastBEVRuntime() {
    core_.reset();
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  py::tuple infer(py::array_t<unsigned char, py::array::c_style> images,
                  py::array_t<float, py::array::c_style> valid_c_idx,
                  py::array_t<int64_t, py::array::c_style> valid_x,
                  py::array_t<int64_t, py::array::c_style> valid_y) {
    py::buffer_info images_info = images.request();
    py::buffer_info valid_c_idx_info = valid_c_idx.request();
    py::buffer_info valid_x_info = valid_x.request();
    py::buffer_info valid_y_info = valid_y.request();

    require_shape(images_info, {kNumCameras, kImageHeight, kImageWidth, kImageChannels}, "images");
    require_shape(valid_c_idx_info, {kNumCameras, kValidPoints}, "valid_c_idx");
    require_shape(valid_x_info, {kNumCameras, kValidPoints}, "valid_x");
    require_shape(valid_y_info, {kNumCameras, kValidPoints}, "valid_y");

    const auto* image_base = static_cast<const unsigned char*>(images_info.ptr);
    constexpr size_t image_size = static_cast<size_t>(kImageHeight) * kImageWidth * kImageChannels;
    std::array<const unsigned char*, kNumCameras> image_ptrs;
    for (int i = 0; i < kNumCameras; ++i) {
      image_ptrs[i] = image_base + static_cast<size_t>(i) * image_size;
    }

    std::vector<fastbev::post::transbbox::BoundingBox> detections;
    {
      py::gil_scoped_release release;
      std::lock_guard<std::mutex> guard(mutex_);
      core_->update(static_cast<const float*>(valid_c_idx_info.ptr), static_cast<const int64_t*>(valid_x_info.ptr),
                    static_cast<const int64_t*>(valid_y_info.ptr), stream_);
      detections = core_->forward(image_ptrs.data(), stream_);
      checkRuntime(cudaStreamSynchronize(stream_));
    }

    py::array_t<float> boxes({static_cast<py::ssize_t>(detections.size()), static_cast<py::ssize_t>(8)});
    py::array_t<int32_t> labels({static_cast<py::ssize_t>(detections.size())});

    auto* boxes_data = static_cast<float*>(boxes.request().ptr);
    auto* labels_data = static_cast<int32_t*>(labels.request().ptr);
    for (size_t i = 0; i < detections.size(); ++i) {
      const auto& box = detections[i];
      boxes_data[i * 8 + 0] = box.position.x;
      boxes_data[i * 8 + 1] = box.position.y;
      boxes_data[i * 8 + 2] = box.position.z;
      boxes_data[i * 8 + 3] = box.size.w;
      boxes_data[i * 8 + 4] = box.size.l;
      boxes_data[i * 8 + 5] = box.size.h;
      boxes_data[i * 8 + 6] = box.z_rotation;
      boxes_data[i * 8 + 7] = box.score;
      labels_data[i] = box.id;
    }

    return py::make_tuple(boxes, labels);
  }

  const std::string& model_dir() const { return model_dir_; }
  const std::string& model_name() const { return model_name_; }
  const std::string& precision() const { return precision_; }
  int device_id() const { return device_id_; }

 private:
  std::string model_dir_;
  std::string model_name_;
  std::string precision_;
  int device_id_ = 0;
  cudaStream_t stream_ = nullptr;
  std::shared_ptr<fastbev::Core> core_;
  std::mutex mutex_;
};

PYBIND11_MODULE(fastbev_native, m) {
  m.doc() = "Resident FastBEV CUDA/TensorRT runtime";
  py::class_<FastBEVRuntime>(m, "FastBEVRuntime")
      .def(py::init<const std::string&, const std::string&, const std::string&, int>(), py::arg("model_dir"),
           py::arg("model_name") = "resnet18", py::arg("precision") = "fp16", py::arg("device_id") = 0)
      .def("infer", &FastBEVRuntime::infer, py::arg("images_uint8"), py::arg("valid_c_idx"), py::arg("x"),
           py::arg("y"), "Run FastBEV and return (boxes[N,8] float32, labels[N] int32).")
      .def_property_readonly("model_dir", &FastBEVRuntime::model_dir)
      .def_property_readonly("model_name", &FastBEVRuntime::model_name)
      .def_property_readonly("precision", &FastBEVRuntime::precision)
      .def_property_readonly("device_id", &FastBEVRuntime::device_id);
}
