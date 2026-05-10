ARG TRITON_IMAGE=nvcr.io/nvidia/tritonserver:23.08-py3
FROM ${TRITON_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libopencv-dev \
    libprotobuf-dev \
    protobuf-compiler \
    pybind11-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . /workspace

RUN python3 -m pip install --no-cache-dir \
    -r /workspace/service/requirements.txt \
    -r /workspace/examples/requirements.txt

ENV PYTHONPATH=/workspace/service:/workspace/build
ENV TRITON_URL=localhost:8000
ENV TRITON_MODEL_NAME=fastbev_pipeline
ENV FASTBEV_MODEL_DIR=/workspace/model
ENV FASTBEV_MODEL_NAME=resnet18
ENV FASTBEV_PRECISION=fp16
ENV FASTBEV_NATIVE_PYTHONPATH=/workspace/build

RUN chmod +x /workspace/docker/entrypoint.sh /workspace/docker/build_artifacts.sh

EXPOSE 8080 8000 8001 8002

ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
CMD ["serve"]
