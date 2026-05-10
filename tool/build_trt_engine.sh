#!/bin/bash
# configure the environment
. tool/environment.sh

if [ "$ConfigurationStatus" != "Success" ]; then
    echo "Exit due to configure failure."
    exit
fi

# tensorrt version
# version=`trtexec | grep -m 1 TensorRT | sed -n "s/.*\[TensorRT v\([0-9]*\)\].*/\1/p"`

# resnet18/resnet18int8/resnet18int8head
base=model/$DEBUG_MODEL

# fp16/int8
precision=$DEBUG_PRECISION
trt_hardware_compatibility=${TRT_HARDWARE_COMPATIBILITY:-ampere+}
trt_force_rebuild=${TRT_FORCE_REBUILD:-0}

# precision flags
trtexec_fp16_flags="--fp16"
trtexec_dynamic_flags="--fp16"
if [ "$precision" == "int8" ]; then
    trtexec_dynamic_flags="--fp16 --int8"
fi

function get_onnx_number_io(){

    # $1=model
    model=$1

    if [ ! -f "$model" ]; then
        echo The model [$model] not exists.
        return
    fi

    number_of_input=`python -c "import onnx;m=onnx.load('$model');print(len(m.graph.input), end='')"`
    number_of_output=`python -c "import onnx;m=onnx.load('$model');print(len(m.graph.output), end='')"`
    # echo The model [$model] has $number_of_input inputs and $number_of_output outputs.
}

function compile_trt_model(){

    # $1: name
    # $2: precision_flags
    # $3: number_of_input
    # $4: number_of_output
    name=$1
    precision_flags=$2
    number_of_input=$3
    number_of_output=$4
    need_output_flg=$5
    result_save_directory=$base/build
    onnx=$base/$name.onnx
    plan_file=${result_save_directory}/$name.plan
    marker_file=${result_save_directory}/$name.plan.meta
    expected_marker="precision=${precision};hardwareCompatibilityLevel=${trt_hardware_compatibility};trt=${TensorRT_Lib}"

    if [ -f "${plan_file}" ] && [ "$trt_force_rebuild" != "1" ] && [ -f "${marker_file}" ] && grep -qxF "$expected_marker" "${marker_file}"; then
        echo "Model ${plan_file} already build for ${trt_hardware_compatibility} 🙋🙋🙋."
        return
    fi

    if [ -f "${plan_file}" ]; then
        echo "Rebuilding ${plan_file}; TensorRT compatibility metadata is missing or changed."
        rm -f "${plan_file}" "${marker_file}"
    fi
    
    # Remove the onnx dependency
    # get_onnx_number_io $onnx
    echo $number_of_input  $number_of_output

    input_formats=""
    output_formats=""
    for i in $(seq 1 $number_of_input); do
        if [ -n "$input_formats" ]; then
            input_formats+=","
        fi
        input_formats+="fp16:chw"
    done

    for i in $(seq 1 $number_of_output); do
        if [ -n "$output_formats" ]; then
            output_formats+=","
        fi
        output_formats+="fp16:chw"
    done
    input_flags="--inputIOFormats=${input_formats}"
    output_flags="--outputIOFormats=${output_formats}"
    hardware_flags=""
    if [ -n "$trt_hardware_compatibility" ] && [ "$trt_hardware_compatibility" != "none" ]; then
        hardware_flags="--hardwareCompatibilityLevel=${trt_hardware_compatibility}"
    fi

    if [ "$need_output_flg" == "need" ]; then
        cmd="--onnx=$base/$name.onnx ${precision_flags} ${input_flags} ${output_flags} \
            ${hardware_flags} \
            --saveEngine=${result_save_directory}/$name.plan \
            --memPoolSize=workspace:2048 --verbose --dumpLayerInfo \
            --dumpProfile --separateProfileRun \
            --skipInference --profilingVerbosity=detailed --exportLayerInfo=${result_save_directory}/$name.json"
    else
        cmd="--onnx=$base/$name.onnx ${precision_flags} ${input_flags} \
            ${hardware_flags} \
            --saveEngine=${result_save_directory}/$name.plan \
            --memPoolSize=workspace:2048 --verbose --dumpLayerInfo \
            --dumpProfile --separateProfileRun \
            --skipInference --profilingVerbosity=detailed --exportLayerInfo=${result_save_directory}/$name.json"
    fi
    echo $cmd
    mkdir -p $result_save_directory
    echo Building the model: ${result_save_directory}/$name.plan, this will take several minutes. Wait a moment 🤗🤗🤗~.
    trtexec $cmd > ${result_save_directory}/$name.log 2>&1
    echo "$expected_marker" > "${marker_file}"
}

# maybe int8 / fp16
compile_trt_model "fastbev_pre_trt" "$trtexec_dynamic_flags"  1 1 "need"
compile_trt_model "fastbev_post_trt_decode" "$trtexec_dynamic_flags"  1 3 "noneed"
