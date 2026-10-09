#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 MODEL_DIR [PORT] [SERVED_MODEL_NAME]" >&2
    exit 2
fi

model_dir=$(realpath "$1")
port=${2:-8000}
served_model_name=${3:-reap-model}
image_name=${REAP_IMAGE_NAME:-reap-ascend:latest}
container_name=${REAP_CONTAINER_NAME:-reap-vllm-ascend}
npu_devices=${NPU_DEVICES:-0,1,2,3}
tensor_parallel_size=${TENSOR_PARALLEL_SIZE:-4}

if [[ ! -f "${model_dir}/config.json" ]]; then
    echo "Model config not found: ${model_dir}/config.json" >&2
    exit 2
fi

docker run --rm \
    --name "${container_name}" \
    --privileged \
    --security-opt label=disable \
    --ipc host \
    --shm-size 64g \
    --network host \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v /etc/vnpu.cfg:/etc/vnpu.cfg \
    -v "${model_dir}:/models/reap-model:ro" \
    -e ASCEND_RT_VISIBLE_DEVICES="${npu_devices}" \
    "${image_name}" \
    vllm serve /models/reap-model \
        --served-model-name "${served_model_name}" \
        --tensor-parallel-size "${tensor_parallel_size}" \
        --max-model-len "${MAX_MODEL_LEN:-4096}" \
        --gpu-memory-utilization "${NPU_MEMORY_UTILIZATION:-0.80}" \
        --port "${port}" \
        --trust-remote-code \
        --enforce-eager
