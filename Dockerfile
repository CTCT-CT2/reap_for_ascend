ARG VLLM_ASCEND_IMAGE=quay.io/ascend/vllm-ascend:v0.23.0-openeuler
FROM ${VLLM_ASCEND_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PYTHONPATH=/workspace/reap/src \
    PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
    HCCL_CONNECT_TIMEOUT=1800 \
    HCCL_EXEC_TIMEOUT=1800 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

WORKDIR /workspace/reap

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY scripts ./scripts
COPY docs ./docs
COPY entrypoint.sh ./entrypoint.sh

# torch, torch-npu, CANN bindings, vLLM and vLLM-Ascend are supplied by the
# version-matched base image. Never replace those accelerator packages here.
RUN python -m pip install --no-cache-dir -e .

ENTRYPOINT ["/bin/bash", "/workspace/reap/entrypoint.sh"]
CMD ["bash"]
