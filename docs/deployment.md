# vLLM-Ascend 部署

## 启动服务

启动参数通过环境变量提供。脚本为了开箱即用，未设置时默认使用 0–3 号 NPU
和四路张量并行；这两个值不代表运行环境要求。

```bash
export MODEL_DIR='<pruned-model-path>'
export NPU_DEVICES='<visible-device-list>'
export TENSOR_PARALLEL_SIZE='<tensor-parallel-size>'
export PORT='<service-port>'
export SERVED_MODEL_NAME='<served-model-name>'

bash scripts/run_vllm_ascend.sh \
  "${MODEL_DIR}" "${PORT}" "${SERVED_MODEL_NAME}"
```

可通过环境变量调整容器名、上下文长度和 NPU 内存利用率：

```bash
REAP_CONTAINER_NAME=reap-api \
MAX_MODEL_LEN=8192 \
NPU_MEMORY_UTILIZATION=0.85 \
bash scripts/run_vllm_ascend.sh \
  "${MODEL_DIR}" "${PORT}" "${SERVED_MODEL_NAME}"
```

## 健康检查

```bash
export API_BASE_URL='<openai-compatible-api-base-url>'
curl "${API_BASE_URL}/v1/models"
```

## 简单生成测试

```bash
curl "${API_BASE_URL}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d @- <<JSON
  {
    "model": "${SERVED_MODEL_NAME}",
    "messages": [
      {"role": "user", "content": "Hello"},
      {"role": "user", "content": "What is 12 + 30?"}
    ],
    "temperature": 0,
    "max_tokens": 64
  }
JSON
```

服务应返回非空文本，并正确回答简单算术。生产部署还应补充并发、长上下文和
业务数据回归测试。
