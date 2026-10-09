# JSON 剪枝计划

JSON 计划直接描述每层保留的专家，不读取 `.pt` 观测文件。示例：

```json
{
  "criterion": "reap",
  "keep": 3,
  "layers": {
    "0": [0, 1, 3],
    "1": [0, 2, 3]
  }
}
```

约束如下：

- `layers` 必须覆盖模型的全部 MoE 层。
- 专家编号必须为不重复、严格递增的整数。
- 每层保留的专家数量必须相同，并与可选的 `keep` 一致。
- Qwen3 safetensors checkpoint 需要包含 `config.json` 和
  `model.safetensors.index.json`。

执行：

```bash
export MODEL_DIR='<source-model-path>'
export PLAN_PATH='<pruning-plan-path>'
export OUTPUT_DIR='<output-model-path>'

reap-prune \
  --model-name "${MODEL_DIR}" \
  --pruning-plan "${PLAN_PATH}" \
  --pruned-model-dir "${OUTPUT_DIR}"
```

该路径在 CPU 上逐 shard 处理权重，原子生成输出目录，并写入
`pruning_plan.json`、`pruning_metadata.json` 和 `reap_args.yaml`。
