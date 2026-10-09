# 观测与剪枝

## 标准观测

```bash
export MODEL_DIR='<source-model-path>'
export OUTPUT_DIR='<output-model-path>'
export CALIBRATION_DATASET='<dataset-name-or-local-path>'
export NPU_DEVICES='<visible-device-list>'

ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES}" reap-observe \
  --model-name "${MODEL_DIR}" \
  --dataset-name "${CALIBRATION_DATASET}" \
  --batches-per-category 128 \
  --batch-size 1 \
  --model-max-length 2048 \
  --output-file-name observations_reap.pt
```

结果写入 `artifacts/<model>/<dataset>/all/`。默认只记录剪枝所需指标，并以
REAP 作为 `reap-prune` 的默认排序标准。

## 观测并剪枝

```bash
ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES}" reap-prune \
  --model-name "${MODEL_DIR}" \
  --dataset-name "${CALIBRATION_DATASET}" \
  --batches-per-category 128 \
  --batch-size 1 \
  --prune-method reap \
  --compression-ratio 0.25 \
  --pruned-model-dir "${OUTPUT_DIR}"
```

`--compression-ratio 0.25` 表示每个 MoE 层裁掉 25% 的专家。也可以用
`--n-experts-to-prune` 指定每层删除数量。

## 分层观测与剪枝

当完整模型观测超过单卡显存时：

```bash
ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES}" reap-layerwise-prune \
  --model-name "${MODEL_DIR}" \
  --dataset-name "${CALIBRATION_DATASET}" \
  --batch-size 1 \
  --batch-group-size 8 \
  --prune-method reap \
  --compression-ratio 0.25
```

该模式将主模型保存在 CPU，每次只把一个 Transformer block 移到 NPU。

## 注意事项

- 校准集应覆盖真实业务输入分布。
- 裁剪后必须重载 checkpoint 并做生成测试。
- 不同模型结构的专家字段不同；未验证架构应先使用小模型测试。
- 输出目录存在时，JSON 流式裁剪会主动退出，防止覆盖已有 checkpoint。
