# REAP for Ascend NPU

面向昇腾 NPU 的 MoE 专家观测与剪枝工具。项目使用 `torch_npu` 执行模型，使用
HCCL 支持多卡，并可将裁剪后的 Hugging Face checkpoint 直接交给
vLLM-Ascend 部署。

当前支持 Qwen3 和 Qwen3.5 MoE。建议先使用目标模型的小规模副本完成观测、
裁剪、重载和生成验证，再处理完整 checkpoint。

## 上游项目与修改说明

本项目基于 Cerebras Research 的开源项目
[`CerebrasResearch/reap`](https://github.com/CerebrasResearch/reap) 修改而来。
感谢原项目作者公开 REAP 算法、实现与研究成果。本仓库在其基础上进行了面向
昇腾 NPU 的运行时、观测、剪枝、容器和 vLLM-Ascend 部署适配，并删除了与当前
昇腾版本无关的训练、专家合并和评测代码。上游项目及本项目的使用均受各自许可证
和第三方依赖许可证约束，详细归属说明见 [`NOTICE`](NOTICE)。

## 功能

- 在昇腾 NPU 上采集路由频率、专家激活范数和 REAP 分数
- 按默认 REAP 指标删除每层最低分专家
- 分层观测，降低大模型校准时的 NPU 显存占用
- 根据 JSON 计划流式裁剪 safetensors，无需读取观测 `.pt`
- 生成兼容 Transformers 和 vLLM-Ascend 的 checkpoint
- 默认使用 0–3 号 NPU 启动 OpenAI 兼容推理服务

## 支持环境

- Linux 与可用的昇腾 NPU
- 彼此兼容的昇腾驱动、CANN、PyTorch 和 torch-npu
- Python 3.11+
- 与 Transformers 5.5 API 兼容的 Qwen MoE checkpoint
- 部署时使用与运行环境匹配的 vLLM-Ascend

仓库不假定固定服务器、NPU 型号或模型存储位置。参考依赖组合见
[`requirements-ascend.txt`](requirements-ascend.txt)，容器基础镜像见
[`Dockerfile`](Dockerfile)，实际版本应按昇腾兼容性矩阵选择。

## 快速开始

```bash
cp .env.template .env
docker compose build reap-npu
docker compose run --rm reap-npu bash
```

在容器内执行观测和剪枝：

```bash
export MODEL_DIR='<source-model-path>'
export OUTPUT_DIR='<output-model-path>'
export CALIBRATION_DATASET='<dataset-name-or-local-path>'

reap-prune \
  --model-name "${MODEL_DIR}" \
  --dataset-name "${CALIBRATION_DATASET}" \
  --batches-per-category 128 \
  --batch-size 1 \
  --model-max-length 2048 \
  --prune-method reap \
  --compression-ratio 0.25 \
  --pruned-model-dir "${OUTPUT_DIR}"
```

如果已经有保留专家计划，可跳过观测文件：

```bash
export PLAN_PATH='<pruning-plan-path>'

reap-prune \
  --model-name "${MODEL_DIR}" \
  --pruning-plan "${PLAN_PATH}" \
  --pruned-model-dir "${OUTPUT_DIR}"
```

启动 vLLM-Ascend 服务。脚本默认使用 0–3 号卡，可通过环境变量覆盖：

```bash
export NPU_DEVICES='<visible-device-list>'
export TENSOR_PARALLEL_SIZE='<tensor-parallel-size>'
export PORT='<service-port>'
export SERVED_MODEL_NAME='<served-model-name>'

bash scripts/run_vllm_ascend.sh \
  "${OUTPUT_DIR}" "${PORT}" "${SERVED_MODEL_NAME}"
```

详细说明：

- [安装与环境](docs/installation.md)
- [观测与剪枝](docs/usage.md)
- [vLLM-Ascend 部署](docs/deployment.md)
- [JSON 计划格式](docs/pruning-plan.md)

## 项目结构

```text
src/reap/       观测、REAP 指标、剪枝与 NPU 运行时
scripts/        镜像构建、结果分析与推理启动脚本
tests/          CPU 单元测试和小模型功能测试
docs/           对外使用文档
```

## 测试

```bash
python -m pytest -q
```

单元测试可在 CPU 环境运行；真实模型观测和生成必须在已正确安装 CANN 与
torch-npu 的昇腾环境运行。

## 致谢与引用

REAP 指标源自 *REAP the Experts: Why Pruning Prevails for One-Shot MoE
Compression*，原始实现见
[`CerebrasResearch/reap`](https://github.com/CerebrasResearch/reap)。使用本项目
开展研究时，请同时引用原论文：

```bibtex
@inproceedings{lasby2026reap,
  title={REAP the Experts: Why Pruning Prevails for One-Shot MoE compression},
  author={Mike Lasby and Ivan Lazarevich and Nish Sinnadurai and Sean Lie and Yani Ioannou and Vithursan Thangarasa},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=ukGxWd2aDG}
}
```

## 许可证

本项目使用 [Apache License 2.0](LICENSE)。模型、数据集、CANN、torch-npu 和
vLLM-Ascend 分别受其自身许可证约束。
