# 安装与环境

## 推荐方式：容器

宿主机应已安装可用的昇腾驱动和容器运行时。基础镜像通过 Dockerfile 的
`VLLM_ASCEND_IMAGE` 构建参数指定，应根据宿主机驱动和 CANN 版本选择兼容镜像。

```bash
cp .env.template .env
# 在 .env 中设置模型根目录、缓存目录、基础镜像和可见 NPU
docker compose build reap-npu
docker compose run --rm reap-npu bash
```

也可以不使用 Compose，直接从 Dockerfile 构建：

```bash
docker build \
  --build-arg VLLM_ASCEND_IMAGE='<compatible-vllm-ascend-image>' \
  -t '<output-image-name>' .
```

容器启动时会导入 `torch_npu` 并检查 NPU；检查失败时不会继续执行命令。

## 本机 Python 环境

只建议在已有匹配版本 CANN 的机器上使用：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-ascend.txt
python -m pip install -e .
```

不要单独升级 `torch` 或 `torch-npu`。二者与 CANN 版本必须匹配。

## 验证

```bash
python -c 'import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())'
npu-smi info
reap-observe --help
reap-prune --help
```

运行时变量均可在启动前覆盖；项目仅在变量缺失时使用以下默认值：

- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`
- `HCCL_CONNECT_TIMEOUT=1800`
- `HCCL_EXEC_TIMEOUT=1800`
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`
