"""Model loading and MoE structure helpers for supported Qwen checkpoints."""

from functools import reduce
import logging

import torch

logger = logging.getLogger(__name__)


def load_reap_model(model_name: str, **kwargs):
    """Load a Qwen causal or multimodal MoE checkpoint."""
    from transformers import AutoConfig, AutoModelForCausalLM

    kwargs.setdefault("attn_implementation", "sdpa")
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=kwargs.get("trust_remote_code", True),
        local_files_only=kwargs.get("local_files_only", False),
    )
    architectures = getattr(config, "architectures", None) or []
    if any(name.endswith("ForConditionalGeneration") for name in architectures):
        from transformers import AutoModelForImageTextToText

        model_class = AutoModelForImageTextToText
    else:
        model_class = AutoModelForCausalLM
    return model_class.from_pretrained(model_name, **kwargs)


MODEL_ATTRS = {
    "Qwen3_5MoeForConditionalGeneration": {
        "layers": "model.language_model.layers",
        "moe_block": "mlp",
        "experts": "experts",
        "fused": True,
        "router": "gate",
        "num_experts": "experts.num_experts",
        "num_experts_per_tok": "gate.top_k",
    },
    "Qwen3_5MoeForCausalLM": {
        "layers": "model.layers",
        "moe_block": "mlp",
        "experts": "experts",
        "fused": True,
        "router": "gate",
        "num_experts": "experts.num_experts",
        "num_experts_per_tok": "gate.top_k",
    },
    "Qwen3MoeForCausalLM": {
        "layers": "model.layers",
        "moe_block": "mlp",
        "experts": "experts",
        "fused": True,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
}


def get_moe(model, layer: int):
    model_attrs = MODEL_ATTRS.get(model.__class__.__name__)
    if model_attrs is None:
        raise KeyError(f"Unsupported model architecture: {model.__class__.__name__}")
    layers = reduce(getattr, model_attrs["layers"].split("."), model)
    return getattr(layers[layer], model_attrs["moe_block"])


def get_super_expert_indices(observer_data, include_last_layers: bool = False):
    """Return unusually high-activation expert coordinates."""
    all_max = [layer["max_activations"] for layer in observer_data.values()]
    num_layers = len(all_max)
    flattened = torch.cat(all_max).flatten()
    threshold = max(
        torch.quantile(flattened, 0.995).item(),
        flattened.max().item() / 10,
    )
    mask = flattened.reshape(num_layers, -1) > threshold
    if not include_last_layers:
        mask[int(num_layers * 0.75) :, :] = False
    indices = torch.argwhere(mask)
    logger.info("Preserving %d high-activation experts", indices.shape[0])
    return indices
