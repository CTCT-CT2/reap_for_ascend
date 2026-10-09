"""Regression coverage for the Qwen3.5-MoE observer integration."""

import pytest
import torch

qwen35_config = pytest.importorskip(
    "transformers.models.qwen3_5_moe.configuration_qwen3_5_moe"
)
qwen35_modeling = pytest.importorskip(
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
)

from reap.model_util import MODEL_ATTRS, get_moe
from reap.observer import MoETransformerObserver, Qwen35MoEObserverHookConfig


def _tiny_model():
    text_config = {
        "vocab_size": 32,
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "linear_conv_kernel_dim": 2,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "moe_intermediate_size": 4,
        "shared_expert_intermediate_size": 4,
        "num_experts_per_tok": 1,
        "num_experts": 2,
        "layer_types": ["full_attention"],
        "max_position_embeddings": 32,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    vision_config = {
        "depth": 1,
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_heads": 1,
        "patch_size": 2,
        "spatial_merge_size": 1,
        "temporal_patch_size": 1,
        "out_hidden_size": 8,
        "num_position_embeddings": 16,
    }
    config = qwen35_config.Qwen3_5MoeConfig(
        text_config=text_config,
        vision_config=vision_config,
    )
    return qwen35_modeling.Qwen3_5MoeForConditionalGeneration(config).eval()


def test_qwen35_observer_collects_routing_state():
    model = _tiny_model()
    assert model.__class__.__name__ == "Qwen3_5MoeForConditionalGeneration"
    assert MODEL_ATTRS[model.__class__.__name__]["layers"] == "model.language_model.layers"
    assert get_moe(model, 0).__class__.__name__ == "Qwen3_5MoeSparseMoeBlock"

    observer = MoETransformerObserver(
        model,
        Qwen35MoEObserverHookConfig(record_pruning_metrics_only=True),
    )
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([[1, 7, 8]]), use_cache=False)
        state = observer.report_state()[0]
        assert state["total_tokens"].item() == 3
        assert state["expert_frequency"].shape == (2,)
    finally:
        observer.close_hooks()
