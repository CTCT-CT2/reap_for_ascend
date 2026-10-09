import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from reap.layerwise_model_utils import extract_model_components, is_decoder_block
from reap.layerwise_observer import LayerwiseMoEObserver
from reap.observer import Qwen3MoEObserverHookConfig


def _make_model():
    model = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=8,
            moe_intermediate_size=8,
            num_hidden_layers=3,
            num_attention_heads=1,
            num_key_value_heads=1,
            num_experts=2,
            num_experts_per_tok=1,
            norm_topk_prob=False,
        )
    )
    model.eval()
    return model


def test_decoder_blocks_and_components():
    model = _make_model()
    block_names = [
        name for name, module in model.named_modules() if is_decoder_block(name, module)
    ]
    assert block_names == ["model.layers.0", "model.layers.1", "model.layers.2"]

    blocks, non_backbone = extract_model_components(model, block_names)
    assert len(blocks) == 3
    assert "model.embed_tokens" in non_backbone
    assert "model.norm" in non_backbone
    assert "lm_head" in non_backbone


def test_layerwise_observer_finds_qwen_moe_blocks():
    model = _make_model()
    observer = LayerwiseMoEObserver(model, Qwen3MoEObserverHookConfig())
    try:
        for block_index in range(3):
            module = observer._find_moe_module_in_block(block_index)
            assert module is model.model.layers[block_index].mlp
    finally:
        observer.close_hooks()


def test_cpu_cleanup_is_safe():
    model = _make_model()
    assert all(parameter.device == torch.device("cpu") for parameter in model.parameters())
