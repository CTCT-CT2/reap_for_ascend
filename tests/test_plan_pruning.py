import json

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from reap.args import PruneArgs
from reap.model_util import get_moe
from reap.prune import load_pruning_plan, prune


def _make_qwen3_model():
    model = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_experts=3,
            num_experts_per_tok=1,
            norm_topk_prob=False,
        )
    )
    model.eval()
    return model


def test_qwen3_json_plan_prunes_without_observer_data(tmp_path):
    model = _make_qwen3_model()
    retained = [0, 2]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "keep": len(retained),
                "criterion": "reap",
                "layers": {
                    str(layer): retained for layer in range(model.config.num_hidden_layers)
                },
            }
        ),
        encoding="utf-8",
    )

    original_router_weights = {
        layer: get_moe(model, layer).gate.weight.detach().clone()
        for layer in range(model.config.num_hidden_layers)
    }
    retained_by_layer, original_count, retained_count, criterion = load_pruning_plan(
        plan_path, model
    )

    assert original_count == 3
    assert retained_count == 2
    assert criterion == "reap"
    output_dir = tmp_path / "planned-model"
    prune(
        observer_data=None,
        model=model,
        prune_args=PruneArgs(pruning_plan=str(plan_path)),
        n_experts_to_prune=1,
        pruned_model_dir=output_dir,
        retained_experts_by_layer=retained_by_layer,
    )

    assert model.config.num_experts == 2
    for layer in range(model.config.num_hidden_layers):
        moe = get_moe(model, layer)
        assert moe.experts.num_experts == 2
        assert moe.gate.num_experts == 2
        torch.testing.assert_close(
            moe.gate.weight,
            original_router_weights[layer][retained],
        )

    reloaded = Qwen3MoeForCausalLM.from_pretrained(output_dir)
    assert reloaded.config.num_experts == 2
    assert get_moe(reloaded, 0).experts.num_experts == 2
    with torch.no_grad():
        output = reloaded(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert output.logits.shape == (1, 3, 32)
