import copy

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from reap.args import DatasetArgs, LayerwiseArgs, ObserverArgs, PruneArgs
from reap.layerwise_prune import record_activations_layerwise
from reap.main import _setup_observer
from reap.model_util import get_moe
from reap.prune import prune


def _make_model():
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


def _batches():
    return [
        {
            "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        },
        {
            "input_ids": torch.tensor([[6, 7, 8, 9], [10, 11, 12, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        },
    ]


def _standard_observations(model, batches, obs_args):
    observer = _setup_observer(model, obs_args)
    try:
        for batch in batches:
            with observer.set_attention_mask(batch["attention_mask"]):
                model(**batch)
        return observer.report_state()
    finally:
        observer.close_hooks()


def test_qwen_observe_prune_reload(tmp_path):
    torch.manual_seed(0)
    base = _make_model()
    standard_model = copy.deepcopy(base)
    layerwise_model = copy.deepcopy(base)
    obs_args = ObserverArgs(record_pruning_metrics_only=True)

    standard = _standard_observations(standard_model, _batches(), obs_args)
    layerwise = record_activations_layerwise(
        model=layerwise_model,
        tokenizer=None,
        data_batches=_batches(),
        ds_args=DatasetArgs(dataset_name="mock"),
        obs_args=obs_args,
        layerwise_args=LayerwiseArgs(batch_group_size=1),
        results_dir=tmp_path / "observations",
    )
    assert standard.keys() == layerwise.keys()
    for layer in standard:
        assert standard[layer]["reap"].shape == layerwise[layer]["reap"].shape
        assert torch.isfinite(layerwise[layer]["reap"]).all()

    output_dir = tmp_path / "pruned"
    prune(standard, base, PruneArgs(), 1, output_dir)
    assert base.config.num_experts == 2
    assert get_moe(base, 0).experts.num_experts == 2

    reloaded = Qwen3MoeForCausalLM.from_pretrained(output_dir)
    assert reloaded.config.num_experts == 2
    with torch.no_grad():
        result = reloaded(input_ids=torch.tensor([[1, 2, 3]]))
    assert result.logits.shape == (1, 3, 32)
