from __future__ import annotations
import json
import re
import time
import logging
import pathlib

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser

from accelerate.utils import set_seed

from reap.main import record_activations, smoke_test, create_results_directory, dump_args_to_yaml
from reap.args import (
    ReapArgs,
    ModelArgs,
    PruneArgs,
    ObserverArgs,
    DatasetArgs,
)
from reap.model_util import (
    get_moe,
    MODEL_ATTRS,
    get_super_expert_indices,
    load_reap_model,
)
import shutil
from reap.npu import require_npu

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

QWEN_EXPERT_WEIGHT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(down_proj|gate_proj|up_proj)\.weight$"
)
QWEN_ROUTER_WEIGHT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.weight$")


def prune(
    observer_data,
    model,
    prune_args,
    n_experts_to_prune,
    pruned_model_dir,
    retained_experts_by_layer: dict[int, list[int]] | None = None,
):
    """
    Prune the model from observation scores or an explicit retained-expert plan.
    """
    pruned_model_dir = pathlib.Path(pruned_model_dir)
    model_attrs = MODEL_ATTRS[model.__class__.__name__]

    if retained_experts_by_layer is None:
        for layer in observer_data:
            if "expert_proba" not in observer_data[layer]:
                # Calculate expert probabilities if not already present
                observer_data[layer]["expert_proba"] = (
                    observer_data[layer]["expert_frequency"]
                    / observer_data[layer]["total_tokens"]
                )

    if retained_experts_by_layer is None and (
        prune_args.preserve_super_experts or prune_args.preserve_outliers
    ):
        super_expert_idx = get_super_expert_indices(
            observer_data, include_last_layers=prune_args.preserve_outliers
        )
        metrics = [
            "expert_proba",
            "ean_sum",
            "ean_mean",
            "weighted_expert_frequency_sum",
            "weighted_ean_sum",
            "reap",
            "reap_l2",
            "weighted_ean_sum_l2",
        ]
        for layer in observer_data:
            super_experts_in_layer = super_expert_idx[super_expert_idx[:, 0] == layer][:, 1]
            if len(super_experts_in_layer) > 0:
                for metric in metrics:
                    if metric in observer_data[layer]:
                        observer_data[layer][metric][super_experts_in_layer] = float("inf")

    layers_to_prune = (
        retained_experts_by_layer.keys()
        if retained_experts_by_layer is not None
        else observer_data.keys()
    )
    retained_count = None
    for layer in tqdm(layers_to_prune, "Pruning layers..."):
        if retained_experts_by_layer is not None:
            retained_expert_indices = retained_experts_by_layer[layer]
        else:
            num_experts = observer_data[layer]["expert_frequency"].shape[0]
            if prune_args.prune_method == "ean_ca":
                ean = torch.zeros(num_experts, device=model.device, dtype=torch.float32)
                for i in range(num_experts):
                    ean[i] = torch.linalg.norm(
                        observer_data[layer]["routed_characteristic_activation"][i], dim=-1
                    ).sum()
                _, experts_to_prune = torch.topk(
                    ean, n_experts_to_prune, largest=False
                )
            else:
                prune_method = prune_args.prune_method
                if prune_method == "frequency":
                    prune_method = "expert_frequency"
                saliency_data = observer_data[layer].get(prune_method)
                if saliency_data is None:
                    raise ValueError(
                        f"Prune method {prune_args.prune_method} not found in observer data for layer {layer}. "
                        f"Available keys: {list(observer_data[layer].keys())}"
                    )
                _, experts_to_prune = torch.topk(
                    saliency_data, n_experts_to_prune, largest=False
                )

            retained_expert_indices = [
                i for i in range(num_experts) if i not in experts_to_prune
            ]
        retained_count = len(retained_expert_indices)
        # prune experts
        moe = get_moe(model, layer)
        all_experts = getattr(moe, model_attrs["experts"])
        fused_experts = model_attrs["fused"] or not isinstance(
            all_experts, torch.nn.ModuleList
        )
        if not fused_experts:
            retained_experts = [all_experts[i] for i in retained_expert_indices]
            retained_experts = torch.nn.ModuleList(retained_experts)
            setattr(moe, model_attrs["experts"], retained_experts)
            # prune router
            router = getattr(moe, model_attrs["router"])
            router.weight.data = router.weight.data[retained_expert_indices, :]
            if getattr(router, "bias", None):
                router.bias.data = router.bias.data[retained_expert_indices]
            router.out_features = len(retained_expert_indices)
            if hasattr(router, "e_score_correction_bias"):
                router.e_score_correction_bias.data = (
                    router.e_score_correction_bias.data[retained_expert_indices]
                )
            setattr(moe, model_attrs["router"], router)
        else:
            # Newer Transformers releases fuse Qwen3 experts at load time.
            all_experts.gate_up_proj.data = all_experts.gate_up_proj[
                retained_expert_indices
            ]
            all_experts.down_proj.data = all_experts.down_proj[retained_expert_indices]
            if hasattr(all_experts, "num_experts"):
                all_experts.num_experts = len(retained_expert_indices)
            router = getattr(moe, model_attrs["router"])
            router.weight.data = router.weight.data[retained_expert_indices]
            if hasattr(router, "out_features"):
                router.out_features = len(retained_expert_indices)
            if hasattr(router, "num_experts"):
                router.num_experts = len(retained_expert_indices)

    # patch config and dump
    logger.info("Saving pruned model...")
    if retained_count is None:
        raise ValueError("No MoE layers were selected for pruning")
    setattr(model.config, model_attrs["num_experts"], retained_count)
    pruned_model_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    model.save_pretrained(pruned_model_dir)
    end = time.time()
    logger.info(
        f"Pruned model saved to {pruned_model_dir} in {end - start:.2f} seconds"
    )
    return pruned_model_dir


def load_pruning_plan(
    plan_path: str | pathlib.Path,
    model,
) -> tuple[dict[int, list[int]], int, int, str]:
    """Load and validate a kept-expert pruning plan against an unpruned model."""
    plan_path = pathlib.Path(plan_path)
    with plan_path.open(encoding="utf-8") as f:
        plan = json.load(f)

    if not isinstance(plan, dict) or not isinstance(plan.get("layers"), dict):
        raise ValueError("Pruning plan must contain a JSON object named 'layers'")
    if not plan["layers"]:
        raise ValueError("Pruning plan contains no layers")

    try:
        retained_by_layer = {
            int(layer): list(indices) for layer, indices in plan["layers"].items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("Plan layer keys must be integer strings") from exc

    configured_layers = getattr(model.config, "num_hidden_layers", None)
    if configured_layers is not None:
        invalid_layers = [
            layer for layer in retained_by_layer if layer < 0 or layer >= configured_layers
        ]
        if invalid_layers:
            raise ValueError(f"Plan contains out-of-range layers: {invalid_layers}")

    expected_keep = plan.get("keep")
    original_counts = set()
    retained_counts = set()
    for layer, retained_indices in sorted(retained_by_layer.items()):
        if not all(isinstance(index, int) and not isinstance(index, bool) for index in retained_indices):
            raise ValueError(f"Layer {layer} contains a non-integer expert index")
        if retained_indices != sorted(set(retained_indices)):
            raise ValueError(
                f"Layer {layer} expert indices must be unique and strictly increasing"
            )

        moe = get_moe(model, layer)
        model_attrs = MODEL_ATTRS[model.__class__.__name__]
        experts = getattr(moe, model_attrs["experts"])
        original_count = (
            len(experts)
            if isinstance(experts, torch.nn.ModuleList)
            else experts.num_experts
        )
        if not retained_indices or retained_indices[-1] >= original_count:
            raise ValueError(
                f"Layer {layer} has an expert index outside [0, {original_count - 1}]"
            )
        original_counts.add(original_count)
        retained_counts.add(len(retained_indices))

    if len(original_counts) != 1 or len(retained_counts) != 1:
        raise ValueError("All planned layers must use uniform original and retained expert counts")
    original_count = original_counts.pop()
    retained_count = retained_counts.pop()
    if expected_keep is not None and expected_keep != retained_count:
        raise ValueError(
            f"Plan keep={expected_keep} does not match {retained_count} listed experts per layer"
        )

    expected_layers = set(range(getattr(model.config, "num_hidden_layers", 0)))
    if expected_layers and set(retained_by_layer) != expected_layers:
        missing = sorted(expected_layers - set(retained_by_layer))
        extra = sorted(set(retained_by_layer) - expected_layers)
        raise ValueError(
            f"Plan layers do not match model layers; missing={missing}, extra={extra}"
        )

    criterion = str(plan.get("criterion", "plan"))
    return retained_by_layer, original_count, retained_count, criterion


def load_checkpoint_pruning_plan(
    plan_path: str | pathlib.Path,
    model_dir: str | pathlib.Path,
) -> tuple[dict[int, list[int]], int, int, str]:
    """Validate a kept-expert plan using only the checkpoint config."""
    plan_path = pathlib.Path(plan_path)
    model_dir = pathlib.Path(model_dir)
    with plan_path.open(encoding="utf-8") as f:
        plan = json.load(f)
    with (model_dir / "config.json").open(encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(plan, dict) or not isinstance(plan.get("layers"), dict):
        raise ValueError("Pruning plan must contain a JSON object named 'layers'")
    num_layers = config.get("num_hidden_layers")
    num_experts = config.get("num_experts")
    if not isinstance(num_layers, int) or not isinstance(num_experts, int):
        raise ValueError("Model config must define integer num_hidden_layers and num_experts")

    try:
        retained_by_layer = {
            int(layer): list(indices) for layer, indices in plan["layers"].items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("Plan layer keys must be integer strings") from exc
    expected_layers = set(range(num_layers))
    if set(retained_by_layer) != expected_layers:
        missing = sorted(expected_layers - set(retained_by_layer))
        extra = sorted(set(retained_by_layer) - expected_layers)
        raise ValueError(
            f"Plan layers do not match model layers; missing={missing}, extra={extra}"
        )

    retained_counts = set()
    for layer, retained_indices in sorted(retained_by_layer.items()):
        if not all(
            isinstance(index, int) and not isinstance(index, bool)
            for index in retained_indices
        ):
            raise ValueError(f"Layer {layer} contains a non-integer expert index")
        if retained_indices != sorted(set(retained_indices)):
            raise ValueError(
                f"Layer {layer} expert indices must be unique and strictly increasing"
            )
        if not retained_indices or retained_indices[-1] >= num_experts:
            raise ValueError(
                f"Layer {layer} has an expert index outside [0, {num_experts - 1}]"
            )
        retained_counts.add(len(retained_indices))
    if len(retained_counts) != 1:
        raise ValueError("All planned layers must retain the same number of experts")
    retained_count = retained_counts.pop()
    if plan.get("keep") is not None and plan["keep"] != retained_count:
        raise ValueError(
            f"Plan keep={plan['keep']} does not match {retained_count} listed experts per layer"
        )
    return retained_by_layer, num_experts, retained_count, str(
        plan.get("criterion", "plan")
    )


def prune_safetensors_checkpoint_from_plan(
    model_dir: str | pathlib.Path,
    plan_path: str | pathlib.Path,
    output_dir: str | pathlib.Path,
) -> pathlib.Path:
    """Stream-prune a sharded Qwen3 checkpoint while preserving vLLM weight names."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    model_dir = pathlib.Path(model_dir).resolve()
    plan_path = pathlib.Path(plan_path).resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    retained_by_layer, original_count, retained_count, criterion = (
        load_checkpoint_pruning_plan(plan_path, model_dir)
    )

    index_path = model_dir / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as f:
        source_index = json.load(f)
    shard_names = sorted(set(source_index["weight_map"].values()))
    staging_dir = output_dir.with_name(f"{output_dir.name}.incomplete")
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError(
            f"Output or staging directory already exists: {output_dir}, {staging_dir}"
        )
    staging_dir.mkdir(parents=True)

    retained_positions = {
        layer: {original: new for new, original in enumerate(indices)}
        for layer, indices in retained_by_layer.items()
    }
    output_weight_map = {}
    total_size = 0
    try:
        for shard_number, shard_name in enumerate(shard_names, start=1):
            source_shard = model_dir / shard_name
            output_tensors = {}
            with safe_open(source_shard, framework="pt", device="cpu") as shard:
                metadata = shard.metadata()
                for key in shard.keys():
                    output_key = key
                    expert_match = QWEN_EXPERT_WEIGHT_RE.match(key)
                    if expert_match:
                        layer = int(expert_match.group(1))
                        expert = int(expert_match.group(2))
                        new_expert = retained_positions[layer].get(expert)
                        if new_expert is None:
                            continue
                        output_key = key.replace(
                            f".experts.{expert}.", f".experts.{new_expert}."
                        )
                        tensor = shard.get_tensor(key)
                    else:
                        tensor = shard.get_tensor(key)
                        router_match = QWEN_ROUTER_WEIGHT_RE.match(key)
                        if router_match:
                            layer = int(router_match.group(1))
                            tensor = tensor[retained_by_layer[layer]].contiguous()
                    output_tensors[output_key] = tensor
                    output_weight_map[output_key] = shard_name
                    total_size += tensor.numel() * tensor.element_size()
            save_file(output_tensors, staging_dir / shard_name, metadata=metadata)
            logger.info(
                "Wrote pruned shard %d/%d: %s",
                shard_number,
                len(shard_names),
                shard_name,
            )

        for source_file in model_dir.iterdir():
            if source_file.is_file() and not source_file.name.endswith(".safetensors"):
                if source_file.name not in {"config.json", "model.safetensors.index.json"}:
                    shutil.copy2(source_file, staging_dir / source_file.name)

        with (model_dir / "config.json").open(encoding="utf-8") as f:
            config = json.load(f)
        config["num_experts"] = retained_count
        with (staging_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            f.write("\n")

        output_index = dict(source_index)
        output_index["weight_map"] = output_weight_map
        output_index["metadata"] = dict(source_index.get("metadata", {}))
        output_index["metadata"]["total_size"] = total_size
        with (staging_dir / "model.safetensors.index.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(output_index, f, ensure_ascii=False, indent=2)
            f.write("\n")
        shutil.copy2(plan_path, staging_dir / "pruning_plan.json")
        with (staging_dir / "pruning_metadata.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "source_model": str(model_dir),
                    "source_plan": str(plan_path),
                    "criterion": criterion,
                    "original_experts_per_layer": original_count,
                    "retained_experts_per_layer": retained_count,
                    "pruned_experts_per_layer": original_count - retained_count,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
            f.write("\n")
        staging_dir.rename(output_dir)
    except Exception:
        logger.exception("Checkpoint pruning failed; partial output remains at %s", staging_dir)
        raise

    logger.info("Pruned checkpoint saved to %s", output_dir)
    return output_dir


def get_pruned_model_dir(
    results_dir: pathlib.Path,
    n_experts_to_prune: int,
    total_experts: int,
    prune_args: PruneArgs,
    seed: int,
    renorm: bool,
    name_prefix: str = None,
) -> pathlib.Path:
    """Generate output directory path for pruned model."""
    compression_ratio_str = f"{(n_experts_to_prune / total_experts):.2f}"
    name_prefix = "" if name_prefix is None else name_prefix
    pruned_model_name = f"{name_prefix}{prune_args.prune_method}"

    if prune_args.preserve_super_experts:
        pruned_model_name += "-preserve_super"
    elif prune_args.preserve_outliers:
        pruned_model_name += "-preserve_outlier"
    if renorm:
        pruned_model_name += f"-renorm_{str(renorm).lower()}"
    pruned_model_name += f"-seed_{seed}"
    pruned_model_name += f"-{compression_ratio_str}"

    pruned_model_dir = results_dir / "pruned_models" / pruned_model_name
    logger.info(f"Using seed {seed}, pruned model dir: {pruned_model_dir}")

    return pruned_model_dir


def main():
    parser = HfArgumentParser(
        (
            ReapArgs,
            DatasetArgs,
            ObserverArgs,
            ModelArgs,
            PruneArgs,
        )
    )
    reap_args, ds_args, obs_args, model_args, prune_args = (
        parser.parse_args_into_dataclasses()
    )
    if prune_args.preserve_super_experts and prune_args.preserve_outliers:
        raise ValueError(
            "Only one of preserve_super_experts or preserve_outliers can be true."
        )
    if prune_args.pruning_plan and (
        prune_args.preserve_super_experts or prune_args.preserve_outliers
    ):
        raise ValueError(
            "A pruning plan already fixes the retained experts; preserve options cannot be combined with it."
        )
    # JSON checkpoint pruning is CPU streaming and intentionally does not need
    # an accelerator. Observation-driven pruning executes the model on NPU.
    if not prune_args.pruning_plan:
        require_npu()
    set_seed(reap_args.seed)
    results_dir = create_results_directory(model_args.model_name, ds_args.dataset_name)

    # get local patched model if req'd
    model_name = model_args.model_name
    if prune_args.pruning_plan:
        if reap_args.run_observer_only:
            raise ValueError("--run-observer-only cannot be used with --pruning-plan")
        if prune_args.overwrite_pruned_model:
            raise ValueError(
                "Atomic plan pruning does not overwrite an existing directory; choose a new --pruned-model-dir"
            )
        if prune_args.pruned_model_dir:
            pruned_model_dir = pathlib.Path(prune_args.pruned_model_dir)
        else:
            pruned_model_dir = (
                results_dir
                / "pruned_models"
                / pathlib.Path(prune_args.pruning_plan).stem
            )
        prune_safetensors_checkpoint_from_plan(
            model_dir=model_name,
            plan_path=prune_args.pruning_plan,
            output_dir=pruned_model_dir,
        )
        dump_args_to_yaml(
            pruned_model_dir,
            reap_args=reap_args,
            ds_args=ds_args,
            obs_args=obs_args,
            model_args=model_args,
            prune_args=prune_args,
        )
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # load model
    model = load_reap_model(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    observer_data = None
    logger.info(
        "Running observer for model %s on dataset %s",
        model_args.model_name,
        ds_args.dataset_name,
    )
    observer_data = record_activations(
        model, tokenizer, reap_args, model_args, ds_args, obs_args, results_dir
    )
    if reap_args.run_observer_only:
        logger.info("Observation completed")
        return

    total_experts = len(observer_data[next(iter(observer_data))]["expert_frequency"])
    n_experts_to_prune = prune_args.n_experts_to_prune
    if n_experts_to_prune is None:
        n_experts_to_prune = int(total_experts * prune_args.compression_ratio)
    if not 0 < n_experts_to_prune < total_experts:
        raise ValueError(
            f"Experts to prune must be between 1 and {total_experts - 1}; "
            f"got {n_experts_to_prune}"
        )

    logger.info("Start of pruning")
    if prune_args.pruned_model_dir:
        pruned_model_dir = pathlib.Path(prune_args.pruned_model_dir)
        logger.info("Using explicit pruned model directory: %s", pruned_model_dir)
    else:
        pruned_model_dir = get_pruned_model_dir(
            results_dir,
            n_experts_to_prune,
            total_experts,
            prune_args,
            reap_args.seed,
            obs_args.renormalize_router_weights,
        )
    if (
        pruned_model_dir.exists()
        and list(pruned_model_dir.glob("*.safetensors"))
        and not prune_args.overwrite_pruned_model
    ):
        logger.info(
            f"Pruned model directory {pruned_model_dir} already exists and contains pruned model files. "
            "Skipping pruning step."
        )
    else:
        logger.info(f"Pruning model to {total_experts - n_experts_to_prune} experts...")
        prune(
            observer_data,
            model,
            prune_args,
            n_experts_to_prune,
            pruned_model_dir,
        )
        logger.info("pruning completed.")

        # smoke test
        if reap_args.smoke_test:
            logger.info("Running smoke test on the pruned model...")
            smoke_test(model, tokenizer)

        tokenizer.save_pretrained(pruned_model_dir)
        logger.info("Pruning completed.")

        dump_args_to_yaml(
            pruned_model_dir,
            reap_args=reap_args,
            ds_args=ds_args,
            obs_args=obs_args,
            model_args=model_args,
            prune_args=prune_args,
        )


if __name__ == "__main__":
    main()
