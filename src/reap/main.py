"""Ascend NPU observation entry point."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import pathlib
import re

import torch
import torch.nn as nn
import yaml
from accelerate.utils import set_seed
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser

from reap.args import DatasetArgs, ModelArgs, ObserverArgs, ReapArgs
from reap.data import load_category_batches, parse_composite_dataset_spec
from reap.model_util import load_reap_model
from reap.npu import require_npu
from reap.observer import OBSERVER_CONFIG_REGISTRY, MoETransformerObserver

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def str_to_directory_name(value: str) -> str:
    return re.sub(r"[^\w\-_.]", "_", value)


def create_results_directory(model_name: str, dataset_name: str) -> pathlib.Path:
    model_part = str_to_directory_name(model_name.rstrip("/").split("/")[-1])
    if "," in dataset_name:
        dataset_part = f"composite_{hashlib.sha256(dataset_name.encode()).hexdigest()[:8]}"
    else:
        dataset_part = str_to_directory_name(dataset_name.rstrip("/").split("/")[-1])
    output = pathlib.Path("artifacts") / model_part / dataset_part
    output.mkdir(parents=True, exist_ok=True)
    return output


def _setup_observer(model, obs_args: ObserverArgs) -> MoETransformerObserver:
    config_class = OBSERVER_CONFIG_REGISTRY.get(model.__class__.__name__)
    if config_class is None:
        raise ValueError(
            f"Unsupported model architecture {model.__class__.__name__}; "
            f"supported architectures: {sorted(OBSERVER_CONFIG_REGISTRY)}"
        )
    config = config_class(
        distance_measure=obs_args.distance_measure,
        renormalize_router_weights=(
            getattr(model.config, "norm_topk_prob", False)
            and obs_args.renormalize_router_weights
        ),
        record_pruning_metrics_only=obs_args.record_pruning_metrics_only,
    )
    return MoETransformerObserver(model=model, hook_config=config)


def _profile_model(model, tokenizer, model_args, obs_args, observer) -> None:
    max_length = obs_args.model_max_length or tokenizer.model_max_length
    tokenized = tokenizer(
        ["hello " * max_length],
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    tokenized = {key: value.to(model.device) for key, value in tokenized.items()}
    with torch.no_grad():
        model(**tokenized)
    logger.info("Profiled %s at sequence length %d", model_args.model_name, max_length)
    observer.reset()


def record_activations(
    model, tokenizer, reap_args, model_args, ds_args, obs_args, results_dir
):
    if ds_args.dataset_name == "combined":
        path = results_dir / "all" / obs_args.output_file_name
        if not path.exists():
            raise FileNotFoundError(f"Cached observation file not found: {path}")
        return torch.load(path, weights_only=False)

    components = parse_composite_dataset_spec(
        ds_args.dataset_name, default_split=ds_args.split
    )
    if components is None:
        category_batches = load_category_batches(
            dataset_name=ds_args.dataset_name,
            split=ds_args.split,
            subset=ds_args.dataset_config_name,
            tokenizer=tokenizer,
            model_max_length=obs_args.model_max_length,
            split_by_category=obs_args.split_by_category,
            return_vllm_tokens_prompt=obs_args.return_vllm_tokens_prompt,
            truncate=obs_args.truncate,
            batches_per_category=obs_args.batches_per_category,
            batch_size=obs_args.batch_size,
        )
    else:
        batches = []
        for component in components:
            loaded = load_category_batches(
                dataset_name=component.name,
                split=component.split,
                subset=component.subset,
                tokenizer=tokenizer,
                model_max_length=obs_args.model_max_length,
                split_by_category=False,
                return_vllm_tokens_prompt=obs_args.return_vllm_tokens_prompt,
                truncate=obs_args.truncate,
                batches_per_category=component.num_batches,
                batch_size=obs_args.batch_size,
            )
            batches.extend(loaded["all"])
        category_batches = {"all": batches}

    observer = _setup_observer(model, obs_args)
    try:
        if reap_args.profile:
            _profile_model(model, tokenizer, model_args, obs_args, observer)
        last_path = None
        for category, batches in category_batches.items():
            output_path = (
                results_dir / str_to_directory_name(category) / obs_args.output_file_name
            )
            last_path = output_path
            if output_path.exists() and not obs_args.overwrite_observations:
                logger.info("Loading existing observations from %s", output_path)
                return torch.load(output_path, weights_only=False)
            for sample in tqdm(batches, desc=f"Observing {category}"):
                attention_mask = sample.get("attention_mask")
                sample = {
                    key: value.to(model.device) if torch.is_tensor(value) else value
                    for key, value in sample.items()
                }
                with torch.no_grad(), observer.set_attention_mask(attention_mask):
                    model(**sample)
            observer.save_state(output_path)
        if last_path is None:
            raise ValueError("The calibration dataset produced no batches")
        return torch.load(last_path, weights_only=False)
    finally:
        observer.close_hooks()


@torch.no_grad()
def smoke_test(model: nn.Module, tokenizer: AutoTokenizer) -> None:
    messages = [{"role": "user", "content": "Hello. What is 2 + 3?"}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, tokenize=True
    ).to(model.device)
    output = model.generate(inputs, max_new_tokens=32, do_sample=False)
    logger.info("Smoke-test response: %s", tokenizer.decode(output[0]))


def dump_args_to_yaml(output_dir: pathlib.Path, **argument_groups) -> None:
    payload = {
        name: dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value
        for name, value in argument_groups.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "reap_args.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=True)


def main() -> None:
    parser = HfArgumentParser((ReapArgs, ModelArgs, DatasetArgs, ObserverArgs))
    reap_args, model_args, ds_args, obs_args = parser.parse_args_into_dataclasses()
    require_npu()
    set_seed(reap_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, trust_remote_code=True)
    model = load_reap_model(
        model_args.model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    output_dir = create_results_directory(model_args.model_name, ds_args.dataset_name)
    record_activations(
        model, tokenizer, reap_args, model_args, ds_args, obs_args, output_dir
    )
    dump_args_to_yaml(
        output_dir,
        reap_args=reap_args,
        model_args=model_args,
        dataset_args=ds_args,
        observer_args=obs_args,
    )


if __name__ == "__main__":
    main()
