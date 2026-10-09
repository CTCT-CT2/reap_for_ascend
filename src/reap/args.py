"""Command-line argument groups for the Ascend REAP pipeline."""

from dataclasses import dataclass, field


@dataclass
class ReapArgs:
    seed: int = field(default=42, metadata={"help": "Random seed."})
    profile: bool = field(default=True, metadata={"help": "Run a warm-up pass."})
    run_observer_only: bool = field(
        default=False, metadata={"help": "Collect observations without pruning."}
    )
    smoke_test: bool = field(
        default=True, metadata={"help": "Generate a response after pruning."}
    )


@dataclass
class ModelArgs:
    model_name: str = field(
        default="Qwen/Qwen3-30B-A3B",
        metadata={"help": "Local checkpoint path or Hugging Face model ID."},
    )
    num_experts_per_tok_override: int | None = field(
        default=None, metadata={"help": "Override routed experts per token."}
    )


@dataclass
class DatasetArgs:
    dataset_name: str = field(
        default="theblackcat102/evol-codealpaca-v1",
        metadata={
            "help": (
                "Dataset name, or comma-separated <dataset>[subset](split):batches "
                "components. Use 'combined' to load cached observations."
            )
        },
    )
    dataset_config_name: str | None = field(
        default=None, metadata={"help": "Optional dataset subset/configuration."}
    )
    split: str = field(default="train", metadata={"help": "Dataset split."})
    shuffle: bool = field(default=True, metadata={"help": "Shuffle calibration data."})


@dataclass
class ObserverArgs:
    batches_per_category: int = 1024
    split_by_category: bool = False
    select_only_categories: list[str] | str | None = None
    batch_size: int = 8
    model_max_length: int | None = 2048
    return_vllm_tokens_prompt: bool = False
    truncate: bool = False
    overwrite_observations: bool = False
    distance_measure: str = field(
        default="cosine",
        metadata={
            "help": "Distance retained for observation-file compatibility.",
            "choices": ["angular", "euclidean", "jsd", "cka", "cosine"],
        },
    )
    output_file_name: str = "observations_reap.pt"
    record_pruning_metrics_only: bool = field(
        default=True, metadata={"help": "Collect only expert-pruning metrics."}
    )
    renormalize_router_weights: bool = True


@dataclass
class PruneArgs:
    pruning_plan: str | None = field(
        default=None,
        metadata={"help": "JSON plan mapping each MoE layer to retained experts."},
    )
    pruned_model_dir: str | None = None
    overwrite_pruned_model: bool = False
    prune_method: str = field(
        default="reap",
        metadata={
            "help": "Expert score used for observation-driven pruning.",
            "choices": [
                "frequency",
                "ean_ca",
                "ean_sum",
                "ean_mean",
                "weighted_expert_frequency_sum",
                "weighted_ean_sum",
                "weighted_ean_sum_l2",
                "reap",
                "reap_l2",
                "max_activations",
            ],
        },
    )
    n_experts_to_prune: int | None = None
    compression_ratio: float = field(
        default=0.25,
        metadata={"help": "Fraction removed when expert count is omitted."},
    )
    preserve_super_experts: bool = field(
        default=False,
        metadata={"help": "Preserve high-activation experts."},
    )
    preserve_outliers: bool = False


@dataclass
class LayerwiseArgs:
    """Memory controls for layer-by-layer observation."""

    batch_group_size: int | None = None
    save_intermediate: bool = False
    low_cpu_mem_usage: bool = True
