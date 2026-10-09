from __future__ import annotations
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Optional
import gc
from functools import reduce

import torch
import torch.nn as nn
import re
from dataclasses import dataclass
import logging
import pathlib
from functools import reduce

from reap.metrics import (
    ttm_online,
    get_routed_characteristic_activation,
    ca_dist_online,
    OnlineStatsTracker,
    get_distance_fn,
)
from reap.pruning_metrics import initialize_pruning_state, update_pruning_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTransformerObserverHookConfig:
    state_attr_name: str = "hook_state"
    hook_attr_name: str = "hooks"
    module_name_to_hook_regex: Optional[str] = None
    module_class_name_to_hook_regex: Optional[nn.Module] = None


class BaseTransformerObserver(ABC):
    def __init__(
        self,
        model,
        hook_config: Optional[BaseTransformerObserverHookConfig] = None,
    ):
        self.model = model
        self.hook_config = hook_config
        self.hooks = []
        self.state: dict[Any, Any] = {}
        self._hook_model()
        logger.info(
            "%s initialized for %s.",
            self.__class__.__name__,
            self.model.__class__.__name__,
        )

    @abstractmethod
    def _hook_factory(self, module: nn.Module, layer_number: int) -> callable:
        """
        Factory method to create a hook function for the given module.
        This method should be implemented by subclasses to define how the
        hook function should behave.
        """
        raise NotImplementedError("Subclasses must implement _hook_factory method.")

    def report_state(self) -> dict[str, Any]:
        """
        Method to report the current state of the observer. Can be overridden to inject
        custom behaviours.
        """
        return self.state

    def close_hooks(self):
        """Close all hooks registered to the model."""
        self.reset()  # Reset the state before closing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.debug("All hooks closed for %s.", self.model.__class__.__name__)

    def reset(self):
        """Reset the observer state."""
        del self.state
        gc.collect()
        self.state = {}
        logger.debug("Observer state reset for %s.", self.model.__class__.__name__)

    def save_state(self, file_path: str | pathlib.Path):
        self._move_state_tensors_to_cpu()
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = self.report_state()
        with open(file_path, "wb") as f:
            torch.save(state_dict, f)
        logger.info("State saved to %s", file_path)

    def _move_state_tensors_to_cpu(self):
        """
        Move all tensors in the state dictionary to CPU.
        This is useful before saving the state to avoid NPU HBM pressure.
        """
        for layer_number, layer_state in self.state.items():
            for key, value in layer_state.items():
                if isinstance(value, torch.Tensor):
                    self.state[layer_number][key] = value.cpu()

    def _validate_hook_config(self):
        if self.hook_config is None:
            return
        if (
            self.hook_config.module_name_to_hook_regex is None
            and self.hook_config.module_class_name_to_hook_regex is None
        ):
            raise ValueError(
                "At least one of 'module_n`ame_to_hook_regex' or "
                "'module_type_to_hook_regex' must be provided in the hook config."
            )
        if (
            self.hook_config.module_name_to_hook_regex is not None
            and self.hook_config.module_class_name_to_hook_regex is not None
        ):
            logger.warning(
                "Both 'module_name_to_hook_regex' and 'module_type_to_hook_regex' are "
                "provided. Both conditions must be satisfied to hook the module."
            )

    def _hook_model(self):
        for name, module in self.model.named_modules():
            hook_module = False
            if (
                self.hook_config.module_name_to_hook_regex
                and re.search(self.hook_config.module_name_to_hook_regex, name)
            ) or (
                self.hook_config.module_class_name_to_hook_regex
                and module.__class__.__name__
                == self.hook_config.module_class_name_to_hook_regex
            ):
                hook_module = True
            if hook_module:
                layer_number = int(re.search(r"\d+", name).group(0))
                hook_fn = self._hook_factory(module, layer_number)
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)
                logger.info("Hooked module: %s at layer %d", name, layer_number)
        if len(self.hooks) == 0:
            raise ValueError(
                "No modules matched the provided hook configuration. "
                "Check your hook configuration settings."
            )

    @classmethod
    def _get_registry_for_cls(cls) -> dict[str, type[BaseTransformerObserver]]:
        """Helper to get the registry from the specific class 'cls'."""
        if not hasattr(cls, "_architecture_registry") or not isinstance(
            cls._architecture_registry, dict
        ):
            raise AttributeError(
                f"Class {cls.__name__} must define its own "
                "`_architecture_registry: dict[str, type] = {{}}` "
                f"to use the common registration/creation methods."
            )
        return cls._architecture_registry

    @classmethod
    def register_implementation(cls, *arch_names: str):
        """
        Class method decorator to register a concrete observer implementation.
        'cls' is the class on which this decorator's factory is called (e.g.,
        MoEExpertObserver) 'sub_cls' is the class being decorated
        (for example a model-specific expert observer).
        """

        def decorator(sub_cls: type[BaseTransformerObserver]):
            registry = cls._get_registry_for_cls()

            for name in arch_names:
                if name in registry:
                    raise RuntimeError(
                        f"Architecture {name} already registered with "
                        f"{registry[name].__name__} for {cls.__name__}."
                    )
                registry[name] = sub_cls
            return sub_cls

        return decorator

    @classmethod
    def create_from_registry(
        cls,
        model: nn.Module,
        hook_config: Optional[BaseTransformerObserverHookConfig] = None,
        return_rank_0_only: bool = True,
        **kwargs: Any,
    ) -> BaseTransformerObserver:
        registry = cls._get_registry_for_cls()
        model_cls_name = model.__class__.__name__

        specific_observer_cls = registry.get(model_cls_name)

        if specific_observer_cls:
            return specific_observer_cls(
                model,
                hook_config=hook_config,
                return_rank_0_only=return_rank_0_only,
                **kwargs,
            )
        else:
            raise ValueError(
                "Unsupported architecture for "
                f"{cls.__name__}: {model_cls_name}. "
                "Registered architectures in "
                f"{cls.__name__}._architecture_registry: "
                f"{list(registry.keys())}"
            )


# --- MoE Transformer Observer ---------------------------------------------------------


@dataclass
class MoETransformerObserverConfig(BaseTransformerObserverHookConfig):
    num_experts_attr_name: str = "num_experts"
    top_k_attr_name: str = "top_k"
    fused_experts: bool = False
    # Some implementations expose the router under a different name (for
    # example Qwen3.5-MoE uses ``gate``), and some do not return routing data
    # from the hooked module at all.
    router_attr_name: str = "router"
    batched_expert_weights: bool = False
    distance_measure: str = "angular"
    renormalize_router_weights: bool = False
    record_pruning_metrics_only: bool = False


@torch.no_grad()
def compute_batched_expert_activations(
    experts: nn.Module,
    flat_input: torch.Tensor,
    num_experts: int,
    chunk_size: int = 16,
) -> torch.Tensor:
    """Evaluate fused expert weights for every input token.

    Qwen3.5 stores ``gate_up_proj`` and ``down_proj`` as batched tensors and
    its expert module only accepts routed top-k indices. REAP needs the output
    of every expert, so compute the expert MLPs in chunks to bound temporary
    memory usage.
    """

    hidden_dim = flat_input.shape[-1]
    activations = torch.zeros(
        (num_experts, *flat_input.shape),
        device=flat_input.device,
        dtype=flat_input.dtype,
    )
    for start in range(0, num_experts, chunk_size):
        end = min(start + chunk_size, num_experts)
        gate_up = torch.einsum(
            "th,cih->tci", flat_input, experts.gate_up_proj[start:end]
        )
        gate, up = gate_up.chunk(2, dim=-1)
        expert_hidden = experts.act_fn(gate) * up
        expert_out = torch.einsum(
            "tci,chi->tch", expert_hidden, experts.down_proj[start:end]
        )
        activations[start:end] = expert_out.permute(1, 0, 2)
    assert activations.shape == (num_experts, flat_input.shape[0], hidden_dim)
    return activations


class MoETransformerObserver(BaseTransformerObserver):
    """Collect MoE routing and activation statistics for expert pruning."""

    def __init__(self, model, hook_config=None):
        self._current_attention_mask: Optional[torch.Tensor] = None
        super().__init__(model, hook_config)

    @contextmanager
    def set_attention_mask(self, attention_mask: Optional[torch.Tensor]):
        """Temporarily set the attention mask for the current forward pass.

        Use this as a context manager around each forward pass when using
        batched inputs with padding, to ensure padding tokens are excluded
        from statistics.

        Args:
            attention_mask: Tensor of shape (batch_size, seq_len) with 1 for real
                tokens and 0 for padding tokens. Can be None for unbatched inputs.
        """
        previous_attention_mask = self._current_attention_mask
        self._current_attention_mask = attention_mask
        try:
            yield
        finally:
            self._current_attention_mask = previous_attention_mask

    def clear_attention_mask(self):
        """Clear the attention mask after forward pass."""
        self._current_attention_mask = None

    def report_state(self) -> dict[str, Any]:
        """
        Method to report the current state of the observer. Can be overridden to inject
        custom behaviours.
        """
        return {
            layer_num: {
                k: v.mean if isinstance(v, OnlineStatsTracker) else v
                for k, v in layer_state.items()
            }
            for layer_num, layer_state in self.state.items()
        }

    def _initialize_state(self, output: torch.Tensor, num_experts: int):
        # get device and shape info
        output_hidden_states = output[0] if isinstance(output, (tuple, list)) else output
        device = "cpu"
        hidden_dim = output_hidden_states.shape[-1]
        layer_state = initialize_pruning_state(num_experts, device=device)

        if not self.hook_config.record_pruning_metrics_only:
            # per routed token normalized states
            layer_state["ttm_similarity_matrix"] = OnlineStatsTracker(
                shape=(num_experts, num_experts),
                count_shape=(num_experts, num_experts),
                device=device,
                dtype=torch.float32,
            )
            layer_state["routed_characteristic_activation"] = OnlineStatsTracker(
                shape=(num_experts, hidden_dim),
                count_shape=(num_experts, hidden_dim),
                device=device,
                dtype=torch.float32,
            )
            # HC-SMoE
            layer_state["characteristic_activation"] = OnlineStatsTracker(
                shape=(num_experts, hidden_dim),
                count_shape=1,
                device=device,
                dtype=torch.float32,
            )
            # SubMoE
            layer_state["online_characteristic_activation_dist"] = OnlineStatsTracker(
                shape=(num_experts, num_experts),
                count_shape=1,
                device=device,
                dtype=torch.float32,
            )
            # per total token normalized states -> MC-SMoE
            layer_state["router_logit_similiarity"] = OnlineStatsTracker(
                shape=(num_experts, num_experts),
                count_shape=1,
                device=device,
                dtype=torch.float32,
            )

        return layer_state

    def _hook_factory(self, module: nn.Module, layer_number: int) -> callable:
        distance_fn = get_distance_fn("cosine") # always use cosine for online dist. metrics
        num_experts = reduce(
            getattr, self.hook_config.num_experts_attr_name.split("."), module
        )
        top_k = reduce(getattr, self.hook_config.top_k_attr_name.split("."), module)
        if num_experts is None or top_k is None:
            raise ValueError(
                f"Module {module.__class__.__name__} at layer {layer_number} "
                "does not have expected 'num_experts' or 'top_k' attributes. Check "
                "HookConfig settings."
            )

        @torch.no_grad()
        def _hook_fn(module, args, output):
            if not self.hook_config.fused_experts and not isinstance(output, (tuple, list)):
                raise ValueError(
                    f"Expected output of module {module.__class__.__name__} at layer "
                    f"{layer_number} to be a tuple of at least length 2, got "
                    f"{type(output).__name__}."
                )
            input = args[0]  # (batch_size, seq_len, hidden_dim)
            device = input.device
            if layer_number not in self.state:
                self.state[layer_number] = self._initialize_state(output, num_experts)
            batch_size, sequence_length, hidden_dim = input.shape
            flat_input = input.view(-1, hidden_dim)  # total_seq_len, hidden

            attention_mask = self._current_attention_mask
            if attention_mask is not None:
                # Flatten mask to match flat_input: (batch_size * seq_len,)
                flat_mask = attention_mask.view(-1).bool().to(device)
            else:
                # No mask provided - treat all tokens as valid
                flat_mask = None

            activations = torch.zeros((num_experts, *flat_input.shape), device=device)

            if self.hook_config.fused_experts:
                if self.hook_config.batched_expert_weights:
                    # Qwen3.5-MoE returns only the combined hidden states from
                    # its block. Re-run the router so the observer can collect
                    # routing statistics, then evaluate all experts from their
                    # fused [num_experts, ...] weight tensors.
                    router = reduce(
                        getattr,
                        self.hook_config.router_attr_name.split("."),
                        module,
                    )
                    router_logits, _, selected_experts = router(flat_input)
                    selected_experts = selected_experts.to(device)
                    activations = compute_batched_expert_activations(
                        module.experts, flat_input, num_experts
                    )
                else:
                    _, router_scores = output  # (num_experts, total_tokens)
                    router = reduce(
                        getattr,
                        self.hook_config.router_attr_name.split("."),
                        module,
                    )
                    router_logits = router(flat_input)  # (total_tokens, num_experts)
                    _, selected_experts = torch.topk(router_logits, top_k, dim=-1)
                    selected_experts = selected_experts.to(device)
                    router_indices = (
                        torch.arange(batch_size * sequence_length, device=device)
                        .view(1, -1)
                        .expand(router_scores.size(0), -1)
                    )
                    router_indices = router_indices.reshape(-1, 1).expand(-1, hidden_dim)
                    routed_in = torch.gather(
                        input=flat_input,
                        dim=0,
                        index=router_indices,
                    ).to(device)
                    # we do not apply router_scores
                    # record unweighted activations for all experts
                    routed_out = module.experts(routed_in)
                    activations = routed_out.view(num_experts, *flat_input.shape)

            else:  # loop based MoE execution
                # ernie returns combined_output, combine_weights, router_loss, gate_logits
                *_, router_logits = output  # (total_tokens, num_experts)
                _, selected_experts = torch.topk(router_logits, top_k, dim=-1)
                # selected_experts = selected_experts.to(device)
                for idx, expert in enumerate(module.experts):
                    activations[idx] = expert(flat_input).to(
                        device
                    )  # (num_experts, total_seq_len, hidden_dim)

            del flat_input
            
            pruning_batch = update_pruning_state(
                self.state[layer_number],
                activations=activations,
                selected_experts=selected_experts,
                router_logits=router_logits,
                num_experts=num_experts,
                valid_token_mask=flat_mask,
                renormalize_router_weights=self.hook_config.renormalize_router_weights,
            )

            # Optional extended similarity statistics retained for compatibility.
            if not self.hook_config.record_pruning_metrics_only:
                ttm_similarity_matrix = ttm_online(
                    pruning_batch.activations,
                    pruning_batch.selected_experts,
                    distance_callable=distance_fn,
                    num_experts=num_experts,
                    pairwise_expert_frequency=pruning_batch.pairwise_expert_frequency,
                )

                # ttm_similarity_matrix with pairwise frequency counts
                self.state[layer_number]["ttm_similarity_matrix"].update(
                    ttm_similarity_matrix, pruning_batch.pairwise_expert_frequency
                )
                del ttm_similarity_matrix

                routed_characteristic_activation = get_routed_characteristic_activation(
                    pruning_batch.activations,
                    pruning_batch.selected_experts,
                    pruning_batch.expert_frequency,
                    device,
                    hidden_dim,
                    num_experts,
                )

                # routed_characteristic_activation with expert frequency counts
                expert_freq_expanded = pruning_batch.expert_frequency.unsqueeze(-1).expand(
                    (-1, hidden_dim)
                )
                self.state[layer_number]["routed_characteristic_activation"].update(
                    routed_characteristic_activation, expert_freq_expanded
                )
                del expert_freq_expanded, routed_characteristic_activation

                online_characteristic_activation_dist = ca_dist_online(
                    pruning_batch.activations,
                    distance_callable=distance_fn,
                ).to(device="cpu")

                # online_characteristic_activation_dist with expert frequency counts
                self.state[layer_number]["online_characteristic_activation_dist"].update(
                    online_characteristic_activation_dist, pruning_batch.num_tokens
                )
                del online_characteristic_activation_dist

                # router logit similarity -> must align with distance_fn shape expectations
                # dim 0 "batch" dim, dims 1,2 expert pairwise, dim 3 token logits
                router_logit_sim = (
                    distance_fn(
                        pruning_batch.router_logits.permute(1, 0).view(
                            1, num_experts, 1, -1
                        ),  # 1, num_experts, 1, logits
                        pruning_batch.router_logits.permute(1, 0).view(
                            1, 1, num_experts, -1
                        ),  # 1, 1, num_experts, logits
                    )
                    .squeeze()
                    .to(device="cpu")
                )  # yields (num_experts, num_experts)

                # router_logit_similarity with total tokens count
                self.state[layer_number]["router_logit_similiarity"].update(
                    router_logit_sim, pruning_batch.num_tokens
                )
                del router_logit_sim

                # characteristic_activation with total tokens count
                self.state[layer_number]["characteristic_activation"].update(
                    pruning_batch.activations.mean(dim=1), pruning_batch.num_tokens
                )

            # --- CLEAN UP -------------------------------------------------------------
            del (
                activations,
                selected_experts,
                router_logits,
                pruning_batch,
            )
            gc.collect()

        return _hook_fn


# --- Concrete Config Implementations ----


@dataclass
class Qwen3MoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "Qwen3MoeSparseMoeBlock"
    num_experts_attr_name: str = "experts.num_experts"
    top_k_attr_name: str = "gate.top_k"
    fused_experts: bool = True
    router_attr_name: str = "gate"
    batched_expert_weights: bool = True


@dataclass
class Qwen35MoEObserverHookConfig(MoETransformerObserverConfig):
    """Observer settings for the Qwen3.5 multimodal MoE implementation.

    Qwen3.5 stores expert projections as batched tensors and returns only the
    combined block output, so routing and per-expert activations must be
    reconstructed from ``gate`` and ``experts``.
    """

    module_class_name_to_hook_regex: Optional[str] = "Qwen3_5MoeSparseMoeBlock"
    num_experts_attr_name: str = "experts.num_experts"
    top_k_attr_name: str = "gate.top_k"
    fused_experts: bool = True
    router_attr_name: str = "gate"
    batched_expert_weights: bool = True


OBSERVER_CONFIG_REGISTRY = {
    "Qwen3MoeForCausalLM": Qwen3MoEObserverHookConfig,
    "Qwen3_5MoeForConditionalGeneration": Qwen35MoEObserverHookConfig,
    "Qwen3_5MoeForCausalLM": Qwen35MoEObserverHookConfig,
}
