from __future__ import annotations

import types
import weakref
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from cache_moe import (
    _generic_router_logits,
    _generic_router_module,
    _generic_selected_experts_from_logits,
    _infer_generic_top_k,
    _is_generic_router_moe_block,
    _is_qwen3_moe_block,
    _router_weight,
)
from lru_cache_metrics import LRUCacheMetricConfig, compute_lru_cache_metrics


@dataclass
class WindowCacheLossConfig:
    enabled: bool = False
    window_size: int = 16
    capacity: int = 20
    target_avg_swap: float = 1.0
    max_swap_thresh: float = 4.0
    mtp_swap_ratio: float = 1.5
    mtp_num_draft_tokens: int = 4
    avg_swap_coeff: float = 1.0
    peak_swap_coeff: float = 2.0
    mtp_swap_coeff: float = 1.0
    sharpness_alpha: float = 2.0
    peak_penalty_gamma: float = 2.0
    training_mode: str = "cache_aware"
    epsilon: float = 1e-8


def _soft_topk_proxy_from_logits(
    router_logits: torch.Tensor,
    active_experts_per_token: int,
    sharpness_alpha: float,
    epsilon: float,
) -> torch.Tensor:
    topk_scale = max(float(active_experts_per_token), 1.0)
    probs = F.softmax(sharpness_alpha * router_logits.to(dtype=torch.float32), dim=-1) * topk_scale
    return probs.clamp(min=0.0, max=1.0 - epsilon)


class WindowCacheTrackedQwen3MoeSparseBlock(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
        owner_model: nn.Module,
        layer_idx: int,
        window_cache_config: WindowCacheLossConfig,
    ) -> None:
        super().__init__()
        self.experts = original_block.experts
        self.gate = original_block.gate
        self.layer_idx = layer_idx
        self.window_cache_config = window_cache_config
        self._owner_model_ref = weakref.ref(owner_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.reshape(-1, hidden_dim)

        router_logits, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        owner_model = self._owner_model_ref()
        if owner_model is not None and getattr(owner_model, "_window_cache_loss_enabled", False):
            activation_probs = _soft_topk_proxy_from_logits(
                router_logits,
                active_experts_per_token=selected_experts.size(-1),
                sharpness_alpha=self.window_cache_config.sharpness_alpha,
                epsilon=self.window_cache_config.epsilon,
            )
            owner_model._window_cache_layer_stats.append(
                {
                    "layer_idx": self.layer_idx,
                    "activation_probs": activation_probs.reshape(batch_size, sequence_length, -1),
                    "promoe_inputs": hidden_states.detach(),
                    "router_logits": router_logits.reshape(batch_size, sequence_length, -1),
                    "selected_experts": selected_experts.reshape(batch_size, sequence_length, -1),
                }
            )

        return final_hidden_states


def _is_deepseek_v2_moe_block(block: nn.Module) -> bool:
    config = getattr(block, "config", None)
    return (
        hasattr(block, "experts")
        and hasattr(block, "gate")
        and config is not None
        and getattr(config, "n_routed_experts", None) is not None
        and getattr(config, "num_experts_per_tok", None) is not None
    )


def _deepseek_router_logits(gate: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_flat = hidden_states.reshape(batch_size * sequence_length, hidden_dim)
    return F.linear(
        hidden_states_flat.to(dtype=torch.float32),
        gate.weight.to(dtype=torch.float32),
        None,
    )


def _deepseek_top_k(gate: nn.Module) -> int:
    config = getattr(gate, "config", None)
    return int(
        getattr(gate, "top_k", None)
        or getattr(gate, "num_experts_per_tok", None)
        or getattr(config, "num_experts_per_tok", None)
    )


def _deepseek_scores_from_logits(gate: nn.Module, router_logits: torch.Tensor) -> torch.Tensor:
    scoring_func = str(getattr(gate, "scoring_func", "softmax")).lower()
    if scoring_func == "softmax":
        return F.softmax(router_logits, dim=-1, dtype=torch.float32)
    if scoring_func == "sigmoid":
        return torch.sigmoid(router_logits.to(dtype=torch.float32))
    raise ValueError(f"Unsupported DeepSeek MoE scoring_func: {scoring_func!r}")


def _deepseek_selected_experts_from_logits(gate: nn.Module, router_logits: torch.Tensor) -> torch.Tensor:
    scores = _deepseek_scores_from_logits(gate, router_logits)
    top_k = _deepseek_top_k(gate)
    topk_method = str(getattr(gate, "topk_method", "greedy")).lower()
    selection_scores = scores
    correction_bias = getattr(gate, "e_score_correction_bias", None)
    if correction_bias is not None:
        selection_scores = selection_scores + correction_bias.to(device=scores.device, dtype=scores.dtype)

    if topk_method in {"greedy", "gready"}:
        return torch.topk(selection_scores, k=top_k, dim=-1, sorted=False).indices.to(dtype=torch.long)
    if topk_method == "group_limited_greedy":
        config = getattr(gate, "config", None)
        n_group = int(getattr(gate, "n_group", None) or getattr(config, "n_group", 1))
        topk_group = int(getattr(gate, "topk_group", None) or getattr(config, "topk_group", n_group))
        if selection_scores.size(-1) % n_group != 0:
            raise ValueError(
                f"DeepSeek group_limited_greedy expects experts divisible by n_group, "
                f"got experts={selection_scores.size(-1)}, n_group={n_group}."
            )
        grouped_scores = selection_scores.view(selection_scores.size(0), n_group, -1)
        group_scores = grouped_scores.max(dim=-1).values
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_idx, True)
        score_mask = group_mask.unsqueeze(-1).expand_as(grouped_scores).reshape_as(selection_scores)
        masked_scores = selection_scores.masked_fill(~score_mask, torch.finfo(selection_scores.dtype).min)
        return torch.topk(masked_scores, k=top_k, dim=-1, sorted=False).indices.to(dtype=torch.long)
    raise ValueError(f"Unsupported DeepSeek MoE topk_method: {topk_method!r}")


def _is_integer_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)


def _deepseek_gate_selection(gate: nn.Module, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gate_outputs = gate(hidden_states)
    if not isinstance(gate_outputs, tuple):
        raise TypeError("DeepSeek MoE gate is expected to return a tuple.")
    if len(gate_outputs) >= 3:
        selected_experts, routing_weights = gate_outputs[0], gate_outputs[1]
        return selected_experts, routing_weights
    if len(gate_outputs) == 2:
        first, second = gate_outputs
        if _is_integer_tensor(first):
            return first, second
        return second, first
    raise TypeError(f"Unsupported DeepSeek MoE gate output length: {len(gate_outputs)}")


def _sparse_probs_from_topk(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
    batch_size: int,
    sequence_length: int,
) -> torch.Tensor:
    dense = torch.zeros(
        selected_experts.size(0),
        num_experts,
        device=routing_weights.device,
        dtype=torch.float32,
    )
    dense.scatter_add_(1, selected_experts.to(dtype=torch.long), routing_weights.to(dtype=torch.float32))
    return dense.reshape(batch_size, sequence_length, num_experts)


class WindowCacheTrackedDeepseekV2MoEBlock(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
        owner_model: nn.Module,
        layer_idx: int,
        window_cache_config: WindowCacheLossConfig,
    ) -> None:
        super().__init__()
        self.original_block = original_block
        self.layer_idx = layer_idx
        self.window_cache_config = window_cache_config
        self._owner_model_ref = weakref.ref(owner_model)

    @property
    def experts(self):
        return self.original_block.experts

    @property
    def gate(self):
        return self.original_block.gate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        final_hidden_states = self.original_block(hidden_states)

        owner_model = self._owner_model_ref()
        if owner_model is not None and getattr(owner_model, "_window_cache_loss_enabled", False):
            router_logits = _deepseek_router_logits(self.gate, hidden_states)
            selected_experts = _deepseek_selected_experts_from_logits(self.gate, router_logits)
            activation_probs = _soft_topk_proxy_from_logits(
                router_logits,
                active_experts_per_token=selected_experts.size(-1),
                sharpness_alpha=self.window_cache_config.sharpness_alpha,
                epsilon=self.window_cache_config.epsilon,
            )
            owner_model._window_cache_layer_stats.append(
                {
                    "layer_idx": self.layer_idx,
                    "activation_probs": activation_probs.reshape(batch_size, sequence_length, -1),
                    "promoe_inputs": hidden_states.detach(),
                    "router_logits": router_logits.reshape(batch_size, sequence_length, -1),
                    "selected_experts": selected_experts.reshape(batch_size, sequence_length, -1),
                }
            )

        return final_hidden_states


class WindowCacheTrackedGenericRouterMoEBlock(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
        owner_model: nn.Module,
        layer_idx: int,
        window_cache_config: WindowCacheLossConfig,
    ) -> None:
        super().__init__()
        self.original_block = original_block
        self.layer_idx = int(layer_idx)
        self.window_cache_config = window_cache_config
        self._owner_model_ref = weakref.ref(owner_model)
        router = _generic_router_module(original_block)
        if router is None:
            raise ValueError("Generic MoE block has no supported router/gate module.")
        router_weight = _router_weight(router)
        if router_weight is None:
            raise ValueError("Generic MoE router has no usable weight for shape inference.")
        self._router_attr_name = "router" if getattr(original_block, "router", None) is router else "gate"
        self.num_experts = int(router_weight.shape[0])
        self.top_k = _infer_generic_top_k(original_block, router, owner_model=owner_model)

    @property
    def router(self):
        return getattr(self.original_block, self._router_attr_name)

    @property
    def gate(self):
        return self.router

    @property
    def experts(self):
        return getattr(self.original_block, "experts", None)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        final_hidden_states = self.original_block(hidden_states, *args, **kwargs)

        owner_model = self._owner_model_ref()
        if owner_model is not None and getattr(owner_model, "_window_cache_loss_enabled", False):
            router_logits = _generic_router_logits(self.router, hidden_states, num_experts=self.num_experts)
            selected_experts, _ = _generic_selected_experts_from_logits(router_logits, top_k=self.top_k)
            activation_probs = _soft_topk_proxy_from_logits(
                router_logits,
                active_experts_per_token=selected_experts.size(-1),
                sharpness_alpha=self.window_cache_config.sharpness_alpha,
                epsilon=self.window_cache_config.epsilon,
            )
            owner_model._window_cache_layer_stats.append(
                {
                    "layer_idx": self.layer_idx,
                    "activation_probs": activation_probs.reshape(batch_size, sequence_length, self.num_experts),
                    "promoe_inputs": hidden_states.detach(),
                    "router_logits": router_logits.reshape(batch_size, sequence_length, self.num_experts),
                    "selected_experts": selected_experts.reshape(batch_size, sequence_length, -1),
                }
            )

        return final_hidden_states


def _valid_lengths(attention_mask: Optional[torch.Tensor], batch_size: int, seq_len: int) -> List[int]:
    if attention_mask is None:
        return [seq_len] * batch_size
    return attention_mask.to(dtype=torch.long).sum(dim=-1).tolist()


def _soft_cache_state_from_window(
    activation_probs: torch.Tensor,
    window_size: int,
    epsilon: float,
) -> torch.Tensor:
    if activation_probs.numel() == 0 or window_size <= 0:
        return torch.zeros_like(activation_probs, dtype=torch.float32)

    probs = activation_probs.to(dtype=torch.float32).clamp(min=0.0, max=1.0 - epsilon)
    log_miss = torch.log((1.0 - probs).clamp_min(epsilon))
    inclusive_cumsum = torch.cumsum(log_miss, dim=0)
    cumsum_shifted = torch.cat(
        [
            torch.zeros((1, probs.size(-1)), device=probs.device, dtype=probs.dtype),
            inclusive_cumsum[:-1],
        ],
        dim=0,
    )

    token_indices = torch.arange(probs.size(0), device=probs.device)
    window_starts = torch.clamp(token_indices - window_size, min=0)
    start_cumsum = torch.zeros_like(cumsum_shifted)
    nonzero_start_mask = window_starts > 0
    if nonzero_start_mask.any():
        start_cumsum[nonzero_start_mask] = inclusive_cumsum[window_starts[nonzero_start_mask] - 1]
    history_log_miss = cumsum_shifted - start_cumsum
    return 1.0 - torch.exp(history_log_miss)


def _mtp_swap_loss_sum_from_future_window(
    base_cache_state: torch.Tensor,
    draft_logits: torch.Tensor,
    num_draft_tokens: int,
    active_experts_per_token: int,
    target_ratio: float,
    sharpness_alpha: float,
    epsilon: float,
) -> tuple[torch.Tensor, int]:
    if draft_logits.numel() == 0 or num_draft_tokens <= 0:
        return torch.zeros((), device=draft_logits.device, dtype=torch.float32), 0

    seq_len = draft_logits.size(0)
    if seq_len <= 1:
        return torch.zeros((), device=draft_logits.device, dtype=torch.float32), 0

    total_loss = torch.zeros((), device=draft_logits.device, dtype=torch.float32)
    counted_positions = 0
    probs = _soft_topk_proxy_from_logits(
        draft_logits,
        active_experts_per_token=active_experts_per_token,
        sharpness_alpha=sharpness_alpha,
        epsilon=epsilon,
    )
    cache_state = base_cache_state.to(dtype=torch.float32).clamp(min=0.0, max=1.0)

    for token_idx in range(seq_len - 1):
        draft_end = min(seq_len, token_idx + 1 + num_draft_tokens)
        draft_probs = probs[token_idx + 1 : draft_end]
        actual_draft_tokens = draft_probs.size(0)
        if actual_draft_tokens <= 0:
            continue

        draft_not_activated_log = torch.log((1.0 - draft_probs).clamp_min(epsilon)).sum(dim=0)
        draft_union_probs = 1.0 - torch.exp(draft_not_activated_log)
        new_experts = (draft_union_probs * (1.0 - cache_state[token_idx])).sum()
        threshold = target_ratio * float(actual_draft_tokens)
        total_loss = total_loss + F.relu(new_experts - threshold)
        counted_positions += 1

    return total_loss, counted_positions


def compute_window_cache_loss(
    layer_stats: List[Dict[str, torch.Tensor]],
    attention_mask: Optional[torch.Tensor],
    window_cache_config: WindowCacheLossConfig,
    compute_hard_metrics: bool = True,
) -> Dict[str, torch.Tensor]:
    device = layer_stats[0]["activation_probs"].device if layer_stats else (
        attention_mask.device if attention_mask is not None else torch.device("cpu")
    )
    zero = torch.zeros((), device=device, dtype=torch.float32)

    if not layer_stats:
        return {
            "avg_swap_metric": zero,
            "peak_swap_metric": zero,
            "avg_swap_loss": zero,
            "peak_swap_loss": zero,
            "mtp_swap_loss": zero,
            "cache_utilization": zero,
            "soft_cache_hit_rate": zero,
            "soft_cache_miss_rate": zero,
            "avg_expected_active_experts": zero,
            "hard_cache_hit_rate": zero,
            "hard_cache_miss_rate": zero,
            "hard_cache_overlap_count": zero,
            "hard_cache_swap_count": zero,
        }

    total_swap = zero
    total_expected_active = zero
    total_cache_utilization = zero
    total_soft_hit_mass = zero
    total_avg_swap_loss = zero
    total_peak_swap_loss = zero
    total_mtp_swap_loss = zero
    total_peak_swap_metric = zero
    total_avg_selected_experts = zero
    counted_tokens = 0
    counted_sequences = 0
    counted_mtp_positions = 0

    lru_ready_stats: List[Dict[str, torch.Tensor]] = []

    for layer in layer_stats:
        activation_probs = layer["activation_probs"]
        router_logits = layer["router_logits"]
        selected_experts = layer["selected_experts"]
        batch_size, seq_len, _ = activation_probs.shape
        lengths = _valid_lengths(attention_mask, batch_size, seq_len)
        lru_ready_stats.append({"selected_experts": selected_experts})

        for batch_idx, length in enumerate(lengths):
            if length <= 0:
                continue

            probs = activation_probs[batch_idx, :length, :]
            logits = router_logits[batch_idx, :length, :]
            cache_state = _soft_cache_state_from_window(
                probs,
                window_size=window_cache_config.window_size,
                epsilon=window_cache_config.epsilon,
            )
            expected_swap = (probs * (1.0 - cache_state)).sum(dim=-1)
            expected_active = probs.sum(dim=-1)
            soft_hit_mass = (probs * cache_state).sum(dim=-1)
            cache_occupancy = cache_state.sum(dim=-1) / max(float(window_cache_config.capacity), 1.0)

            total_swap = total_swap + expected_swap.sum()
            total_expected_active = total_expected_active + expected_active.sum()
            total_soft_hit_mass = total_soft_hit_mass + soft_hit_mass.sum()
            total_cache_utilization = total_cache_utilization + cache_occupancy.sum()
            total_avg_swap_loss = total_avg_swap_loss + F.relu(
                expected_swap - window_cache_config.target_avg_swap
            ).sum()
            total_peak_swap_loss = total_peak_swap_loss + torch.pow(
                F.relu(expected_swap - window_cache_config.max_swap_thresh),
                window_cache_config.peak_penalty_gamma,
            ).sum()
            mtp_swap_loss_sum, mtp_positions = _mtp_swap_loss_sum_from_future_window(
                base_cache_state=cache_state,
                draft_logits=logits,
                num_draft_tokens=window_cache_config.mtp_num_draft_tokens,
                active_experts_per_token=selected_experts.size(-1),
                target_ratio=window_cache_config.mtp_swap_ratio,
                sharpness_alpha=window_cache_config.sharpness_alpha,
                epsilon=window_cache_config.epsilon,
            )
            total_mtp_swap_loss = total_mtp_swap_loss + mtp_swap_loss_sum
            counted_mtp_positions += mtp_positions
            total_peak_swap_metric = total_peak_swap_metric + expected_swap.max()
            total_avg_selected_experts = total_avg_selected_experts + torch.full(
                (),
                float(selected_experts.size(-1)),
                device=device,
                dtype=torch.float32,
            )
            counted_tokens += length
            counted_sequences += 1

    if counted_tokens == 0:
        return {
            "avg_swap_metric": zero,
            "peak_swap_metric": zero,
            "avg_swap_loss": zero,
            "peak_swap_loss": zero,
            "mtp_swap_loss": zero,
            "cache_utilization": zero,
            "soft_cache_hit_rate": zero,
            "soft_cache_miss_rate": zero,
            "avg_expected_active_experts": zero,
            "hard_cache_hit_rate": zero,
            "hard_cache_miss_rate": zero,
            "hard_cache_overlap_count": zero,
            "hard_cache_swap_count": zero,
        }

    token_denom = torch.tensor(float(counted_tokens), device=device, dtype=torch.float32)
    sequence_denom = torch.tensor(float(counted_sequences), device=device, dtype=torch.float32)
    mtp_denom = torch.tensor(float(counted_mtp_positions), device=device, dtype=torch.float32)
    active_denom = total_expected_active.clamp_min(window_cache_config.epsilon)

    hard_metrics = {
        "hard_cache_hit_rate": zero,
        "hard_cache_miss_rate": zero,
        "hard_cache_overlap_count": zero,
        "hard_cache_swap_count": zero,
    }
    if compute_hard_metrics:
        hard_metrics = compute_lru_cache_metrics(
            lru_ready_stats,
            attention_mask,
            LRUCacheMetricConfig(enabled=True, cache_size=window_cache_config.capacity),
            compute_hard_metrics=True,
        )
        avg_selected_experts = total_avg_selected_experts / sequence_denom
        hard_metrics["hard_cache_swap_count"] = avg_selected_experts - hard_metrics["hard_cache_overlap_count"]

    return {
        "avg_swap_metric": total_swap / token_denom,
        "peak_swap_metric": total_peak_swap_metric / sequence_denom,
        "avg_swap_loss": total_avg_swap_loss / token_denom,
        "peak_swap_loss": total_peak_swap_loss / token_denom,
        "mtp_swap_loss": total_mtp_swap_loss / mtp_denom.clamp_min(1.0),
        "cache_utilization": total_cache_utilization / token_denom,
        "soft_cache_hit_rate": total_soft_hit_mass / active_denom,
        "soft_cache_miss_rate": total_swap / active_denom,
        "avg_expected_active_experts": total_expected_active / token_denom,
        "hard_cache_hit_rate": hard_metrics["hard_cache_hit_rate"],
        "hard_cache_miss_rate": hard_metrics["hard_cache_miss_rate"],
        "hard_cache_overlap_count": hard_metrics["hard_cache_overlap_count"],
        "hard_cache_swap_count": hard_metrics["hard_cache_swap_count"],
    }


def _reset_window_cache_state(self):
    self._window_cache_layer_stats = []


def _compute_window_cache_aux(
    self,
    attention_mask: Optional[torch.Tensor] = None,
    compute_hard_metrics: bool = True,
) -> Dict[str, torch.Tensor]:
    return compute_window_cache_loss(
        self._window_cache_layer_stats,
        attention_mask,
        self._window_cache_config,
        compute_hard_metrics=compute_hard_metrics,
    )


def install_window_cache_loss(model: nn.Module, window_cache_config: WindowCacheLossConfig) -> nn.Module:
    if not window_cache_config.enabled or window_cache_config.training_mode == "none":
        return model

    if getattr(model, "_window_cache_loss_enabled", False):
        return model

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Window cache loss patching expects a decoder model with model.layers")

    valid_modes = {"cache_aware", "monitor"}
    if window_cache_config.training_mode not in valid_modes:
        raise ValueError(
            f"Unsupported window cache training_mode={window_cache_config.training_mode!r}. "
            f"Expected one of {sorted(valid_modes)}."
        )

    model._window_cache_loss_enabled = True
    model._window_cache_config = window_cache_config
    model._window_cache_layer_stats = []
    model.reset_window_cache_state = types.MethodType(_reset_window_cache_state, model)
    model.compute_window_cache_aux = types.MethodType(_compute_window_cache_aux, model)

    moe_layer_idx = 0
    for decoder_layer in model.model.layers:
        mlp = getattr(decoder_layer, "mlp", None)
        if mlp is None:
            continue
        if _is_deepseek_v2_moe_block(mlp):
            decoder_layer.mlp = WindowCacheTrackedDeepseekV2MoEBlock(
                original_block=mlp,
                owner_model=model,
                layer_idx=moe_layer_idx,
                window_cache_config=window_cache_config,
            )
            moe_layer_idx += 1
        elif _is_qwen3_moe_block(mlp):
            decoder_layer.mlp = WindowCacheTrackedQwen3MoeSparseBlock(
                original_block=mlp,
                owner_model=model,
                layer_idx=moe_layer_idx,
                window_cache_config=window_cache_config,
            )
            moe_layer_idx += 1
        elif _is_generic_router_moe_block(mlp):
            decoder_layer.mlp = WindowCacheTrackedGenericRouterMoEBlock(
                original_block=mlp,
                owner_model=model,
                layer_idx=moe_layer_idx,
                window_cache_config=window_cache_config,
            )
            moe_layer_idx += 1

    if moe_layer_idx == 0:
        raise ValueError("No sparse MoE blocks were found to patch with the window cache loss.")

    return model
