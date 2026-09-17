from __future__ import annotations

import types
import weakref
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from cache_moe import (
    _generic_router_logits,
    _generic_router_module,
    _generic_selected_experts_from_logits,
    _infer_generic_top_k,
    _is_generic_router_moe_block,
    _is_qwen3_moe_block,
    _router_weight,
)


@dataclass
class LRUCacheMetricConfig:
    enabled: bool = False
    cache_size: int = 20


class LRUTrackedQwen3MoeSparseBlock(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
        owner_model: nn.Module,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.experts = original_block.experts
        self.gate = original_block.gate
        self.layer_idx = layer_idx
        self._owner_model_ref = weakref.ref(owner_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.reshape(-1, hidden_dim)

        router_logits, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        owner_model = self._owner_model_ref()
        if owner_model is not None and getattr(owner_model, "_lru_cache_metrics_enabled", False):
            owner_model._lru_cache_metric_layer_stats.append(
                {
                    "layer_idx": self.layer_idx,
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


def _is_integer_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)


def _deepseek_selected_experts(gate: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    gate_outputs = gate(hidden_states)
    if not isinstance(gate_outputs, tuple):
        raise TypeError("DeepSeek MoE gate is expected to return a tuple.")
    if len(gate_outputs) >= 3:
        return gate_outputs[0]
    if len(gate_outputs) == 2:
        first, second = gate_outputs
        return first if _is_integer_tensor(first) else second
    raise TypeError(f"Unsupported DeepSeek MoE gate output length: {len(gate_outputs)}")


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


class LRUTrackedDeepseekV2MoEBlock(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
        owner_model: nn.Module,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.original_block = original_block
        self.layer_idx = layer_idx
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
        if owner_model is not None and getattr(owner_model, "_lru_cache_metrics_enabled", False):
            router_logits = _deepseek_router_logits(self.gate, hidden_states)
            selected_experts = _deepseek_selected_experts_from_logits(self.gate, router_logits)
            owner_model._lru_cache_metric_layer_stats.append(
                {
                    "layer_idx": self.layer_idx,
                    "promoe_inputs": hidden_states.detach(),
                    "router_logits": router_logits.reshape(batch_size, sequence_length, -1),
                    "selected_experts": selected_experts.reshape(batch_size, sequence_length, -1),
                }
            )

        return final_hidden_states


class LRUTrackedGenericRouterMoEBlock(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
        owner_model: nn.Module,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.original_block = original_block
        self.layer_idx = int(layer_idx)
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
        if owner_model is not None and getattr(owner_model, "_lru_cache_metrics_enabled", False):
            router_logits = _generic_router_logits(self.router, hidden_states, num_experts=self.num_experts)
            selected_experts, _ = _generic_selected_experts_from_logits(router_logits, top_k=self.top_k)
            owner_model._lru_cache_metric_layer_stats.append(
                {
                    "layer_idx": self.layer_idx,
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


def _contains_any(values: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    if candidates.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    return values.unsqueeze(-1).eq(candidates.unsqueeze(0)).any(dim=-1)


def _touch_lru(cache: torch.Tensor, expert: torch.Tensor, cache_size: int) -> torch.Tensor:
    if cache.numel() > 0:
        cache = cache[cache.ne(expert)]
    cache = torch.cat([cache, expert.reshape(1)], dim=0)
    if cache.numel() > cache_size:
        cache = cache[-cache_size:]
    return cache


def compute_lru_cache_metrics(
    layer_stats: List[Dict[str, torch.Tensor]],
    attention_mask: Optional[torch.Tensor],
    lru_config: LRUCacheMetricConfig,
    compute_hard_metrics: bool = True,
) -> Dict[str, torch.Tensor]:
    device = layer_stats[0]["selected_experts"].device if layer_stats else (
        attention_mask.device if attention_mask is not None else torch.device("cpu")
    )
    zero = torch.zeros((), device=device, dtype=torch.float32)
    if not layer_stats or not compute_hard_metrics:
        return {
            "hard_cache_hit_rate": zero,
            "hard_cache_miss_rate": zero,
            "hard_cache_overlap_count": zero,
        }

    total_hit_rate = zero
    total_overlap = zero
    counted_steps = 0

    for layer in layer_stats:
        selected_experts = layer["selected_experts"]
        batch_size, seq_len, _ = selected_experts.shape
        lengths = _valid_lengths(attention_mask, batch_size, seq_len)

        for batch_idx, length in enumerate(lengths):
            if length <= 0:
                continue

            cache = torch.empty(0, device=device, dtype=selected_experts.dtype)
            selected = selected_experts[batch_idx, :length, :]
            for token_idx in range(length):
                token_selected = selected[token_idx]
                hit_count = _contains_any(token_selected, cache).to(dtype=torch.float32).sum()
                total_overlap = total_overlap + hit_count
                total_hit_rate = total_hit_rate + hit_count / token_selected.numel()
                counted_steps += 1

                # LRU order is oldest -> newest. Every routed expert is touched after hit accounting.
                for expert in token_selected:
                    cache = _touch_lru(cache, expert.detach(), lru_config.cache_size)

    if counted_steps == 0:
        return {
            "hard_cache_hit_rate": zero,
            "hard_cache_miss_rate": zero,
            "hard_cache_overlap_count": zero,
        }

    denom = torch.tensor(float(counted_steps), device=device, dtype=torch.float32)
    hit_rate = total_hit_rate / denom
    return {
        "hard_cache_hit_rate": hit_rate,
        "hard_cache_miss_rate": 1.0 - hit_rate,
        "hard_cache_overlap_count": total_overlap / denom,
    }


def _reset_lru_cache_metric_state(self):
    self._lru_cache_metric_layer_stats = []


def _compute_lru_cache_metrics(
    self,
    attention_mask: Optional[torch.Tensor] = None,
    compute_hard_metrics: bool = True,
) -> Dict[str, torch.Tensor]:
    return compute_lru_cache_metrics(
        self._lru_cache_metric_layer_stats,
        attention_mask,
        self._lru_cache_metric_config,
        compute_hard_metrics=compute_hard_metrics,
    )


def install_lru_cache_metrics(model: nn.Module, lru_config: LRUCacheMetricConfig) -> nn.Module:
    if not lru_config.enabled:
        return model

    if getattr(model, "_cache_moe_enabled", False):
        raise ValueError("LRU baseline metrics must not be enabled together with future cache MoE.")

    if getattr(model, "_lru_cache_metrics_enabled", False):
        return model

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("LRU metric patching expects a decoder model with model.layers")

    model._lru_cache_metrics_enabled = True
    model._lru_cache_metric_config = lru_config
    model._lru_cache_metric_layer_stats = []
    model.reset_lru_cache_metric_state = types.MethodType(_reset_lru_cache_metric_state, model)
    model.compute_lru_cache_metrics = types.MethodType(_compute_lru_cache_metrics, model)

    moe_layer_idx = 0
    for decoder_layer in model.model.layers:
        mlp = getattr(decoder_layer, "mlp", None)
        if mlp is None:
            continue
        if _is_deepseek_v2_moe_block(mlp):
            decoder_layer.mlp = LRUTrackedDeepseekV2MoEBlock(
                original_block=mlp,
                owner_model=model,
                layer_idx=moe_layer_idx,
            )
            moe_layer_idx += 1
        elif _is_qwen3_moe_block(mlp):
            decoder_layer.mlp = LRUTrackedQwen3MoeSparseBlock(
                original_block=mlp,
                owner_model=model,
                layer_idx=moe_layer_idx,
            )
            moe_layer_idx += 1
        elif _is_generic_router_moe_block(mlp):
            decoder_layer.mlp = LRUTrackedGenericRouterMoEBlock(
                original_block=mlp,
                owner_model=model,
                layer_idx=moe_layer_idx,
            )
            moe_layer_idx += 1

    if moe_layer_idx == 0:
        raise ValueError("No sparse MoE blocks were found to patch with LRU cache metrics.")

    return model


def unwrap_lru_cache_metrics(model: nn.Module) -> nn.Module:
    if not getattr(model, "_lru_cache_metrics_enabled", False):
        return model

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return model

    for decoder_layer in model.model.layers:
        mlp = getattr(decoder_layer, "mlp", None)
        if isinstance(mlp, LRUTrackedQwen3MoeSparseBlock):
            continue
        if isinstance(mlp, (LRUTrackedDeepseekV2MoEBlock, LRUTrackedGenericRouterMoEBlock)):
            decoder_layer.mlp = mlp.original_block

    for attr in (
        "_lru_cache_metrics_enabled",
        "_lru_cache_metric_config",
        "_lru_cache_metric_layer_stats",
        "reset_lru_cache_metric_state",
        "compute_lru_cache_metrics",
    ):
        if hasattr(model, attr):
            delattr(model, attr)

    return model
