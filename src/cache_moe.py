from __future__ import annotations

import types
import weakref
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class CacheMoEConfig:
    enabled: bool = False
    cache_size: int = 20
    temperature: float = 0.25
    balance_weight: float = 0.01
    swap_weight: float = 0.10
    future_gate_type: str = "linear"
    future_gate_hidden_dim: int = 0
    future_gate_target_params_per_layer: int = 2_000_000
    copy_current_router_init: bool = True
    future_cache_priority_mode: str = "current_plus_future"
    spatio_temporal_enabled: bool = False
    spatio_only_enabled: bool = False
    swap_loss_future_gate_only: bool = False
    split_grad_clip: bool = False
    layer_swap_weight_schedule: str = "uniform"
    layer_swap_weight_min: float = 1.0
    layer_swap_weight_max: float = 1.0
    layer_swap_weights: Optional[List[float]] = None
    normalize_layer_swap_weights: bool = True
    layer_adaptive_swap_weight: bool = False
    layer_adaptive_weight_min: float = 0.5
    layer_adaptive_weight_max: float = 1.5
    layer_adaptive_warmup_steps: int = 0
    layer_adaptive_update_interval: int = 20
    layer_adaptive_lr: float = 0.02
    layer_adaptive_cosine_ema: float = 0.95
    layer_adaptive_cosine_scale: float = 1.0
    layer_adaptive_cosine_deadband: float = 0.0
    layer_adaptive_history: bool = True


def _spatio_router_enabled(cache_moe_config: CacheMoEConfig) -> bool:
    return bool(cache_moe_config.spatio_temporal_enabled or cache_moe_config.spatio_only_enabled)


def _temporal_router_enabled(cache_moe_config: CacheMoEConfig) -> bool:
    return not bool(cache_moe_config.spatio_only_enabled)


def _make_linear(
    in_features: int,
    out_features: int,
    device: torch.device,
    dtype: torch.dtype,
    bias: bool = False,
) -> nn.Linear:
    try:
        return nn.Linear(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
    except TypeError:
        return nn.Linear(in_features, out_features, bias=bias).to(device=device, dtype=dtype)


class FutureGateMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            _make_linear(in_features, self.hidden_dim, device=device, dtype=dtype, bias=True),
            nn.ReLU(),
            _make_linear(self.hidden_dim, out_features, device=device, dtype=dtype, bias=True),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


def infer_future_gate_mlp_hidden_dim(
    input_dim: int,
    num_experts: int,
    target_params: int = 2_000_000,
) -> int:
    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive.")
    target_params = max(int(target_params), int(num_experts) + 1)
    # Two-layer MLP with bias: input*hidden + hidden + hidden*num_experts + num_experts.
    return max(1, int(round((target_params - num_experts) / float(input_dim + num_experts + 1))))


def _make_future_gate(
    in_features: int,
    out_features: int,
    device: torch.device,
    dtype: torch.dtype,
    cache_moe_config: CacheMoEConfig,
) -> nn.Module:
    gate_type = str(cache_moe_config.future_gate_type or "linear").lower()
    if gate_type in {"linear", "router_linear"}:
        return _make_linear(in_features, out_features, device=device, dtype=dtype, bias=False)
    if gate_type in {"mlp", "two_layer_mlp"}:
        hidden_dim = int(cache_moe_config.future_gate_hidden_dim or 0)
        if hidden_dim <= 0:
            hidden_dim = infer_future_gate_mlp_hidden_dim(
                input_dim=in_features,
                num_experts=out_features,
                target_params=int(cache_moe_config.future_gate_target_params_per_layer),
            )
        return FutureGateMLP(
            in_features=in_features,
            out_features=out_features,
            hidden_dim=hidden_dim,
            device=device,
            dtype=dtype,
        )
    raise ValueError(f"Unsupported future_gate_type: {cache_moe_config.future_gate_type!r}")


def _can_copy_current_router_init(cache_moe_config: CacheMoEConfig, future_gate: nn.Module) -> bool:
    gate_type = str(cache_moe_config.future_gate_type or "linear").lower()
    return gate_type in {"linear", "router_linear"} and isinstance(future_gate, nn.Linear)


def _maybe_register_layer_adaptive_cosine_hooks(
    cache_moe_config: CacheMoEConfig,
    layer_idx: int,
    task_probe: torch.Tensor,
    future_logits: torch.Tensor,
    selected_experts: Optional[torch.Tensor] = None,
    num_experts: Optional[int] = None,
) -> None:
    if not getattr(cache_moe_config, "layer_adaptive_swap_weight", False):
        return
    if not getattr(cache_moe_config, "_layer_adaptive_capture_cosine", False):
        return
    if not task_probe.requires_grad or not future_logits.requires_grad:
        return

    def save_grad(kind: str):
        def hook(grad: torch.Tensor) -> None:
            probe = getattr(cache_moe_config, "_layer_adaptive_grad_probe", None)
            if probe is None:
                probe = {}
                cache_moe_config._layer_adaptive_grad_probe = probe
            layer_probe = probe.setdefault(int(layer_idx), {})
            captured_grad = grad.detach().float()
            if kind == "current" and selected_experts is not None and num_experts is not None:
                selected = selected_experts.detach().to(device=captured_grad.device, dtype=torch.long)
                if captured_grad.dim() == 2 and selected.shape == captured_grad.shape:
                    dense_grad = torch.zeros(
                        captured_grad.size(0),
                        int(num_experts),
                        device=captured_grad.device,
                        dtype=torch.float32,
                    )
                    captured_grad = dense_grad.scatter_add(1, selected, captured_grad)
            layer_probe[kind] = captured_grad
            current_grad = layer_probe.get("current")
            future_grad = layer_probe.get("future")
            if current_grad is None or future_grad is None:
                return None
            current_flat = current_grad.reshape(-1)
            future_flat = future_grad.reshape(-1)
            denom = current_flat.norm() * future_flat.norm()
            if torch.isfinite(denom) and denom.item() >= 1e-12:
                cosine = torch.dot(current_flat, future_flat) / denom.clamp_min(1e-12)
                if torch.isfinite(cosine):
                    pending = getattr(cache_moe_config, "_layer_adaptive_pending_cosine", None)
                    if pending is None:
                        pending = {}
                        cache_moe_config._layer_adaptive_pending_cosine = pending
                    pending[int(layer_idx)] = float(cosine.detach().cpu().item())
            layer_probe.pop("current", None)
            layer_probe.pop("future", None)
            return None

        return hook

    task_probe.register_hook(save_grad("current"))
    future_logits.register_hook(save_grad("future"))


class CacheAwareQwen3MoeSparseBlock(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
        owner_model: nn.Module,
        layer_idx: int,
        cache_moe_config: CacheMoEConfig,
    ) -> None:
        super().__init__()
        self.experts = original_block.experts
        self.gate = original_block.gate
        self.layer_idx = layer_idx
        self.cache_moe_config = cache_moe_config
        self.hidden_dim = self.gate.hidden_dim
        self.num_experts = self.gate.num_experts
        gate_weight = getattr(self.gate, "weight", None)
        if gate_weight is not None:
            future_gate_dtype = gate_weight.dtype
            future_gate_device = gate_weight.device
        else:
            reference_param = next(original_block.parameters())
            future_gate_dtype = reference_param.dtype
            future_gate_device = reference_param.device
        self.future_gate = (
            _make_future_gate(
                in_features=self.hidden_dim,
                out_features=self.num_experts,
                device=future_gate_device,
                dtype=future_gate_dtype,
                cache_moe_config=cache_moe_config,
            )
            if _temporal_router_enabled(cache_moe_config)
            else None
        )
        self.spatio_gate = (
            _make_linear(
                self.hidden_dim,
                self.num_experts,
                device=future_gate_device,
                dtype=future_gate_dtype,
                bias=False,
            )
            if _spatio_router_enabled(cache_moe_config)
            else None
        )
        self._owner_model_ref = weakref.ref(owner_model)

        if (
            cache_moe_config.copy_current_router_init
            and hasattr(self.gate, "weight")
            and self.future_gate is not None
            and _can_copy_current_router_init(cache_moe_config, self.future_gate)
        ):
            with torch.no_grad():
                self.future_gate.weight.copy_(self.gate.weight)
        if cache_moe_config.copy_current_router_init and self.spatio_gate is not None and hasattr(self.gate, "weight"):
            with torch.no_grad():
                self.spatio_gate.weight.copy_(self.gate.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.reshape(-1, hidden_dim)

        router_logits, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        future_hidden_states = (
            hidden_states_reshaped.detach()
            if self.cache_moe_config.swap_loss_future_gate_only
            else hidden_states_reshaped
        )
        future_logits = self.future_gate(future_hidden_states) if self.future_gate is not None else None
        spatio_logits = self.spatio_gate(future_hidden_states) if self.spatio_gate is not None else None
        if future_logits is not None:
            _maybe_register_layer_adaptive_cosine_hooks(
                self.cache_moe_config,
                self.layer_idx,
                routing_weights,
                future_logits,
                selected_experts=selected_experts,
                num_experts=self.num_experts,
            )
        future_probs = (
            F.softmax(future_logits, dim=-1, dtype=torch.float32) if future_logits is not None else None
        )
        spatio_probs = (
            F.softmax(spatio_logits, dim=-1, dtype=torch.float32) if spatio_logits is not None else None
        )
        current_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        owner_model = self._owner_model_ref()
        if owner_model is not None and getattr(owner_model, "_cache_moe_enabled", False):
            layer_record = {
                "layer_idx": self.layer_idx,
                "current_probs": current_probs.reshape(batch_size, sequence_length, self.num_experts),
                "promoe_inputs": hidden_states.detach(),
                "router_logits": router_logits.reshape(batch_size, sequence_length, self.num_experts),
                "selected_experts": selected_experts.reshape(batch_size, sequence_length, -1),
            }
            if future_probs is not None and future_logits is not None:
                layer_record["future_probs"] = future_probs.reshape(
                    batch_size,
                    sequence_length,
                    self.num_experts,
                )
                layer_record["future_logits"] = future_logits.reshape(
                    batch_size,
                    sequence_length,
                    self.num_experts,
                )
            if spatio_probs is not None and spatio_logits is not None:
                layer_record["spatio_probs"] = spatio_probs.reshape(
                    batch_size,
                    sequence_length,
                    self.num_experts,
                )
                layer_record["spatio_logits"] = spatio_logits.reshape(
                    batch_size,
                    sequence_length,
                    self.num_experts,
                )
            owner_model._cache_moe_layer_stats.append(layer_record)

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


def _router_weight(router: nn.Module) -> Optional[torch.Tensor]:
    weight = getattr(router, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight
    for name in ("linear", "gate", "router", "classifier"):
        child = getattr(router, name, None)
        weight = getattr(child, "weight", None)
        if isinstance(weight, torch.Tensor):
            return weight
    return None


def _generic_router_module(block: nn.Module) -> Optional[nn.Module]:
    for name in ("router", "gate"):
        router = getattr(block, name, None)
        if isinstance(router, nn.Module) and _router_weight(router) is not None:
            return router
    return None


def _is_qwen3_moe_block(block: nn.Module) -> bool:
    gate = getattr(block, "gate", None)
    return (
        hasattr(block, "experts")
        and gate is not None
        and hasattr(gate, "hidden_dim")
        and hasattr(gate, "num_experts")
    )


def _is_generic_router_moe_block(block: nn.Module) -> bool:
    if _is_deepseek_v2_moe_block(block) or _is_qwen3_moe_block(block):
        return False
    return _generic_router_module(block) is not None


def _infer_generic_top_k(block: nn.Module, router: nn.Module, owner_model: Optional[nn.Module] = None) -> int:
    config = getattr(block, "config", None) or getattr(owner_model, "config", None)
    for source in (block, router, config):
        if source is None:
            continue
        for name in ("top_k", "num_experts_per_tok", "experts_per_token", "num_experts_per_token"):
            value = getattr(source, name, None)
            if value is not None:
                return max(1, int(value))
    return 1


def _generic_router_logits(
    router: nn.Module,
    hidden_states: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_flat = hidden_states.reshape(batch_size * sequence_length, hidden_dim)
    try:
        router_outputs = router(hidden_flat)
    except TypeError:
        router_outputs = router(hidden_states)
    if isinstance(router_outputs, tuple):
        tensor_outputs = [item for item in router_outputs if isinstance(item, torch.Tensor)]
        for tensor in tensor_outputs:
            if tensor.shape[-1] == num_experts and tensor.dtype.is_floating_point:
                router_outputs = tensor
                break
        else:
            raise TypeError("Generic MoE router returned no floating logits tensor with num_experts as last dim.")
    if not isinstance(router_outputs, torch.Tensor):
        raise TypeError(f"Generic MoE router returned unsupported type: {type(router_outputs)!r}.")
    if router_outputs.shape[-1] != num_experts:
        raise ValueError(
            "Generic MoE router logits last dimension does not match num_experts: "
            f"logits={tuple(router_outputs.shape)}, num_experts={num_experts}."
        )
    return router_outputs.reshape(batch_size * sequence_length, num_experts)


def _generic_selected_experts_from_logits(
    router_logits: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = F.softmax(router_logits.to(dtype=torch.float32), dim=-1)
    routing_weights, selected_experts = torch.topk(scores, k=top_k, dim=-1, sorted=False)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return selected_experts.to(dtype=torch.long), routing_weights


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


def _deepseek_gate_selection_from_logits(
    gate: nn.Module,
    router_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = _deepseek_scores_from_logits(gate, router_logits)
    top_k = _deepseek_top_k(gate)
    topk_method = str(getattr(gate, "topk_method", "greedy")).lower()
    selection_scores = scores
    correction_bias = getattr(gate, "e_score_correction_bias", None)
    if correction_bias is not None:
        selection_scores = selection_scores + correction_bias.to(device=scores.device, dtype=scores.dtype)

    if topk_method in {"greedy", "gready"}:
        _, selected_experts = torch.topk(selection_scores, k=top_k, dim=-1, sorted=False)
    elif topk_method == "group_limited_greedy":
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
        _, selected_experts = torch.topk(masked_scores, k=top_k, dim=-1, sorted=False)
    else:
        raise ValueError(f"Unsupported DeepSeek MoE topk_method: {topk_method!r}")

    routing_weights = scores.gather(dim=-1, index=selected_experts)
    if top_k > 1 and bool(getattr(gate, "norm_topk_prob", False)):
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    scale = float(getattr(gate, "routed_scaling_factor", 1.0))
    if scale != 1.0:
        routing_weights = routing_weights * scale
    return selected_experts.to(dtype=torch.long), routing_weights


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


class CacheAwareDeepseekV2MoEBlock(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
        owner_model: nn.Module,
        layer_idx: int,
        cache_moe_config: CacheMoEConfig,
    ) -> None:
        super().__init__()
        self.original_block = original_block
        self.layer_idx = layer_idx
        self.cache_moe_config = cache_moe_config
        self._owner_model_ref = weakref.ref(owner_model)

        gate_weight = original_block.gate.weight
        self.hidden_dim = gate_weight.shape[1]
        self.num_experts = gate_weight.shape[0]
        self.future_gate = (
            _make_future_gate(
                in_features=self.hidden_dim,
                out_features=self.num_experts,
                device=gate_weight.device,
                dtype=gate_weight.dtype,
                cache_moe_config=cache_moe_config,
            )
            if _temporal_router_enabled(cache_moe_config)
            else None
        )
        self.spatio_gate = (
            _make_linear(
                self.hidden_dim,
                self.num_experts,
                device=gate_weight.device,
                dtype=gate_weight.dtype,
                bias=False,
            )
            if _spatio_router_enabled(cache_moe_config)
            else None
        )
        if (
            cache_moe_config.copy_current_router_init
            and self.future_gate is not None
            and _can_copy_current_router_init(cache_moe_config, self.future_gate)
        ):
            with torch.no_grad():
                self.future_gate.weight.copy_(gate_weight)
        if cache_moe_config.copy_current_router_init and self.spatio_gate is not None:
            with torch.no_grad():
                self.spatio_gate.weight.copy_(gate_weight)

    @property
    def experts(self):
        return self.original_block.experts

    @property
    def gate(self):
        return self.original_block.gate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        final_hidden_states = self.original_block(hidden_states)

        owner_model = self._owner_model_ref()
        if owner_model is not None and getattr(owner_model, "_cache_moe_enabled", False):
            router_logits = _deepseek_router_logits(self.gate, hidden_states)
            selected_experts, routing_weights = _deepseek_gate_selection_from_logits(self.gate, router_logits)
            probe_hidden_states = hidden_states.reshape(-1, hidden_dim)
            future_hidden_states = (
                probe_hidden_states.detach()
                if self.cache_moe_config.swap_loss_future_gate_only
                else probe_hidden_states
            )
            future_logits = self.future_gate(future_hidden_states) if self.future_gate is not None else None
            spatio_logits = self.spatio_gate(future_hidden_states) if self.spatio_gate is not None else None
            layer_record = {
                "layer_idx": self.layer_idx,
                "current_probs": _sparse_probs_from_topk(
                    selected_experts,
                    routing_weights,
                    num_experts=self.num_experts,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                ),
                "promoe_inputs": hidden_states.detach(),
                "router_logits": router_logits.reshape(batch_size, sequence_length, self.num_experts),
                "selected_experts": selected_experts.reshape(batch_size, sequence_length, -1),
            }
            if future_logits is not None:
                layer_record["future_probs"] = F.softmax(
                    future_logits,
                    dim=-1,
                    dtype=torch.float32,
                ).reshape(batch_size, sequence_length, self.num_experts)
                layer_record["future_logits"] = future_logits.reshape(
                    batch_size,
                    sequence_length,
                    self.num_experts,
                )
            if spatio_logits is not None:
                layer_record["spatio_probs"] = F.softmax(
                    spatio_logits,
                    dim=-1,
                    dtype=torch.float32,
                ).reshape(batch_size, sequence_length, self.num_experts)
                layer_record["spatio_logits"] = spatio_logits.reshape(
                    batch_size,
                    sequence_length,
                    self.num_experts,
                )
            owner_model._cache_moe_layer_stats.append(layer_record)

        return final_hidden_states


class CacheAwareGenericRouterMoEBlock(nn.Module):
    """Record router/top-k stats for MoE blocks whose expert forward is model-specific.

    GPT-OSS keeps the actual expert computation inside the original block. This wrapper
    leaves that path untouched and only adds the project-specific future/spatio heads
    plus cache metric traces.
    """

    def __init__(
        self,
        original_block: nn.Module,
        owner_model: nn.Module,
        layer_idx: int,
        cache_moe_config: CacheMoEConfig,
    ) -> None:
        super().__init__()
        self.original_block = original_block
        self.layer_idx = int(layer_idx)
        self.cache_moe_config = cache_moe_config
        self._owner_model_ref = weakref.ref(owner_model)

        router = _generic_router_module(original_block)
        if router is None:
            raise ValueError("Generic MoE block has no supported router/gate module.")
        router_weight = _router_weight(router)
        if router_weight is None:
            raise ValueError("Generic MoE router has no usable weight for shape inference.")
        self._router_attr_name = "router" if getattr(original_block, "router", None) is router else "gate"
        self.hidden_dim = int(router_weight.shape[1])
        self.num_experts = int(router_weight.shape[0])
        self.top_k = _infer_generic_top_k(original_block, router, owner_model=owner_model)
        self.future_gate = (
            _make_future_gate(
                in_features=self.hidden_dim,
                out_features=self.num_experts,
                device=router_weight.device,
                dtype=router_weight.dtype,
                cache_moe_config=cache_moe_config,
            )
            if _temporal_router_enabled(cache_moe_config)
            else None
        )
        self.spatio_gate = (
            _make_linear(
                self.hidden_dim,
                self.num_experts,
                device=router_weight.device,
                dtype=router_weight.dtype,
                bias=False,
            )
            if _spatio_router_enabled(cache_moe_config)
            else None
        )
        if (
            cache_moe_config.copy_current_router_init
            and self.future_gate is not None
            and _can_copy_current_router_init(cache_moe_config, self.future_gate)
        ):
            with torch.no_grad():
                self.future_gate.weight.copy_(router_weight.to(device=self.future_gate.weight.device, dtype=self.future_gate.weight.dtype))
        if cache_moe_config.copy_current_router_init and self.spatio_gate is not None:
            with torch.no_grad():
                self.spatio_gate.weight.copy_(router_weight.to(device=self.spatio_gate.weight.device, dtype=self.spatio_gate.weight.dtype))

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
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        final_hidden_states = self.original_block(hidden_states, *args, **kwargs)

        owner_model = self._owner_model_ref()
        if owner_model is not None and getattr(owner_model, "_cache_moe_enabled", False):
            router_logits = _generic_router_logits(self.router, hidden_states, num_experts=self.num_experts)
            selected_experts, routing_weights = _generic_selected_experts_from_logits(
                router_logits,
                top_k=self.top_k,
            )
            probe_hidden_states = hidden_states.reshape(-1, hidden_dim)
            future_hidden_states = (
                probe_hidden_states.detach()
                if self.cache_moe_config.swap_loss_future_gate_only
                else probe_hidden_states
            )
            future_logits = self.future_gate(future_hidden_states) if self.future_gate is not None else None
            spatio_logits = self.spatio_gate(future_hidden_states) if self.spatio_gate is not None else None
            layer_record = {
                "layer_idx": self.layer_idx,
                "current_probs": F.softmax(router_logits, dim=-1, dtype=torch.float32).reshape(
                    batch_size,
                    sequence_length,
                    self.num_experts,
                ),
                "promoe_inputs": hidden_states.detach(),
                "router_logits": router_logits.reshape(batch_size, sequence_length, self.num_experts),
                "selected_experts": selected_experts.reshape(batch_size, sequence_length, -1),
            }
            if future_logits is not None:
                layer_record["future_probs"] = F.softmax(
                    future_logits,
                    dim=-1,
                    dtype=torch.float32,
                ).reshape(batch_size, sequence_length, self.num_experts)
                layer_record["future_logits"] = future_logits.reshape(
                    batch_size,
                    sequence_length,
                    self.num_experts,
                )
            if spatio_logits is not None:
                layer_record["spatio_probs"] = F.softmax(
                    spatio_logits,
                    dim=-1,
                    dtype=torch.float32,
                ).reshape(batch_size, sequence_length, self.num_experts)
                layer_record["spatio_logits"] = spatio_logits.reshape(batch_size, sequence_length, self.num_experts)
            owner_model._cache_moe_layer_stats.append(layer_record)

        return final_hidden_states


def _soft_cache_state(priority_scores: torch.Tensor, cache_size: int, temperature: float) -> torch.Tensor:
    num_experts = priority_scores.size(-1)
    k = max(1, min(cache_size, num_experts))
    threshold = torch.topk(priority_scores, k=k, dim=-1).values[..., -1]
    return torch.sigmoid((priority_scores - threshold.unsqueeze(-1)) / temperature)


def hard_cache_selection(priority_scores: torch.Tensor, cache_size: int) -> torch.Tensor:
    num_experts = priority_scores.size(-1)
    k = max(1, min(cache_size, num_experts))
    return torch.topk(priority_scores, k=k, dim=-1).indices


def future_cache_priority_scores(
    current_probs: Optional[torch.Tensor],
    future_probs: torch.Tensor,
    cache_moe_config: CacheMoEConfig,
) -> torch.Tensor:
    mode = str(getattr(cache_moe_config, "future_cache_priority_mode", "current_plus_future") or "").lower()
    if mode in {"future_only", "future", "mlp_only", "predictor_only"}:
        return future_probs
    if mode not in {"current_plus_future", "current+future", "hybrid", "sum"}:
        raise ValueError(f"Unsupported future_cache_priority_mode: {cache_moe_config.future_cache_priority_mode!r}")
    if current_probs is None:
        return future_probs
    return current_probs + future_probs


def next_cache_from_probs(
    current_probs: torch.Tensor,
    future_probs: torch.Tensor,
    cache_size: int,
    cache_moe_config: Optional[CacheMoEConfig] = None,
) -> torch.Tensor:
    if cache_moe_config is None:
        return hard_cache_selection(current_probs + future_probs, cache_size=cache_size)
    priority_scores = future_cache_priority_scores(
        current_probs,
        future_probs,
        cache_moe_config,
    )
    return hard_cache_selection(priority_scores, cache_size=cache_size)


def _contains_any(values: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    if candidates.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    return values.unsqueeze(-1).eq(candidates.unsqueeze(0)).any(dim=-1)


def _fill_cache_from_future(
    cache: torch.Tensor,
    future_scores: torch.Tensor,
    cache_size: int,
) -> torch.Tensor:
    num_experts = future_scores.size(-1)
    k = max(1, min(cache_size, num_experts))
    if cache.numel() >= k:
        return cache[:k]

    ranked_experts = torch.argsort(future_scores, descending=True)
    fill_candidates = ranked_experts[~_contains_any(ranked_experts, cache)]
    needed = k - cache.numel()
    return torch.cat([cache, fill_candidates[:needed]], dim=0)


def _top_priority_cache(
    priority_scores: torch.Tensor,
    cache_size: int,
) -> torch.Tensor:
    num_experts = priority_scores.size(-1)
    k = max(1, min(cache_size, num_experts))
    return torch.topk(priority_scores.detach(), k=k, largest=True).indices


def _initial_inference_cache(
    selected_experts: torch.Tensor,
    future_scores: torch.Tensor,
    cache_size: int,
    current_scores: Optional[torch.Tensor] = None,
    cache_moe_config: Optional[CacheMoEConfig] = None,
) -> torch.Tensor:
    initial_cache = torch.unique(selected_experts.detach())
    if cache_moe_config is not None:
        future_scores = future_cache_priority_scores(
            current_scores.detach() if current_scores is not None else None,
            future_scores.detach(),
            cache_moe_config,
        )
    return _fill_cache_from_future(initial_cache, future_scores.detach(), cache_size)


def _update_inference_cache(
    cache: torch.Tensor,
    selected_experts: torch.Tensor,
    future_scores: torch.Tensor,
    cache_size: int,
    current_scores: Optional[torch.Tensor] = None,
    cache_moe_config: Optional[CacheMoEConfig] = None,
) -> torch.Tensor:
    selected = torch.unique(selected_experts.detach())
    if cache_moe_config is not None:
        future_scores = future_cache_priority_scores(
            current_scores.detach() if current_scores is not None else None,
            future_scores.detach(),
            cache_moe_config,
        )
    future_scores = future_scores.detach()
    miss_mask = ~_contains_any(selected, cache)
    misses = selected[miss_mask]
    if misses.numel() == 0:
        return cache

    protected_mask = _contains_any(cache, selected)
    eviction_candidates = cache[~protected_mask]
    evict_count = min(misses.numel(), eviction_candidates.numel())
    if evict_count > 0:
        eviction_scores = future_scores[eviction_candidates]
        evicted = eviction_candidates[torch.topk(eviction_scores, k=evict_count, largest=False).indices]
        cache = cache[~_contains_any(cache, evicted)]
        cache = torch.cat([cache, misses[:evict_count]], dim=0)

    return _fill_cache_from_future(cache, future_scores, cache_size)


def _valid_lengths(attention_mask: Optional[torch.Tensor], batch_size: int, seq_len: int) -> List[int]:
    if attention_mask is None:
        return [seq_len] * batch_size
    return attention_mask.to(dtype=torch.long).sum(dim=-1).tolist()


def _scheduled_layer_weights(layer_count: int, cache_moe_config: CacheMoEConfig) -> List[float]:
    if layer_count <= 0:
        return []

    adaptive_weights = getattr(cache_moe_config, "_layer_adaptive_weights", None)
    if cache_moe_config.layer_adaptive_swap_weight and adaptive_weights:
        weights = [float(value) for value in adaptive_weights]
        if len(weights) < layer_count:
            weights.extend([1.0] * (layer_count - len(weights)))
        return weights[:layer_count]

    explicit_weights = cache_moe_config.layer_swap_weights
    if explicit_weights:
        weights = [float(value) for value in explicit_weights]
        if len(weights) < layer_count:
            weights.extend([weights[-1]] * (layer_count - len(weights)))
        weights = weights[:layer_count]
    else:
        schedule = cache_moe_config.layer_swap_weight_schedule
        min_weight = float(cache_moe_config.layer_swap_weight_min)
        max_weight = float(cache_moe_config.layer_swap_weight_max)
        if layer_count == 1:
            positions = [0.0]
        else:
            positions = [idx / float(layer_count - 1) for idx in range(layer_count)]

        if schedule == "uniform":
            weights = [1.0] * layer_count
        elif schedule == "late_heavy":
            weights = [min_weight + (max_weight - min_weight) * pos for pos in positions]
        elif schedule == "early_heavy":
            weights = [max_weight - (max_weight - min_weight) * pos for pos in positions]
        else:
            raise ValueError(
                "Unsupported layer_swap_weight_schedule: "
                f"{schedule!r}. Expected one of: uniform, late_heavy, early_heavy."
            )

    if cache_moe_config.normalize_layer_swap_weights:
        mean_weight = sum(weights) / max(len(weights), 1)
        if mean_weight > 0.0:
            weights = [weight / mean_weight for weight in weights]
    return weights


def _ensure_layer_adaptive_state(cache_moe_config: CacheMoEConfig, layer_count: int) -> None:
    if not cache_moe_config.layer_adaptive_swap_weight:
        return
    current = getattr(cache_moe_config, "_layer_adaptive_weights", None)
    if current is not None and len(current) == layer_count:
        return
    base_weights = _scheduled_layer_weights(layer_count, cache_moe_config)
    if not base_weights:
        base_weights = [1.0] * layer_count
    cache_moe_config._layer_adaptive_weights = [float(value) for value in base_weights[:layer_count]]
    cache_moe_config._layer_adaptive_update_count = 0
    cache_moe_config._layer_adaptive_cosine_ema = None
    cache_moe_config._layer_adaptive_last_cosine = None
    cache_moe_config._layer_adaptive_last_score = None


def _zero_cache_moe_metrics(device: torch.device) -> Dict[str, torch.Tensor]:
    zero = torch.zeros((), device=device, dtype=torch.float32)
    return {
        "swap_loss": zero,
        "cache_hit_rate": zero,
        "cache_retention_rate": zero,
        "hard_cache_hit_rate": zero,
        "hard_cache_miss_rate": zero,
        "hard_cache_overlap_count": zero,
        "future_entropy": zero,
        "layer_swap_weight": zero,
    }


def _compute_spatio_temporal_cache_moe_loss(
    layer_stats: List[Dict[str, torch.Tensor]],
    attention_mask: Optional[torch.Tensor],
    cache_moe_config: CacheMoEConfig,
    compute_hard_metrics: bool = True,
) -> Dict[str, torch.Tensor]:
    spatio_only = bool(cache_moe_config.spatio_only_enabled)
    device = layer_stats[0]["current_probs"].device if layer_stats else (
        attention_mask.device if attention_mask is not None else torch.device("cpu")
    )
    zero = torch.zeros((), device=device, dtype=torch.float32)
    if not layer_stats:
        result = _zero_cache_moe_metrics(device)
        result.update(
            {
                "hybrid_cache_hit_rate": zero,
                "temporal_cache_hit_rate": zero,
                "current_cache_hit_rate": zero,
                "spatio_cache_hit_rate": zero,
                "spatio_entropy": zero,
            }
        )
        return result

    layers = {int(layer["layer_idx"]): layer for layer in layer_stats}
    layer_ids = sorted(layers)
    layer_count = max(layer_ids) + 1
    if len(layer_ids) != layer_count:
        raise ValueError("Spatio-temporal Future Router expects contiguous MoE layer_idx values.")
    if any("spatio_probs" not in layers[layer_idx] for layer_idx in layer_ids):
        raise ValueError("spatio_temporal_enabled=true requires recorded spatio_probs for every MoE layer.")

    _ensure_layer_adaptive_state(cache_moe_config, layer_count)
    layer_swap_weights = _scheduled_layer_weights(layer_count, cache_moe_config)
    per_layer_swap = [zero for _ in range(layer_count)]
    per_layer_weighted_swap = [zero for _ in range(layer_count)]
    per_layer_hits = [zero for _ in range(layer_count)]
    per_layer_retention = [zero for _ in range(layer_count)]
    per_layer_entropy = [zero for _ in range(layer_count)]
    per_layer_steps = [0 for _ in range(layer_count)]

    total_swap = zero
    total_hits = zero
    total_temporal_hits = zero
    total_current_hits = zero
    total_spatio_hits = zero
    total_retention = zero
    total_hard_hit_rate = zero
    total_hard_overlap = zero
    total_entropy = zero
    total_spatio_entropy = zero
    total_layer_weight = zero
    counted_steps = 0
    counted_hard_steps = 0

    for layer_idx in layer_ids:
        layer = layers[layer_idx]
        layer_weight = layer_swap_weights[layer_idx]
        current_probs = layer["current_probs"]
        if cache_moe_config.swap_loss_future_gate_only:
            current_probs = current_probs.detach()
        temporal_probs = layer.get("future_probs")
        if not spatio_only and temporal_probs is None:
            raise ValueError("spatio_temporal_enabled=true requires future_probs for every MoE layer.")
        selected_experts = layer.get("selected_experts")
        spatio_source_layer = layers[layer_count - 1] if layer_idx == 0 else layers[layer_idx - 1]
        spatio_probs = spatio_source_layer["spatio_probs"]

        batch_size, seq_len, _ = current_probs.shape
        lengths = _valid_lengths(attention_mask, batch_size, seq_len)

        for batch_idx, length in enumerate(lengths):
            if length <= 1:
                continue

            curr = current_probs[batch_idx, :length, :]
            temporal = temporal_probs[batch_idx, :length, :] if temporal_probs is not None else None
            if layer_idx == 0:
                # Previous token last layer predicts next token layer 0, matching ProMoE's first-layer bridge.
                spatio = spatio_probs[batch_idx, : length - 1, :]
            else:
                # Current token layer l-1 predicts current token layer l before layer l is accessed.
                spatio = spatio_probs[batch_idx, 1:length, :]

            current_priority = curr[:-1]
            temporal_priority = (
                current_priority if spatio_only else current_priority + temporal[:-1]
            )
            spatio_priority = spatio
            priority_scores = temporal_priority + spatio_priority
            cache_state = _soft_cache_state(
                priority_scores,
                cache_size=cache_moe_config.cache_size,
                temperature=cache_moe_config.temperature,
            )
            temporal_cache_state = _soft_cache_state(
                temporal_priority,
                cache_size=cache_moe_config.cache_size,
                temperature=cache_moe_config.temperature,
            )
            current_cache_state = _soft_cache_state(
                current_priority,
                cache_size=cache_moe_config.cache_size,
                temperature=cache_moe_config.temperature,
            )
            spatio_cache_state = _soft_cache_state(
                spatio_priority,
                cache_size=cache_moe_config.cache_size,
                temperature=cache_moe_config.temperature,
            )
            next_current = curr[1:]

            swap_per_step = (next_current * (1.0 - cache_state)).sum(dim=-1)
            hit_per_step = (next_current * cache_state).sum(dim=-1)
            temporal_hit_per_step = (next_current * temporal_cache_state).sum(dim=-1)
            current_hit_per_step = (next_current * current_cache_state).sum(dim=-1)
            spatio_hit_per_step = (next_current * spatio_cache_state).sum(dim=-1)
            swap_contribution = swap_per_step * layer_weight
            entropy_per_step = (
                torch.zeros_like(swap_per_step)
                if temporal is None
                else (-(temporal[:-1] * (temporal[:-1].clamp_min(1e-8)).log()).sum(dim=-1))
            )
            spatio_entropy_per_step = (-(spatio * (spatio.clamp_min(1e-8)).log()).sum(dim=-1))

            total_swap = total_swap + swap_contribution.sum()
            total_hits = total_hits + hit_per_step.sum()
            if not spatio_only:
                total_temporal_hits = total_temporal_hits + temporal_hit_per_step.sum()
            total_current_hits = total_current_hits + current_hit_per_step.sum()
            total_spatio_hits = total_spatio_hits + spatio_hit_per_step.sum()
            total_retention = total_retention + cache_state.mean(dim=-1).sum()
            total_entropy = total_entropy + entropy_per_step.sum()
            total_spatio_entropy = total_spatio_entropy + spatio_entropy_per_step.sum()
            total_layer_weight = total_layer_weight + torch.tensor(
                float(layer_weight * (length - 1)),
                device=device,
                dtype=torch.float32,
            )
            counted_steps += length - 1
            per_layer_swap[layer_idx] = per_layer_swap[layer_idx] + swap_per_step.sum()
            per_layer_weighted_swap[layer_idx] = per_layer_weighted_swap[layer_idx] + swap_contribution.sum()
            per_layer_hits[layer_idx] = per_layer_hits[layer_idx] + hit_per_step.sum()
            per_layer_retention[layer_idx] = per_layer_retention[layer_idx] + cache_state.mean(dim=-1).sum()
            per_layer_entropy[layer_idx] = per_layer_entropy[layer_idx] + entropy_per_step.sum()
            per_layer_steps[layer_idx] += length - 1

            if compute_hard_metrics and selected_experts is not None:
                selected = selected_experts[batch_idx, :length, :]
                cache = _initial_inference_cache(
                    selected_experts=selected[0],
                    future_scores=temporal_priority[0],
                    cache_size=cache_moe_config.cache_size,
                    cache_moe_config=None,
                )
                for token_idx in range(1, length):
                    cache = _top_priority_cache(
                        priority_scores=priority_scores[token_idx - 1],
                        cache_size=cache_moe_config.cache_size,
                    )
                    token_selected = selected[token_idx]
                    hard_hit_count = _contains_any(token_selected, cache).to(dtype=torch.float32).sum()
                    total_hard_overlap = total_hard_overlap + hard_hit_count
                    total_hard_hit_rate = total_hard_hit_rate + hard_hit_count / token_selected.numel()
                    counted_hard_steps += 1
                    if not spatio_only and token_idx + 1 < length:
                        cache = _update_inference_cache(
                            cache=cache,
                            selected_experts=token_selected,
                            future_scores=temporal_priority[token_idx],
                            cache_size=cache_moe_config.cache_size,
                            cache_moe_config=None,
                        )

    if counted_steps == 0:
        result = _zero_cache_moe_metrics(device)
        result.update(
            {
                "hybrid_cache_hit_rate": zero,
                "temporal_cache_hit_rate": zero,
                "current_cache_hit_rate": zero,
                "spatio_cache_hit_rate": zero,
                "spatio_entropy": zero,
            }
        )
        return result

    denom = torch.tensor(float(counted_steps), device=device, dtype=torch.float32)
    hard_denom = torch.tensor(float(counted_hard_steps), device=device, dtype=torch.float32)
    hard_cache_hit_rate = total_hard_hit_rate / hard_denom if counted_hard_steps else zero
    per_layer_denoms = [
        torch.tensor(float(max(step_count, 1)), device=device, dtype=torch.float32)
        for step_count in per_layer_steps
    ]
    per_layer_step_tensor = torch.tensor(per_layer_steps, device=device, dtype=torch.float32)
    per_layer_weight_tensor = torch.tensor(layer_swap_weights, device=device, dtype=torch.float32)
    cache_hit_rate = total_hits / denom
    return {
        "swap_loss": total_swap / denom,
        "cache_hit_rate": cache_hit_rate,
        "hybrid_cache_hit_rate": cache_hit_rate,
        "temporal_cache_hit_rate": total_temporal_hits / denom,
        "current_cache_hit_rate": total_current_hits / denom,
        "spatio_cache_hit_rate": total_spatio_hits / denom,
        "cache_retention_rate": total_retention / denom,
        "hard_cache_hit_rate": hard_cache_hit_rate,
        "hard_cache_miss_rate": 1.0 - hard_cache_hit_rate if counted_hard_steps else zero,
        "hard_cache_overlap_count": total_hard_overlap / hard_denom if counted_hard_steps else zero,
        "future_entropy": total_entropy / denom,
        "spatio_entropy": total_spatio_entropy / denom,
        "layer_swap_weight": total_layer_weight / denom,
        "per_layer_swap_loss": torch.stack([
            value / denom_value for value, denom_value in zip(per_layer_swap, per_layer_denoms)
        ]),
        "per_layer_weighted_swap_loss": torch.stack([
            value / denom_value for value, denom_value in zip(per_layer_weighted_swap, per_layer_denoms)
        ]),
        "per_layer_cache_hit_rate": torch.stack([
            value / denom_value for value, denom_value in zip(per_layer_hits, per_layer_denoms)
        ]),
        "per_layer_cache_retention_rate": torch.stack([
            value / denom_value for value, denom_value in zip(per_layer_retention, per_layer_denoms)
        ]),
        "per_layer_future_entropy": torch.stack([
            value / denom_value for value, denom_value in zip(per_layer_entropy, per_layer_denoms)
        ]),
        "per_layer_counted_steps": per_layer_step_tensor,
        "per_layer_swap_weight": per_layer_weight_tensor,
    }


def compute_cache_moe_loss(
    layer_stats: List[Dict[str, torch.Tensor]],
    attention_mask: Optional[torch.Tensor],
    cache_moe_config: CacheMoEConfig,
    compute_hard_metrics: bool = True,
) -> Dict[str, torch.Tensor]:
    device = layer_stats[0]["current_probs"].device if layer_stats else (
        attention_mask.device if attention_mask is not None else torch.device("cpu")
    )
    zero = torch.zeros((), device=device, dtype=torch.float32)
    if not layer_stats:
        return _zero_cache_moe_metrics(device)

    if cache_moe_config.spatio_temporal_enabled or cache_moe_config.spatio_only_enabled:
        return _compute_spatio_temporal_cache_moe_loss(
            layer_stats=layer_stats,
            attention_mask=attention_mask,
            cache_moe_config=cache_moe_config,
            compute_hard_metrics=compute_hard_metrics,
        )

    total_swap = zero
    total_hits = zero
    total_retention = zero
    total_hard_hit_rate = zero
    total_hard_overlap = zero
    total_entropy = zero
    total_layer_weight = zero
    counted_steps = 0
    counted_hard_steps = 0
    layer_count = max(int(layer["layer_idx"]) for layer in layer_stats) + 1
    _ensure_layer_adaptive_state(cache_moe_config, layer_count)
    layer_swap_weights = _scheduled_layer_weights(layer_count, cache_moe_config)
    per_layer_swap = [zero for _ in range(layer_count)]
    per_layer_weighted_swap = [zero for _ in range(layer_count)]
    per_layer_hits = [zero for _ in range(layer_count)]
    per_layer_retention = [zero for _ in range(layer_count)]
    per_layer_entropy = [zero for _ in range(layer_count)]
    per_layer_steps = [0 for _ in range(layer_count)]

    for layer in layer_stats:
        layer_idx = int(layer["layer_idx"])
        layer_weight = layer_swap_weights[layer_idx]
        current_probs = layer["current_probs"]
        if cache_moe_config.swap_loss_future_gate_only:
            current_probs = current_probs.detach()
        future_probs = layer["future_probs"]
        selected_experts = layer.get("selected_experts")
        batch_size, seq_len, _ = current_probs.shape
        lengths = _valid_lengths(attention_mask, batch_size, seq_len)

        for batch_idx, length in enumerate(lengths):
            if length <= 1:
                continue

            curr = current_probs[batch_idx, :length, :]
            fut = future_probs[batch_idx, :length, :]

            priority_scores = future_cache_priority_scores(
                curr[:-1],
                fut[:-1],
                cache_moe_config,
            )
            cache_state = _soft_cache_state(
                priority_scores,
                cache_size=cache_moe_config.cache_size,
                temperature=cache_moe_config.temperature,
            )
            next_current = curr[1:]
            swap_per_step = (next_current * (1.0 - cache_state)).sum(dim=-1)
            hit_per_step = (next_current * cache_state).sum(dim=-1)
            swap_contribution = swap_per_step * layer_weight
            entropy_per_step = (-(fut * (fut.clamp_min(1e-8)).log()).sum(dim=-1))

            total_swap = total_swap + swap_contribution.sum()
            total_hits = total_hits + hit_per_step.sum()
            total_retention = total_retention + cache_state.mean(dim=-1).sum()
            total_entropy = total_entropy + entropy_per_step.sum()
            total_layer_weight = total_layer_weight + torch.tensor(
                float(layer_weight * (length - 1)),
                device=device,
                dtype=torch.float32,
            )
            counted_steps += length - 1
            per_layer_swap[layer_idx] = per_layer_swap[layer_idx] + swap_per_step.sum()
            per_layer_weighted_swap[layer_idx] = per_layer_weighted_swap[layer_idx] + swap_contribution.sum()
            per_layer_hits[layer_idx] = per_layer_hits[layer_idx] + hit_per_step.sum()
            per_layer_retention[layer_idx] = per_layer_retention[layer_idx] + cache_state.mean(dim=-1).sum()
            per_layer_entropy[layer_idx] = per_layer_entropy[layer_idx] + entropy_per_step.sum()
            per_layer_steps[layer_idx] += length - 1

            if compute_hard_metrics and selected_experts is not None:
                selected = selected_experts[batch_idx, :length, :]
                cache = _initial_inference_cache(
                    selected_experts=selected[0],
                    future_scores=fut[0],
                    cache_size=cache_moe_config.cache_size,
                    current_scores=curr[0],
                    cache_moe_config=cache_moe_config,
                )
                for token_idx in range(1, length):
                    token_selected = selected[token_idx]
                    hard_hit_count = _contains_any(token_selected, cache).to(dtype=torch.float32).sum()
                    total_hard_overlap = total_hard_overlap + hard_hit_count
                    total_hard_hit_rate = total_hard_hit_rate + hard_hit_count / token_selected.numel()
                    counted_hard_steps += 1
                    cache = _update_inference_cache(
                        cache=cache,
                        selected_experts=token_selected,
                        future_scores=fut[token_idx],
                        cache_size=cache_moe_config.cache_size,
                        current_scores=curr[token_idx],
                        cache_moe_config=cache_moe_config,
                    )

    if counted_steps == 0:
        return _zero_cache_moe_metrics(device)

    denom = torch.tensor(float(counted_steps), device=device, dtype=torch.float32)
    hard_denom = torch.tensor(float(counted_hard_steps), device=device, dtype=torch.float32)
    hard_cache_hit_rate = total_hard_hit_rate / hard_denom if counted_hard_steps else zero
    per_layer_denoms = [
        torch.tensor(float(max(step_count, 1)), device=device, dtype=torch.float32)
        for step_count in per_layer_steps
    ]
    per_layer_step_tensor = torch.tensor(per_layer_steps, device=device, dtype=torch.float32)
    per_layer_weight_tensor = torch.tensor(layer_swap_weights, device=device, dtype=torch.float32)
    return {
        "swap_loss": total_swap / denom,
        "cache_hit_rate": total_hits / denom,
        "cache_retention_rate": total_retention / denom,
        "hard_cache_hit_rate": hard_cache_hit_rate,
        "hard_cache_miss_rate": 1.0 - hard_cache_hit_rate if counted_hard_steps else zero,
        "hard_cache_overlap_count": total_hard_overlap / hard_denom if counted_hard_steps else zero,
        "future_entropy": total_entropy / denom,
        "layer_swap_weight": total_layer_weight / denom,
        "per_layer_swap_loss": torch.stack([
            value / denom_value for value, denom_value in zip(per_layer_swap, per_layer_denoms)
        ]),
        "per_layer_weighted_swap_loss": torch.stack([
            value / denom_value for value, denom_value in zip(per_layer_weighted_swap, per_layer_denoms)
        ]),
        "per_layer_cache_hit_rate": torch.stack([
            value / denom_value for value, denom_value in zip(per_layer_hits, per_layer_denoms)
        ]),
        "per_layer_cache_retention_rate": torch.stack([
            value / denom_value for value, denom_value in zip(per_layer_retention, per_layer_denoms)
        ]),
        "per_layer_future_entropy": torch.stack([
            value / denom_value for value, denom_value in zip(per_layer_entropy, per_layer_denoms)
        ]),
        "per_layer_counted_steps": per_layer_step_tensor,
        "per_layer_swap_weight": per_layer_weight_tensor,
    }


def _reset_cache_moe_state(self):
    self._cache_moe_layer_stats = []


def _compute_cache_moe_aux(
    self,
    attention_mask: Optional[torch.Tensor] = None,
    compute_hard_metrics: bool = True,
) -> Dict[str, torch.Tensor]:
    return compute_cache_moe_loss(
        self._cache_moe_layer_stats,
        attention_mask,
        self._cache_moe_config,
        compute_hard_metrics=compute_hard_metrics,
    )


def _initialize_spatio_gates_from_target_routers(moe_blocks: List[nn.Module]) -> None:
    if not moe_blocks:
        return

    for source_idx, block in enumerate(moe_blocks):
        spatio_gate = getattr(block, "spatio_gate", None)
        if spatio_gate is None:
            continue
        if not isinstance(spatio_gate, nn.Linear):
            raise TypeError("spatio_gate is expected to be a linear router-style head.")

        target_block = moe_blocks[(source_idx + 1) % len(moe_blocks)]
        target_gate = getattr(target_block, "gate", None)
        target_weight = getattr(target_gate, "weight", None)
        if target_weight is None:
            raise ValueError("Cannot initialize spatio_gate: target MoE router has no weight.")
        if tuple(spatio_gate.weight.shape) != tuple(target_weight.shape):
            raise ValueError(
                "Cannot initialize spatio_gate from target router due to shape mismatch: "
                f"spatio_gate={tuple(spatio_gate.weight.shape)}, "
                f"target_gate={tuple(target_weight.shape)}."
            )

        with torch.no_grad():
            spatio_gate.weight.copy_(target_weight.to(device=spatio_gate.weight.device, dtype=spatio_gate.weight.dtype))
        block._spatio_gate_init_source_layer_idx = int(getattr(target_block, "layer_idx", (source_idx + 1) % len(moe_blocks)))


def install_cache_moe(model: nn.Module, cache_moe_config: CacheMoEConfig) -> nn.Module:
    if not cache_moe_config.enabled:
        return model

    if cache_moe_config.spatio_temporal_enabled and cache_moe_config.spatio_only_enabled:
        raise ValueError(
            "cache_moe.spatio_temporal_enabled and cache_moe.spatio_only_enabled "
            "are mutually exclusive."
        )

    if getattr(model, "_cache_moe_enabled", False):
        return model

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("CacheMoE patching currently expects a decoder model with model.layers")

    model._cache_moe_enabled = True
    model._cache_moe_config = cache_moe_config
    model._cache_moe_layer_stats = []
    model.reset_cache_moe_state = types.MethodType(_reset_cache_moe_state, model)
    model.compute_cache_moe_aux = types.MethodType(_compute_cache_moe_aux, model)

    moe_layer_idx = 0
    moe_blocks: List[nn.Module] = []
    for decoder_layer in model.model.layers:
        mlp = getattr(decoder_layer, "mlp", None)
        if mlp is None:
            continue
        if _is_deepseek_v2_moe_block(mlp):
            wrapped_mlp = CacheAwareDeepseekV2MoEBlock(
                original_block=mlp,
                owner_model=model,
                layer_idx=moe_layer_idx,
                cache_moe_config=cache_moe_config,
            )
            decoder_layer.mlp = wrapped_mlp
            moe_blocks.append(wrapped_mlp)
            moe_layer_idx += 1
        elif _is_qwen3_moe_block(mlp):
            wrapped_mlp = CacheAwareQwen3MoeSparseBlock(
                original_block=mlp,
                owner_model=model,
                layer_idx=moe_layer_idx,
                cache_moe_config=cache_moe_config,
            )
            decoder_layer.mlp = wrapped_mlp
            moe_blocks.append(wrapped_mlp)
            moe_layer_idx += 1
        elif _is_generic_router_moe_block(mlp):
            wrapped_mlp = CacheAwareGenericRouterMoEBlock(
                original_block=mlp,
                owner_model=model,
                layer_idx=moe_layer_idx,
                cache_moe_config=cache_moe_config,
            )
            decoder_layer.mlp = wrapped_mlp
            moe_blocks.append(wrapped_mlp)
            moe_layer_idx += 1

    if moe_layer_idx == 0:
        raise ValueError("No sparse MoE blocks were found to patch with the future cache router.")

    if _spatio_router_enabled(cache_moe_config) and cache_moe_config.copy_current_router_init:
        _initialize_spatio_gates_from_target_routers(moe_blocks)

    return model
