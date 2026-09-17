from __future__ import annotations

import types
import weakref
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from cache_moe import (
    _contains_any,
    _generic_router_logits,
    _generic_router_module,
    _generic_selected_experts_from_logits,
    _infer_generic_top_k,
    _is_generic_router_moe_block,
    _is_qwen3_moe_block,
    _router_weight,
    _valid_lengths,
)


@dataclass
class TemporalOptionMoEConfig:
    enabled: bool = False
    option_size: int = 16
    switch_threshold: float = 0.5
    switch_init_bias: float = -2.0
    controller_hidden_dim: int = 0
    set_embed_dim: int = 0
    copy_router_init: bool = True
    selection_loss_weight: float = 0.10
    termination_loss_weight: float = 0.05
    deliberation_cost: float = 0.02
    entropy_weight: float = 0.0
    force_switch_on_teacher_miss: bool = False
    train_base_model: bool = True
    train_router: bool = False
    train_lora: bool = False
    fast_training: bool = False


def _make_linear(
    in_features: int,
    out_features: int,
    device: torch.device,
    dtype: torch.dtype,
    bias: bool = True,
) -> nn.Linear:
    try:
        return nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    except TypeError:
        return nn.Linear(in_features, out_features, bias=bias).to(device=device, dtype=dtype)


class _FallbackRMSNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(dtype=torch.float32).pow(2).mean(dim=-1, keepdim=True)
        output = hidden_states * torch.rsqrt(variance + self.eps).to(dtype=hidden_states.dtype)
        return output * self.weight.to(device=hidden_states.device, dtype=hidden_states.dtype)


def _make_rms_norm(normalized_shape: int, device: torch.device, dtype: torch.dtype) -> nn.Module:
    rms_norm_cls = getattr(nn, "RMSNorm", None)
    if rms_norm_cls is not None:
        return rms_norm_cls(normalized_shape).to(device=device, dtype=dtype)
    return _FallbackRMSNorm(normalized_shape).to(device=device, dtype=dtype)


def _multi_hot(indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    result = torch.zeros(
        indices.size(0),
        num_experts,
        device=indices.device,
        dtype=torch.float32,
    )
    return result.scatter_(1, indices.to(dtype=torch.long), 1.0)


class TemporalOptionController(nn.Module):
    """Activation-based controller for temporally persistent expert options.

    This mirrors the public TEM controller structure at the level needed for this
    project: a router-initialized selection head and a termination head that sees
    both the current hidden state and a DeepSets embedding of the active expert
    set. The training objective in this repository uses supervised router
    coverage targets instead of the paper's full rollout/value update.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        config: TemporalOptionMoEConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_experts = int(num_experts)
        self.option_size = int(config.option_size)
        set_embed_dim = int(config.set_embed_dim or hidden_dim)
        controller_hidden_dim = int(config.controller_hidden_dim or hidden_dim)
        self.set_embed_dim = set_embed_dim

        self.selection_head = _make_linear(hidden_dim, num_experts, device=device, dtype=dtype, bias=False)
        try:
            self.expert_embedding = nn.Embedding(num_experts, hidden_dim, device=device, dtype=dtype)
        except TypeError:
            self.expert_embedding = nn.Embedding(num_experts, hidden_dim).to(device=device, dtype=dtype)
        self.hidden_norm = _make_rms_norm(hidden_dim, device=device, dtype=dtype)
        self.set_norm = _make_rms_norm(set_embed_dim, device=device, dtype=dtype)
        self.deepsets_phi = nn.Sequential(
            _make_linear(hidden_dim, set_embed_dim, device=device, dtype=dtype, bias=True),
            nn.ReLU(),
            _make_linear(set_embed_dim, set_embed_dim, device=device, dtype=dtype, bias=True),
        )
        self.termination_head = nn.Sequential(
            _make_linear(hidden_dim + set_embed_dim, controller_hidden_dim, device=device, dtype=dtype, bias=True),
            nn.ReLU(),
            _make_linear(controller_hidden_dim, 1, device=device, dtype=dtype, bias=True),
        )
        with torch.no_grad():
            last = self.termination_head[-1]
            if isinstance(last, nn.Linear):
                nn.init.xavier_uniform_(last.weight)
                last.weight.mul_(0.01)
                last.bias.fill_(float(config.switch_init_bias))

    def initialize_from_router(self, router_weight: Optional[torch.Tensor]) -> None:
        if router_weight is None:
            return
        if tuple(self.selection_head.weight.shape) != tuple(router_weight.shape):
            raise ValueError(
                "Cannot initialize temporal option selection head from router: "
                f"selection={tuple(self.selection_head.weight.shape)}, "
                f"router={tuple(router_weight.shape)}."
            )
        with torch.no_grad():
            copied = router_weight.to(device=self.selection_head.weight.device, dtype=self.selection_head.weight.dtype)
            self.selection_head.weight.copy_(copied)
            if tuple(self.expert_embedding.weight.shape) == tuple(router_weight.shape):
                self.expert_embedding.weight.copy_(
                    router_weight.to(device=self.expert_embedding.weight.device, dtype=self.expert_embedding.weight.dtype)
                )

    def all_expert_set_embeddings(self, dtype: torch.dtype) -> torch.Tensor:
        # Under FSDP, directly reading embedding.weight can expose a local flat
        # shard. Calling the module forward lets FSDP materialize the full table.
        embedding_param = next(self.expert_embedding.parameters())
        expert_ids = torch.arange(self.num_experts, device=embedding_param.device, dtype=torch.long)
        expert_embeddings = self.expert_embedding(expert_ids).to(dtype=dtype)
        return self.deepsets_phi(expert_embeddings)

    def set_embedding(
        self,
        option_mask: torch.Tensor,
        dtype: torch.dtype,
        all_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = option_mask.to(dtype=dtype)
        if all_embeddings is None:
            all_embeddings = self.all_expert_set_embeddings(dtype=dtype)
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return mask.matmul(all_embeddings) / denom


class TemporalOptionQwen3MoeSparseBlock(nn.Module):
    def __init__(
        self,
        original_block: nn.Module,
        owner_model: nn.Module,
        layer_idx: int,
        config: TemporalOptionMoEConfig,
    ) -> None:
        super().__init__()
        self.experts = original_block.experts
        self.gate = original_block.gate
        self.layer_idx = int(layer_idx)
        self.config = config
        self._owner_model_ref = weakref.ref(owner_model)
        self.hidden_dim = int(getattr(self.gate, "hidden_dim", getattr(self.gate.weight, "shape", [0, 0])[1]))
        self.num_experts = int(getattr(self.gate, "num_experts", getattr(self.gate.weight, "shape", [0])[0]))
        gate_weight = getattr(self.gate, "weight", None)
        if gate_weight is not None:
            dtype = gate_weight.dtype
            device = gate_weight.device
        else:
            reference_param = next(original_block.parameters())
            dtype = reference_param.dtype
            device = reference_param.device
        self.controller = TemporalOptionController(
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            config=config,
            device=device,
            dtype=dtype,
        )
        if config.copy_router_init:
            self.controller.initialize_from_router(gate_weight)

    def _top_k(self, teacher_selected: torch.Tensor) -> int:
        candidate = getattr(self.gate, "top_k", None)
        if candidate is None:
            candidate = getattr(self.gate, "num_experts_per_tok", None)
        if candidate is None:
            candidate = int(teacher_selected.size(-1))
        return max(1, int(candidate))

    def _masked_route(
        self,
        router_logits: torch.Tensor,
        option_masks: torch.Tensor,
        top_k: int,
        target_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_logits = router_logits.reshape(-1, self.num_experts)
        flat_masks = option_masks.reshape(-1, self.num_experts)
        masked_logits = flat_logits.masked_fill(~flat_masks, torch.finfo(flat_logits.dtype).min)
        probs = F.softmax(masked_logits.to(dtype=torch.float32), dim=-1)
        routing_weights, selected_experts = torch.topk(probs, k=top_k, dim=-1)
        norm_topk = bool(getattr(self.gate, "norm_topk_prob", True))
        if norm_topk:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = float(getattr(self.gate, "routed_scaling_factor", 1.0))
        routing_weights = (routing_weights * scale).to(dtype=target_dtype)
        return selected_experts.to(dtype=torch.long), routing_weights

    def _build_options(
        self,
        hidden_states: torch.Tensor,
        selection_logits: torch.Tensor,
        teacher_selected: torch.Tensor,
        persistent_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = hidden_states.shape
        device = hidden_states.device
        option_size = max(1, min(int(self.config.option_size), self.num_experts))
        if option_size < int(teacher_selected.size(-1)):
            raise ValueError(
                "temporal_option_moe.option_size must be >= routed experts per token. "
                f"option_size={option_size}, routed={int(teacher_selected.size(-1))}."
            )

        if persistent_mask is not None and persistent_mask.shape == (batch_size, self.num_experts):
            active_mask: Optional[torch.Tensor] = persistent_mask.to(device=device, dtype=torch.bool)
        else:
            active_mask = None

        option_masks: List[torch.Tensor] = []
        switch_probs: List[torch.Tensor] = []
        switch_flags: List[torch.Tensor] = []
        switch_targets: List[torch.Tensor] = []
        all_set_embeddings = self.controller.all_expert_set_embeddings(dtype=hidden_states.dtype)

        for token_idx in range(sequence_length):
            candidate_indices = torch.topk(selection_logits[:, token_idx, :], k=option_size, dim=-1).indices
            candidate_mask = torch.zeros(batch_size, self.num_experts, device=device, dtype=torch.bool)
            candidate_mask.scatter_(1, candidate_indices.to(dtype=torch.long), True)

            if active_mask is None:
                prev_mask = torch.zeros(batch_size, self.num_experts, device=device, dtype=torch.bool)
                forced_first = torch.ones(batch_size, device=device, dtype=torch.bool)
            else:
                prev_mask = active_mask
                forced_first = torch.zeros(batch_size, device=device, dtype=torch.bool)

            set_embed = self.controller.set_embedding(
                prev_mask,
                dtype=hidden_states.dtype,
                all_embeddings=all_set_embeddings,
            )
            termination_input = torch.cat(
                [
                    self.controller.hidden_norm(hidden_states[:, token_idx, :]),
                    self.controller.set_norm(set_embed),
                ],
                dim=-1,
            )
            switch_logit = self.controller.termination_head(termination_input).squeeze(-1)
            switch_prob = torch.sigmoid(switch_logit.to(dtype=torch.float32))

            teacher_now = teacher_selected[:, token_idx, :].to(device=device, dtype=torch.long)
            prev_covered = prev_mask.gather(1, teacher_now)
            target_switch = (~prev_covered.all(dim=-1)).to(dtype=torch.float32)
            should_switch = forced_first | switch_prob.ge(float(self.config.switch_threshold))
            if self.config.force_switch_on_teacher_miss:
                should_switch = should_switch | target_switch.bool()
            switch_mask = should_switch.unsqueeze(-1)
            active_mask = (switch_mask & candidate_mask) | (~switch_mask & prev_mask)

            option_masks.append(active_mask)
            switch_probs.append(switch_prob)
            switch_flags.append(should_switch.to(dtype=torch.float32))
            switch_targets.append(target_switch)

        return (
            torch.stack(option_masks, dim=1),
            torch.stack(switch_probs, dim=1),
            torch.stack(switch_flags, dim=1),
            torch.stack(switch_targets, dim=1),
        )

    def _build_training_options_fast(
        self,
        hidden_states: torch.Tensor,
        selection_logits: torch.Tensor,
        teacher_selected: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = hidden_states.shape
        device = hidden_states.device
        option_size = max(1, min(int(self.config.option_size), self.num_experts))
        if option_size < int(teacher_selected.size(-1)):
            raise ValueError(
                "temporal_option_moe.option_size must be >= routed experts per token. "
                f"option_size={option_size}, routed={int(teacher_selected.size(-1))}."
            )

        candidate_indices = torch.topk(selection_logits, k=option_size, dim=-1).indices
        option_masks = torch.zeros(
            batch_size,
            sequence_length,
            self.num_experts,
            device=device,
            dtype=torch.bool,
        )
        option_masks.scatter_(2, candidate_indices.to(dtype=torch.long), True)

        prev_masks = torch.zeros_like(option_masks)
        if sequence_length > 1:
            prev_masks[:, 1:, :] = option_masks[:, :-1, :]

        all_set_embeddings = self.controller.all_expert_set_embeddings(dtype=hidden_states.dtype)
        flat_prev_masks = prev_masks.reshape(batch_size * sequence_length, self.num_experts)
        set_embed = self.controller.set_embedding(
            flat_prev_masks,
            dtype=hidden_states.dtype,
            all_embeddings=all_set_embeddings,
        ).reshape(batch_size, sequence_length, self.controller.set_embed_dim)
        hidden_norm = self.controller.hidden_norm(
            hidden_states.reshape(batch_size * sequence_length, self.hidden_dim)
        ).reshape(batch_size, sequence_length, self.hidden_dim)
        set_norm = self.controller.set_norm(
            set_embed.reshape(batch_size * sequence_length, self.controller.set_embed_dim)
        ).reshape(batch_size, sequence_length, self.controller.set_embed_dim)
        termination_input = torch.cat([hidden_norm, set_norm], dim=-1).reshape(
            batch_size * sequence_length,
            self.hidden_dim + self.controller.set_embed_dim,
        )
        switch_logits = self.controller.termination_head(termination_input).reshape(batch_size, sequence_length)
        switch_probs = torch.sigmoid(switch_logits.to(dtype=torch.float32))

        teacher = teacher_selected.to(device=device, dtype=torch.long)
        prev_covered = prev_masks.gather(2, teacher)
        switch_targets = (~prev_covered.all(dim=-1)).to(dtype=torch.float32)
        switch_flags = switch_probs.ge(float(self.config.switch_threshold)).to(dtype=torch.float32)
        if sequence_length > 0:
            switch_targets[:, 0] = 1.0
            switch_flags[:, 0] = 1.0
        return option_masks, switch_probs, switch_flags, switch_targets

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden_dim)

        router_logits, _, teacher_selected = self.gate(hidden_flat)
        teacher_selected_seq = teacher_selected.reshape(batch_size, sequence_length, -1)
        selection_logits = self.controller.selection_head(hidden_flat).reshape(
            batch_size,
            sequence_length,
            self.num_experts,
        )

        owner_model = self._owner_model_ref()
        persistent_mask = None
        if owner_model is not None and not self.training:
            runtime_masks = getattr(owner_model, "_temporal_option_moe_runtime_masks", None)
            if isinstance(runtime_masks, dict):
                persistent_mask = runtime_masks.get(self.layer_idx)

        if self.training and self.config.fast_training:
            option_masks, switch_probs, switch_flags, switch_targets = self._build_training_options_fast(
                hidden_states=hidden_states,
                selection_logits=selection_logits,
                teacher_selected=teacher_selected_seq,
            )
        else:
            option_masks, switch_probs, switch_flags, switch_targets = self._build_options(
                hidden_states=hidden_states,
                selection_logits=selection_logits,
                teacher_selected=teacher_selected_seq,
                persistent_mask=persistent_mask,
            )
        if owner_model is not None and not self.training:
            runtime_masks = getattr(owner_model, "_temporal_option_moe_runtime_masks", None)
            if not isinstance(runtime_masks, dict):
                runtime_masks = {}
                owner_model._temporal_option_moe_runtime_masks = runtime_masks
            runtime_masks[self.layer_idx] = option_masks[:, -1, :].detach()

        top_k = self._top_k(teacher_selected)
        selected_experts, routing_weights = self._masked_route(
            router_logits=router_logits.reshape(batch_size, sequence_length, self.num_experts),
            option_masks=option_masks,
            top_k=top_k,
            target_dtype=hidden_states.dtype,
        )
        final_hidden_states = self.experts(hidden_flat, selected_experts, routing_weights)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        if owner_model is not None and getattr(owner_model, "_temporal_option_moe_enabled", False):
            current_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32).reshape(
                batch_size,
                sequence_length,
                self.num_experts,
            )
            owner_model._temporal_option_moe_layer_stats.append(
                {
                    "layer_idx": self.layer_idx,
                    "current_probs": current_probs,
                    "router_logits": router_logits.reshape(batch_size, sequence_length, self.num_experts),
                    "selection_logits": selection_logits,
                    "option_masks": option_masks,
                    "switch_probs": switch_probs,
                    "switch_flags": switch_flags,
                    "switch_targets": switch_targets,
                    "teacher_selected_experts": teacher_selected_seq,
                    "selected_experts": selected_experts.reshape(batch_size, sequence_length, -1),
                }
            )

        return final_hidden_states


class TemporalOptionGenericRouterMoEBlock(TemporalOptionQwen3MoeSparseBlock):
    def __init__(
        self,
        original_block: nn.Module,
        owner_model: nn.Module,
        layer_idx: int,
        config: TemporalOptionMoEConfig,
    ) -> None:
        nn.Module.__init__(self)
        self.original_block = original_block
        self.experts = getattr(original_block, "experts", None)
        self.gate = _generic_router_module(original_block)
        if self.gate is None:
            raise ValueError("Generic Temporal Option MoE block has no supported router/gate module.")
        gate_weight = _router_weight(self.gate)
        if gate_weight is None:
            raise ValueError("Generic Temporal Option MoE router has no usable weight for shape inference.")
        self.layer_idx = int(layer_idx)
        self.config = config
        self._owner_model_ref = weakref.ref(owner_model)
        self.hidden_dim = int(gate_weight.shape[1])
        self.num_experts = int(gate_weight.shape[0])
        self.top_k = _infer_generic_top_k(original_block, self.gate, owner_model=owner_model)
        module_name = type(original_block).__module__.lower()
        class_name = type(original_block).__name__.lower()
        self._returns_router_tuple = "gpt_oss" in module_name or "gptoss" in class_name
        self.controller = TemporalOptionController(
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            config=config,
            device=gate_weight.device,
            dtype=gate_weight.dtype,
        )
        if config.copy_router_init:
            self.controller.initialize_from_router(gate_weight)

    @property
    def router(self):
        return self.gate

    def _top_k(self, teacher_selected: torch.Tensor) -> int:
        return max(1, int(self.top_k))

    def _call_experts(
        self,
        hidden_flat: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.experts is None or not callable(self.experts):
            raise TypeError(
                "Generic Temporal Option MoE requires a callable experts module to apply masked routing."
            )
        try:
            return self.experts(hidden_flat, selected_experts, routing_weights)
        except TypeError as selected_first_error:
            try:
                return self.experts(hidden_flat, routing_weights, selected_experts)
            except TypeError:
                raise selected_first_error

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden_dim)

        router_logits = _generic_router_logits(self.gate, hidden_states, num_experts=self.num_experts)
        teacher_selected, _ = _generic_selected_experts_from_logits(router_logits, top_k=self.top_k)
        teacher_selected_seq = teacher_selected.reshape(batch_size, sequence_length, -1)
        selection_logits = self.controller.selection_head(hidden_flat).reshape(
            batch_size,
            sequence_length,
            self.num_experts,
        )

        owner_model = self._owner_model_ref()
        persistent_mask = None
        if owner_model is not None and not self.training:
            runtime_masks = getattr(owner_model, "_temporal_option_moe_runtime_masks", None)
            if isinstance(runtime_masks, dict):
                persistent_mask = runtime_masks.get(self.layer_idx)

        if self.training and self.config.fast_training:
            option_masks, switch_probs, switch_flags, switch_targets = self._build_training_options_fast(
                hidden_states=hidden_states,
                selection_logits=selection_logits,
                teacher_selected=teacher_selected_seq,
            )
        else:
            option_masks, switch_probs, switch_flags, switch_targets = self._build_options(
                hidden_states=hidden_states,
                selection_logits=selection_logits,
                teacher_selected=teacher_selected_seq,
                persistent_mask=persistent_mask,
            )
        if owner_model is not None and not self.training:
            runtime_masks = getattr(owner_model, "_temporal_option_moe_runtime_masks", None)
            if not isinstance(runtime_masks, dict):
                runtime_masks = {}
                owner_model._temporal_option_moe_runtime_masks = runtime_masks
            runtime_masks[self.layer_idx] = option_masks[:, -1, :].detach()

        selected_experts, routing_weights = self._masked_route(
            router_logits=router_logits.reshape(batch_size, sequence_length, self.num_experts),
            option_masks=option_masks,
            top_k=self.top_k,
            target_dtype=hidden_states.dtype,
        )
        final_hidden_states = self._call_experts(hidden_flat, selected_experts, routing_weights)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        if owner_model is not None and getattr(owner_model, "_temporal_option_moe_enabled", False):
            current_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32).reshape(
                batch_size,
                sequence_length,
                self.num_experts,
            )
            owner_model._temporal_option_moe_layer_stats.append(
                {
                    "layer_idx": self.layer_idx,
                    "current_probs": current_probs,
                    "router_logits": router_logits.reshape(batch_size, sequence_length, self.num_experts),
                    "selection_logits": selection_logits,
                    "option_masks": option_masks,
                    "switch_probs": switch_probs,
                    "switch_flags": switch_flags,
                    "switch_targets": switch_targets,
                    "teacher_selected_experts": teacher_selected_seq,
                    "selected_experts": selected_experts.reshape(batch_size, sequence_length, -1),
                }
            )

        if self._returns_router_tuple:
            return final_hidden_states, router_logits.reshape(batch_size, sequence_length, self.num_experts)
        return final_hidden_states


def _zero(device: torch.device) -> torch.Tensor:
    return torch.zeros((), device=device, dtype=torch.float32)


def compute_temporal_option_moe_loss(
    layer_stats: List[Dict[str, torch.Tensor]],
    attention_mask: Optional[torch.Tensor],
    config: TemporalOptionMoEConfig,
) -> Dict[str, torch.Tensor]:
    device = layer_stats[0]["selection_logits"].device if layer_stats else (
        attention_mask.device if attention_mask is not None else torch.device("cpu")
    )
    zero = _zero(device)
    if not layer_stats:
        return {
            "temporal_option_loss": zero,
            "option_selection_loss": zero,
            "option_termination_loss": zero,
            "option_deliberation_loss": zero,
            "option_entropy": zero,
            "option_switch_rate": zero,
            "option_target_switch_rate": zero,
            "option_teacher_coverage": zero,
            "option_router_overlap": zero,
            "option_size": zero,
        }

    total_selection = zero
    total_termination = zero
    total_deliberation = zero
    total_entropy = zero
    total_switch = zero
    total_target_switch = zero
    total_coverage = zero
    total_router_overlap = zero
    total_option_size = zero
    counted = 0

    for layer in layer_stats:
        selection_logits = layer["selection_logits"]
        option_masks = layer["option_masks"]
        switch_probs = layer["switch_probs"]
        switch_flags = layer["switch_flags"]
        switch_targets = layer["switch_targets"]
        teacher_selected = layer["teacher_selected_experts"]
        selected = layer["selected_experts"]
        batch_size, seq_len, num_experts = selection_logits.shape
        lengths = _valid_lengths(attention_mask, batch_size, seq_len)

        for batch_idx, length in enumerate(lengths):
            if length <= 0:
                continue
            logits = selection_logits[batch_idx, :length, :]
            teacher = teacher_selected[batch_idx, :length, :].to(dtype=torch.long)
            masks = option_masks[batch_idx, :length, :]
            targets = _multi_hot(teacher.reshape(-1, teacher.size(-1)), num_experts).reshape(length, num_experts)
            selection_loss = F.binary_cross_entropy_with_logits(logits.to(dtype=torch.float32), targets)
            termination_loss = F.binary_cross_entropy(
                switch_probs[batch_idx, :length].to(dtype=torch.float32).clamp(min=1e-6, max=1.0 - 1e-6),
                switch_targets[batch_idx, :length].to(dtype=torch.float32),
            )
            selection_probs = F.softmax(logits.to(dtype=torch.float32), dim=-1)
            entropy = (-(selection_probs * selection_probs.clamp_min(1e-8).log()).sum(dim=-1)).mean()
            covered = masks.gather(1, teacher).to(dtype=torch.float32).mean(dim=-1)
            routed = selected[batch_idx, :length, :].to(dtype=torch.long)
            overlap = routed.unsqueeze(-1).eq(teacher.unsqueeze(1)).any(dim=-1).to(dtype=torch.float32).mean(dim=-1)

            total_selection = total_selection + selection_loss
            total_termination = total_termination + termination_loss
            total_deliberation = total_deliberation + switch_probs[batch_idx, :length].to(dtype=torch.float32).mean()
            total_entropy = total_entropy + entropy
            total_switch = total_switch + switch_flags[batch_idx, :length].to(dtype=torch.float32).mean()
            total_target_switch = total_target_switch + switch_targets[batch_idx, :length].to(dtype=torch.float32).mean()
            total_coverage = total_coverage + covered.mean()
            total_router_overlap = total_router_overlap + overlap.mean()
            total_option_size = total_option_size + masks.to(dtype=torch.float32).sum(dim=-1).mean()
            counted += 1

    if counted <= 0:
        return {
            "temporal_option_loss": zero,
            "option_selection_loss": zero,
            "option_termination_loss": zero,
            "option_deliberation_loss": zero,
            "option_entropy": zero,
            "option_switch_rate": zero,
            "option_target_switch_rate": zero,
            "option_teacher_coverage": zero,
            "option_router_overlap": zero,
            "option_size": zero,
        }

    denom = torch.tensor(float(counted), device=device, dtype=torch.float32)
    selection = total_selection / denom
    termination = total_termination / denom
    deliberation = total_deliberation / denom
    entropy = total_entropy / denom
    aux = (
        float(config.selection_loss_weight) * selection
        + float(config.termination_loss_weight) * termination
        + float(config.deliberation_cost) * deliberation
        - float(config.entropy_weight) * entropy
    )
    return {
        "temporal_option_loss": aux,
        "option_selection_loss": selection,
        "option_termination_loss": termination,
        "option_deliberation_loss": deliberation,
        "option_entropy": entropy,
        "option_switch_rate": total_switch / denom,
        "option_target_switch_rate": total_target_switch / denom,
        "option_teacher_coverage": total_coverage / denom,
        "option_router_overlap": total_router_overlap / denom,
        "option_size": total_option_size / denom,
    }


def _reset_temporal_option_moe_state(self):
    self._temporal_option_moe_layer_stats = []
    self._temporal_option_moe_runtime_masks = {}


def _compute_temporal_option_moe_aux(
    self,
    attention_mask: Optional[torch.Tensor] = None,
    compute_hard_metrics: bool = True,
) -> Dict[str, torch.Tensor]:
    return compute_temporal_option_moe_loss(
        self._temporal_option_moe_layer_stats,
        attention_mask,
        self._temporal_option_moe_config,
    )


def _decoder_layers(model: nn.Module) -> Optional[nn.ModuleList]:
    candidates: List[nn.Module] = []
    queue: List[nn.Module] = []
    current: Optional[nn.Module] = model
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        candidates.append(current)
        queue.append(current)
        current = getattr(current, "module", None) or getattr(current, "_orig_mod", None)

    while queue:
        candidate = queue.pop(0)
        for attr in ("base_model", "model"):
            nested = getattr(candidate, attr, None)
            if isinstance(nested, nn.Module) and id(nested) not in visited:
                visited.add(id(nested))
                candidates.append(nested)
                queue.append(nested)

    for candidate in candidates:
        direct_model = getattr(candidate, "model", None)
        if direct_model is not None and hasattr(direct_model, "layers"):
            return direct_model.layers
        if hasattr(candidate, "layers"):
            return candidate.layers
    return None


def install_temporal_option_moe(model: nn.Module, config: TemporalOptionMoEConfig) -> nn.Module:
    if not config.enabled:
        return model
    if getattr(model, "_temporal_option_moe_enabled", False):
        return model
    decoder_layers = _decoder_layers(model)
    if decoder_layers is None:
        raise ValueError(
            "TemporalOptionMoE patching expects a decoder model with model.layers, "
            "or a PEFT-wrapped model exposing base_model.model.model.layers."
        )

    model._temporal_option_moe_enabled = True
    model._temporal_option_moe_config = config
    model._temporal_option_moe_layer_stats = []
    model._temporal_option_moe_runtime_masks = {}
    model.reset_temporal_option_moe_state = types.MethodType(_reset_temporal_option_moe_state, model)
    model.compute_temporal_option_moe_aux = types.MethodType(_compute_temporal_option_moe_aux, model)

    moe_layer_idx = 0
    for decoder_layer in decoder_layers:
        mlp = getattr(decoder_layer, "mlp", None)
        if mlp is None:
            continue
        if _is_qwen3_moe_block(mlp):
            decoder_layer.mlp = TemporalOptionQwen3MoeSparseBlock(
                original_block=mlp,
                owner_model=model,
                layer_idx=moe_layer_idx,
                config=config,
            )
            moe_layer_idx += 1
        elif _is_generic_router_moe_block(mlp):
            decoder_layer.mlp = TemporalOptionGenericRouterMoEBlock(
                original_block=mlp,
                owner_model=model,
                layer_idx=moe_layer_idx,
                config=config,
            )
            moe_layer_idx += 1

    if moe_layer_idx == 0:
        raise ValueError("No sparse MoE blocks were found to patch with TemporalOptionMoE.")
    return model


def _is_router_parameter_name(name: str) -> bool:
    return (
        ".mlp.gate." in name
        or ".mlp.original_block.gate." in name
        or ".original_block.gate." in name
        or ".mlp.router." in name
        or ".mlp.original_block.router." in name
        or ".original_block.router." in name
    )


def _is_lora_parameter_name(name: str) -> bool:
    return "lora_" in name or ".lora_embedding_" in name


def set_temporal_option_trainable_parameters(
    model: nn.Module,
    train_router: bool = False,
    train_lora: bool = False,
) -> Dict[str, int]:
    trainable_params = 0
    frozen_params = 0
    trainable_tensors = 0
    frozen_tensors = 0
    controller_params = 0
    router_params = 0
    lora_params = 0
    for name, parameter in model.named_parameters():
        is_controller = ".controller." in name
        is_router = bool(train_router and _is_router_parameter_name(name))
        is_lora = bool(train_lora and _is_lora_parameter_name(name))
        should_train = is_controller or is_router or is_lora
        parameter.requires_grad = should_train
        if should_train:
            trainable_params += int(parameter.numel())
            trainable_tensors += 1
            if is_controller:
                controller_params += int(parameter.numel())
            elif is_router:
                router_params += int(parameter.numel())
            elif is_lora:
                lora_params += int(parameter.numel())
        else:
            frozen_params += int(parameter.numel())
            frozen_tensors += 1
    if trainable_tensors == 0:
        raise RuntimeError(
            "temporal_option_moe.train_base_model=false, but no temporal-option trainable parameters were found."
        )
    return {
        "temporal_option_trainable_only_trainable_tensor_count": trainable_tensors,
        "temporal_option_trainable_only_trainable_parameter_count": trainable_params,
        "temporal_option_trainable_only_frozen_tensor_count": frozen_tensors,
        "temporal_option_trainable_only_frozen_parameter_count": frozen_params,
        "temporal_option_trainable_controller_parameter_count": controller_params,
        "temporal_option_trainable_router_parameter_count": router_params,
        "temporal_option_trainable_lora_parameter_count": lora_params,
    }


def set_temporal_option_trainable_only(model: nn.Module) -> Dict[str, int]:
    return set_temporal_option_trainable_parameters(model, train_router=False, train_lora=False)


def temporal_option_parameter_summary(model: nn.Module) -> Dict[str, int | List[str]]:
    names: List[str] = []
    params = 0
    trainable_params = 0
    for name, parameter in model.named_parameters():
        if ".controller." in name:
            names.append(name)
            params += int(parameter.numel())
            if parameter.requires_grad:
                trainable_params += int(parameter.numel())
    return {
        "temporal_option_tensor_count": len(names),
        "temporal_option_parameter_count": params,
        "temporal_option_trainable_parameter_count": trainable_params,
        "first_temporal_option_tensors": names[:8],
    }
