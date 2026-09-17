from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from cache_moe import _contains_any
from finemoe_store import (
    finemoe_dynamic_threshold,
    finemoe_predict_expert_scores,
    finemoe_semantic_candidates,
)
from promoe_predictor import (
    build_promoe_layer_mapping,
    load_promoe_predictor_runtime,
    promoe_predict,
)


@dataclass
class CachePolicyConfig:
    policy: str = "auto"
    recency_weight: float = 1.0
    frequency_weight: float = 1.0
    recency_decay: float = 128.0
    lrfu_lambda: float = 0.5
    prefetch_budget: int = -1
    prefetch_lookahead: int = 3
    future_cache_priority_mode: str = "current_plus_future"
    promoe_predictor: Optional[Dict[str, Any]] = None
    promoe_layer_predict_interval: int = 1
    promoe_use_last_output: bool = True
    spatio_temporal_refine_mode: str = "topk"
    spatio_temporal_refine_budget: int = -1
    finemoe_store: Optional[Dict[str, Any]] = None
    finemoe_top_k: int = 8
    finemoe_candidate_pool: int = 64
    finemoe_semantic_weight: float = 1.0
    finemoe_trajectory_weight: float = 1.0
    finemoe_prefetch_threshold: float = -1.0
    finemoe_min_prefetch_threshold: float = 0.05
    finemoe_max_prefetch_threshold: float = 0.35
    finemoe_eviction_probability_weight: float = 1.0


def _touch_lru(cache: torch.Tensor, expert: torch.Tensor, cache_size: int) -> torch.Tensor:
    if cache.numel() > 0:
        cache = cache[cache.ne(expert)]
    cache = torch.cat([cache, expert.detach().reshape(1)])
    if cache.numel() > cache_size:
        cache = cache[-cache_size:]
    return cache


def _evict_by_score(
    cache: torch.Tensor,
    score: torch.Tensor,
    cache_size: int,
    protected: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if cache.numel() <= cache_size:
        return cache
    keep_count = max(0, min(cache_size, cache.numel()))
    if keep_count <= 0:
        return cache[:0]

    if protected is not None and protected.numel() > 0:
        protected = torch.unique(protected.detach().to(device=cache.device, dtype=cache.dtype))
        protected = protected[_contains_any(protected, cache)]
        if protected.numel() >= keep_count:
            protected_scores = score[protected.to(dtype=torch.long)]
            keep_indices = torch.topk(protected_scores, k=keep_count, largest=True).indices
            return protected[keep_indices]

        unprotected = cache[~_contains_any(cache, protected)]
        fill_count = keep_count - int(protected.numel())
        if fill_count <= 0 or unprotected.numel() <= 0:
            return protected[:keep_count]
        unprotected_scores = score[unprotected.to(dtype=torch.long)]
        keep_indices = torch.topk(unprotected_scores, k=min(fill_count, unprotected.numel()), largest=True).indices
        return torch.cat([protected, unprotected[keep_indices]], dim=0)

    cache_scores = score[cache.to(dtype=torch.long)]
    keep_indices = torch.topk(cache_scores, k=keep_count, largest=True).indices
    return cache[keep_indices]


def _score_lfu(freq: torch.Tensor) -> torch.Tensor:
    return freq


def _score_lrfu(
    crf: torch.Tensor,
    last_update: torch.Tensor,
    step: int,
    lrfu_lambda: float,
) -> torch.Tensor:
    # Original LRFU CRF: each historical access contributes (1/2) ** (lambda * age).
    age = torch.clamp(torch.tensor(float(step), device=crf.device, dtype=torch.float32) - last_update, min=0.0)
    lambda_value = max(float(lrfu_lambda), 0.0)
    score = crf * torch.pow(torch.tensor(0.5, device=crf.device, dtype=torch.float32), lambda_value * age)
    return score.masked_fill(last_update.lt(0), 0.0)


def _least_stale_next_layer_distance(layer_idx: int, current_layer_pos: int, layer_count: int) -> int:
    if layer_count <= 0:
        return 0
    if layer_idx > current_layer_pos:
        return layer_idx - current_layer_pos
    return layer_count - current_layer_pos + layer_idx


def _least_stale_evict_priority(
    key: Tuple[int, int],
    current: set[Tuple[int, int]],
    entry_order: Dict[Tuple[int, int], int],
    layer_positions: Dict[int, int],
    current_layer_pos: int,
    layer_count: int,
) -> tuple[int, int, int]:
    return (
        1 if key not in current else 0,
        _least_stale_next_layer_distance(layer_positions[key[0]], current_layer_pos, layer_count),
        -entry_order.get(key, -1),
    )


def _evict_least_stale(
    cache: set[Tuple[int, int]],
    current: set[Tuple[int, int]],
    entry_order: Dict[Tuple[int, int], int],
    protected: set[Tuple[int, int]],
    layer_positions: Dict[int, int],
    current_layer_pos: int,
    layer_count: int,
) -> None:
    candidates = [key for key in cache if key not in protected]
    if not candidates:
        candidates = list(cache)

    victim = max(
        candidates,
        key=lambda key: _least_stale_evict_priority(
            key=key,
            current=current,
            entry_order=entry_order,
            layer_positions=layer_positions,
            current_layer_pos=current_layer_pos,
            layer_count=layer_count,
        ),
    )
    cache.remove(victim)
    current.discard(victim)
    entry_order.pop(victim, None)


def _build_selected_sequences(grouped: Dict[int, List[torch.Tensor]]) -> Dict[int, torch.Tensor]:
    layer_sequences: Dict[int, torch.Tensor] = {}
    for layer_idx, selected_chunks in grouped.items():
        if not selected_chunks:
            continue
        selected = torch.cat(selected_chunks, dim=1)[0]
        if selected.size(0) > 0:
            layer_sequences[layer_idx] = selected
    return layer_sequences


def _build_score_sequences(grouped: Dict[int, List[torch.Tensor]]) -> Dict[int, torch.Tensor]:
    layer_sequences: Dict[int, torch.Tensor] = {}
    for layer_idx, score_chunks in grouped.items():
        if not score_chunks:
            continue
        scores = torch.cat(score_chunks, dim=1)[0]
        if scores.size(0) > 0:
            layer_sequences[layer_idx] = scores
    return layer_sequences


def _simulate_least_stale_metrics(
    grouped: Dict[int, List[torch.Tensor]],
    cache_size: int,
    device: torch.device,
    count_from_token_idx: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    zero = torch.zeros((), device=device, dtype=torch.float64)
    layer_sequences = _build_selected_sequences(grouped)
    layer_ids = sorted(layer_sequences)
    layer_positions = {layer_idx: pos for pos, layer_idx in enumerate(layer_ids)}
    layer_count = len(layer_ids)
    total_capacity = max(0, int(cache_size) * layer_count)
    max_tokens = max((int(selected.size(0)) for selected in layer_sequences.values()), default=0)

    total_hit_rate = zero.clone()
    total_overlap = zero.clone()
    total_access = zero.clone()
    total_miss = zero.clone()
    counted_steps = 0
    access_order = 0
    cache: set[Tuple[int, int]] = set()
    cache_by_layer: Dict[int, set[int]] = {layer_idx: set() for layer_idx in layer_ids}
    current: set[Tuple[int, int]] = set()
    entry_order: Dict[Tuple[int, int], int] = {}

    for token_idx in range(max_tokens):
        current.clear()
        for current_layer_pos, layer_idx in enumerate(layer_ids):
            selected = layer_sequences[layer_idx]
            if token_idx >= selected.size(0):
                continue

            token_selected = selected[token_idx]
            token_experts = [int(expert) for expert in token_selected.detach().to(device="cpu", dtype=torch.long).tolist()]
            token_keys = [(layer_idx, expert) for expert in token_experts]
            unique_keys = set(token_keys)
            hit_count_int = sum(1 for key in token_keys if key in cache)
            should_count = count_from_token_idx is None or token_idx >= count_from_token_idx
            if should_count:
                access_count = torch.tensor(float(len(token_keys)), device=device, dtype=torch.float64)
                hit_count = torch.tensor(float(hit_count_int), device=device, dtype=torch.float64)
                total_overlap = total_overlap + hit_count
                total_hit_rate = total_hit_rate + hit_count / access_count
                total_access = total_access + access_count
                total_miss = total_miss + (access_count - hit_count)
                counted_steps += 1

            if total_capacity <= 0:
                continue
            access_order = _add_least_stale_entries(
                cache=cache,
                current=current,
                entry_order=entry_order,
                keys=unique_keys,
                access_order=access_order,
                cache_by_layer=cache_by_layer,
            )
            _trim_least_stale_cache(
                cache=cache,
                current=current,
                entry_order=entry_order,
                protected=unique_keys,
                layer_positions=layer_positions,
                current_layer_pos=current_layer_pos,
                layer_count=layer_count,
                total_capacity=total_capacity,
                cache_by_layer=cache_by_layer,
            )

    return {
        "hard_hit_rate_sum": total_hit_rate,
        "hard_overlap_sum": total_overlap,
        "hard_step_count": torch.tensor(float(counted_steps), device=device, dtype=torch.float64),
        "hard_access_count": total_access,
        "hard_miss_count": total_miss,
    }


def _infer_specmd_prefetch_budget(
    config: CachePolicyConfig,
    cache_size: int,
    selected_width: int,
) -> int:
    cache_capacity = max(0, int(cache_size))
    if cache_capacity <= 0:
        return 0
    requested_budget = int(config.prefetch_budget)
    if requested_budget == 0:
        return 0
    if requested_budget > 0:
        return min(requested_budget, cache_capacity)
    return cache_capacity


def infer_specmd_prefetch_budget(config: CachePolicyConfig, cache_size: int) -> int:
    return _infer_specmd_prefetch_budget(
        config=config,
        cache_size=cache_size,
        selected_width=0,
    )


def _add_least_stale_entries(
    cache: set[Tuple[int, int]],
    current: set[Tuple[int, int]],
    entry_order: Dict[Tuple[int, int], int],
    keys: set[Tuple[int, int]],
    access_order: int,
    cache_by_layer: Optional[Dict[int, set[int]]] = None,
) -> int:
    for key in keys:
        cache.add(key)
        if cache_by_layer is not None:
            cache_by_layer.setdefault(key[0], set()).add(key[1])
        current.add(key)
        if key not in entry_order:
            entry_order[key] = access_order
            access_order += 1
    return access_order


def _trim_least_stale_cache(
    cache: set[Tuple[int, int]],
    current: set[Tuple[int, int]],
    entry_order: Dict[Tuple[int, int], int],
    protected: set[Tuple[int, int]],
    layer_positions: Dict[int, int],
    current_layer_pos: int,
    layer_count: int,
    total_capacity: int,
    cache_by_layer: Optional[Dict[int, set[int]]] = None,
) -> None:
    overflow = len(cache) - total_capacity
    if overflow <= 0:
        return

    def remove_victim(victim: Tuple[int, int]) -> None:
        cache.discard(victim)
        current.discard(victim)
        entry_order.pop(victim, None)
        if cache_by_layer is not None:
            layer_cache = cache_by_layer.get(victim[0])
            if layer_cache is not None:
                layer_cache.discard(victim[1])

    if cache_by_layer is not None:
        layer_by_pos = {pos: layer_idx for layer_idx, pos in layer_positions.items()}
        layer_order = list(range(current_layer_pos, -1, -1)) + list(range(layer_count - 1, current_layer_pos, -1))
        victims: List[Tuple[int, int]] = []
        victim_set: set[Tuple[int, int]] = set()

        for ignore_protected in (False, True):
            for want_current in (False, True):
                for layer_pos in layer_order:
                    layer_idx = layer_by_pos.get(layer_pos)
                    if layer_idx is None:
                        continue
                    layer_cache = cache_by_layer.get(layer_idx, set())
                    candidates = []
                    for expert_id in layer_cache:
                        key = (layer_idx, expert_id)
                        if key in victim_set:
                            continue
                        if not ignore_protected and key in protected:
                            continue
                        if (key in current) != want_current:
                            continue
                        candidates.append(key)
                    if not candidates:
                        continue
                    candidates.sort(key=lambda key: entry_order.get(key, -1))
                    need = overflow - len(victims)
                    for victim in candidates[:need]:
                        victims.append(victim)
                        victim_set.add(victim)
                    if len(victims) >= overflow:
                        break
                if len(victims) >= overflow:
                    break
            if len(victims) >= overflow:
                break

        for victim in victims:
            remove_victim(victim)
        return

    def priority(key: Tuple[int, int]) -> tuple[int, int, int]:
        return _least_stale_evict_priority(
            key=key,
            current=current,
            entry_order=entry_order,
            layer_positions=layer_positions,
            current_layer_pos=current_layer_pos,
            layer_count=layer_count,
        )

    victims: List[Tuple[int, int]] = []
    candidates = [key for key in cache if key not in protected]
    if candidates:
        take_count = min(overflow, len(candidates))
        victims.extend(heapq.nlargest(take_count, candidates, key=priority))

    remaining = overflow - len(victims)
    if remaining > 0:
        victim_set = set(victims)
        fallback_candidates = [key for key in cache if key not in victim_set]
        victims.extend(heapq.nlargest(remaining, fallback_candidates, key=priority))

    for victim in victims:
        remove_victim(victim)


def _specmd_history_token_idx(
    current_token_idx: int,
    current_layer_pos: int,
    target_layer_pos: int,
) -> Optional[int]:
    # Use only the latest already-observed router distribution for the target layer.
    # This avoids both target-logit oracle replay and cross-layer expert-id matching.
    if target_layer_pos <= current_layer_pos:
        return current_token_idx
    history_token_idx = current_token_idx - 1
    if history_token_idx < 0:
        return None
    return history_token_idx


def _new_cache_metric_counter(count_from_token_idx: Optional[int]) -> Dict[str, Any]:
    return {
        "count_from_token_idx": count_from_token_idx,
        "hard_hit_rate_sum": 0.0,
        "hard_overlap_sum": 0.0,
        "hard_step_count": 0,
        "hard_access_count": 0.0,
        "hard_miss_count": 0.0,
        "hard_demand_miss_count": 0.0,
        "hard_prefetch_load_count": 0.0,
    }


def _counter_to_tensors(counter: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "hard_hit_rate_sum": torch.tensor(float(counter["hard_hit_rate_sum"]), device=device, dtype=torch.float64),
        "hard_overlap_sum": torch.tensor(float(counter["hard_overlap_sum"]), device=device, dtype=torch.float64),
        "hard_step_count": torch.tensor(float(counter["hard_step_count"]), device=device, dtype=torch.float64),
        "hard_access_count": torch.tensor(float(counter["hard_access_count"]), device=device, dtype=torch.float64),
        "hard_miss_count": torch.tensor(float(counter["hard_miss_count"]), device=device, dtype=torch.float64),
        "hard_demand_miss_count": torch.tensor(
            float(counter["hard_demand_miss_count"]),
            device=device,
            dtype=torch.float64,
        ),
        "hard_prefetch_load_count": torch.tensor(
            float(counter["hard_prefetch_load_count"]),
            device=device,
            dtype=torch.float64,
        ),
    }


def _simulate_specmd_prefetch_metrics_many(
    grouped_selected: Dict[int, List[torch.Tensor]],
    grouped_scores: Dict[int, List[torch.Tensor]],
    cache_size: int,
    device: torch.device,
    config: CachePolicyConfig,
    count_from_token_indices: List[Optional[int]],
) -> List[Dict[str, torch.Tensor]]:
    selected_sequences = _build_selected_sequences(grouped_selected)
    score_sequences = _build_score_sequences(grouped_scores)
    layer_ids = sorted(layer_idx for layer_idx in selected_sequences if layer_idx in score_sequences)
    if not layer_ids:
        raise ValueError(
            "--cache-policy specmd_prefetch requires recorded router_logits or current_probs. "
            "Use it with a method profile that records router outputs, such as baseline_lru."
        )

    layer_positions = {layer_idx: pos for pos, layer_idx in enumerate(layer_ids)}
    layer_count = len(layer_ids)
    total_capacity = max(0, int(cache_size) * layer_count)
    max_tokens = max(
        (
            min(int(selected_sequences[layer_idx].size(0)), int(score_sequences[layer_idx].size(0)))
            for layer_idx in layer_ids
        ),
        default=0,
    )
    selected_width = max((int(selected_sequences[layer_idx].size(-1)) for layer_idx in layer_ids), default=0)
    prefetch_budget = _infer_specmd_prefetch_budget(
        config=config,
        cache_size=cache_size,
        selected_width=selected_width,
    )
    prefetch_lookahead = max(0, int(config.prefetch_lookahead))

    selected_by_layer = {
        layer_idx: selected_sequences[layer_idx].detach().to(device="cpu", dtype=torch.long).tolist()
        for layer_idx in layer_ids
    }
    prefetch_candidates_by_layer: Dict[int, List[List[int]]] = {}
    if prefetch_budget > 0 and prefetch_lookahead > 0:
        for layer_idx in layer_ids:
            scores = score_sequences[layer_idx]
            k = min(prefetch_budget, int(scores.size(-1)))
            if k <= 0:
                prefetch_candidates_by_layer[layer_idx] = []
                continue
            topk = torch.topk(scores.detach(), k=k, dim=-1, largest=True, sorted=False).indices
            prefetch_candidates_by_layer[layer_idx] = topk.to(device="cpu", dtype=torch.long).tolist()

    counters = [_new_cache_metric_counter(count_from_token_idx) for count_from_token_idx in count_from_token_indices]
    access_order = 0
    cache: set[Tuple[int, int]] = set()
    cache_by_layer: Dict[int, set[int]] = {layer_idx: set() for layer_idx in layer_ids}
    current: set[Tuple[int, int]] = set()
    prefetched_by_token: Dict[int, set[Tuple[int, int]]] = {}
    entry_order: Dict[Tuple[int, int], int] = {}

    for token_idx in range(max_tokens):
        current = {key for key in prefetched_by_token.pop(token_idx, set()) if key in cache}
        for current_layer_pos, layer_idx in enumerate(layer_ids):
            selected = selected_by_layer[layer_idx]
            if token_idx >= len(selected):
                continue

            token_experts = [int(expert) for expert in selected[token_idx]]
            token_keys = [(layer_idx, expert) for expert in token_experts]
            unique_keys = set(token_keys)
            hit_count_int = sum(1 for key in token_keys if key in cache)
            access_count = float(len(token_keys))
            if access_count > 0:
                hit_count = float(hit_count_int)
                demand_miss = access_count - hit_count
                for counter in counters:
                    count_from_token_idx = counter["count_from_token_idx"]
                    if count_from_token_idx is not None and token_idx < count_from_token_idx:
                        continue
                    counter["hard_overlap_sum"] += hit_count
                    counter["hard_hit_rate_sum"] += hit_count / access_count
                    counter["hard_access_count"] += access_count
                    counter["hard_demand_miss_count"] += demand_miss
                    counter["hard_miss_count"] += demand_miss
                    counter["hard_step_count"] += 1

            if total_capacity <= 0:
                continue

            access_order = _add_least_stale_entries(
                cache=cache,
                current=current,
                entry_order=entry_order,
                keys=unique_keys,
                access_order=access_order,
                cache_by_layer=cache_by_layer,
            )
            _trim_least_stale_cache(
                cache=cache,
                current=current,
                entry_order=entry_order,
                protected=unique_keys,
                layer_positions=layer_positions,
                current_layer_pos=current_layer_pos,
                layer_count=layer_count,
                total_capacity=total_capacity,
                cache_by_layer=cache_by_layer,
            )

            if prefetch_budget <= 0 or prefetch_lookahead <= 0:
                continue

            for offset in range(1, prefetch_lookahead + 1):
                absolute_pos = current_layer_pos + offset
                target_token_idx = token_idx + absolute_pos // layer_count
                if target_token_idx >= max_tokens:
                    continue
                target_layer_pos = absolute_pos % layer_count
                target_layer_idx = layer_ids[target_layer_pos]
                history_token_idx = _specmd_history_token_idx(
                    current_token_idx=token_idx,
                    current_layer_pos=current_layer_pos,
                    target_layer_pos=target_layer_pos,
                )
                if history_token_idx is None:
                    continue
                prefetch_candidates = prefetch_candidates_by_layer.get(target_layer_idx, [])
                if history_token_idx >= len(prefetch_candidates):
                    continue
                expert_ids = prefetch_candidates[history_token_idx]
                if not expert_ids:
                    continue
                prefetch_keys = {(target_layer_idx, int(expert_id)) for expert_id in expert_ids}
                new_prefetch_keys = {key for key in prefetch_keys if key not in cache}
                if new_prefetch_keys:
                    prefetch_load = float(len(new_prefetch_keys))
                    for counter in counters:
                        count_from_token_idx = counter["count_from_token_idx"]
                        if count_from_token_idx is not None and target_token_idx < count_from_token_idx:
                            continue
                        counter["hard_prefetch_load_count"] += prefetch_load
                        counter["hard_miss_count"] += prefetch_load
                if target_token_idx == token_idx:
                    target_current = current
                else:
                    target_current = prefetched_by_token.setdefault(target_token_idx, set())
                access_order = _add_least_stale_entries(
                    cache=cache,
                    current=target_current,
                    entry_order=entry_order,
                    keys=prefetch_keys,
                    access_order=access_order,
                    cache_by_layer=cache_by_layer,
                )
                _trim_least_stale_cache(
                    cache=cache,
                    current=current,
                    entry_order=entry_order,
                    protected=prefetch_keys,
                    layer_positions=layer_positions,
                    current_layer_pos=current_layer_pos,
                    layer_count=layer_count,
                    total_capacity=total_capacity,
                    cache_by_layer=cache_by_layer,
                )

    return [_counter_to_tensors(counter, device=device) for counter in counters]


def _simulate_specmd_prefetch_metrics(
    grouped_selected: Dict[int, List[torch.Tensor]],
    grouped_scores: Dict[int, List[torch.Tensor]],
    cache_size: int,
    device: torch.device,
    config: CachePolicyConfig,
    count_from_token_idx: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    return _simulate_specmd_prefetch_metrics_many(
        grouped_selected=grouped_selected,
        grouped_scores=grouped_scores,
        cache_size=cache_size,
        device=device,
        config=config,
        count_from_token_indices=[count_from_token_idx],
    )[0]


def build_promoe_trace_sample(
    layer_stats: List[Dict[str, torch.Tensor]],
    sample_index: int,
) -> Dict[str, Any]:
    grouped: Dict[int, Dict[str, Any]] = {}
    for layer in layer_stats:
        selected = layer.get("selected_experts")
        promoe_inputs = layer.get("promoe_inputs")
        if selected is None or promoe_inputs is None:
            continue
        layer_idx = int(layer["layer_idx"])
        selected_cpu = selected.detach().to(device="cpu", dtype=torch.int16)
        inputs_cpu = promoe_inputs.detach().to(device="cpu", dtype=torch.float16)
        if selected_cpu.dim() == 3:
            selected_cpu = selected_cpu.reshape(-1, selected_cpu.size(-1))
        if inputs_cpu.dim() == 3:
            inputs_cpu = inputs_cpu.reshape(-1, inputs_cpu.size(-1))
        current_probs = layer.get("current_probs")
        router_logits = layer.get("router_logits")
        expert_maps = None
        if current_probs is not None:
            expert_maps = current_probs.detach().to(device="cpu", dtype=torch.float16)
            if expert_maps.dim() == 3:
                expert_maps = expert_maps.reshape(-1, expert_maps.size(-1))
            num_experts = int(expert_maps.size(-1))
        elif router_logits is not None:
            router_probs = torch.softmax(router_logits.detach().to(device="cpu", dtype=torch.float32), dim=-1)
            expert_maps = router_probs.to(dtype=torch.float16)
            if expert_maps.dim() == 3:
                expert_maps = expert_maps.reshape(-1, expert_maps.size(-1))
            num_experts = int(expert_maps.size(-1))
        elif selected_cpu.numel() > 0:
            num_experts = int(selected_cpu.to(dtype=torch.long).max().item()) + 1
        else:
            num_experts = 0
        if expert_maps is None and num_experts > 0:
            selected_long = selected_cpu.to(dtype=torch.long)
            expert_maps_float = torch.zeros((selected_long.size(0), num_experts), dtype=torch.float32)
            topk_width = max(int(selected_long.size(-1)), 1)
            expert_maps_float.scatter_add_(
                1,
                selected_long,
                torch.full(selected_long.shape, 1.0 / float(topk_width), dtype=torch.float32),
            )
            expert_maps = expert_maps_float.to(dtype=torch.float16)
        bucket = grouped.setdefault(
            layer_idx,
            {"selected": [], "promoe_inputs": [], "expert_maps": [], "num_experts": num_experts},
        )
        bucket["selected"].append(selected_cpu)
        bucket["promoe_inputs"].append(inputs_cpu)
        if expert_maps is not None:
            bucket["expert_maps"].append(expert_maps)
        bucket["num_experts"] = max(int(bucket["num_experts"]), num_experts)

    layers = []
    for layer_idx, bucket in grouped.items():
        layer_payload = {
            "layer_idx": layer_idx,
            "selected_experts": torch.cat(bucket["selected"], dim=0),
            "promoe_inputs": torch.cat(bucket["promoe_inputs"], dim=0),
            "num_experts": int(bucket["num_experts"]),
        }
        if bucket.get("expert_maps"):
            layer_payload["expert_maps"] = torch.cat(bucket["expert_maps"], dim=0)
        layers.append(
            layer_payload
        )
    layers.sort(key=lambda item: int(item["layer_idx"]))
    return {
        "format": "promoe_moe_layer_logits_trace_sample_v1",
        "index": int(sample_index),
        "layers": layers,
    }


def _touch_layer_cache(
    cache: List[int],
    expert_id: int,
    cache_size: int,
) -> None:
    if cache_size <= 0:
        return
    if expert_id in cache:
        cache.remove(expert_id)
    cache.append(expert_id)
    while len(cache) > cache_size:
        cache.pop(0)


def _predict_promoe_layers(
    predictor: Dict[str, Any],
    source_layer_idx: int,
    source_input: torch.Tensor,
    target_layer_ids: List[int],
    budget: int,
) -> Dict[int, List[int]]:
    predictions: Dict[int, List[int]] = {}
    if budget <= 0:
        return predictions
    for target_layer_idx in target_layer_ids:
        scores = promoe_predict(
            predictor=predictor,
            source_layer_idx=source_layer_idx,
            target_layer_idx=target_layer_idx,
            layer_input=source_input,
        )
        if scores is None or scores.numel() == 0:
            continue
        k = max(1, min(int(budget), int(scores.numel())))
        predictions[target_layer_idx] = [int(item) for item in torch.topk(scores, k=k, largest=True).indices.tolist()]
    return predictions


def _apply_promoe_prefetch(
    layer_caches: Dict[int, List[int]],
    predictions: Dict[int, List[int]],
    cache_size: int,
    device: torch.device,
    total_load: torch.Tensor,
    total_prefetch_load: torch.Tensor,
    count_from_token_idx: Optional[int],
    target_token_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_size <= 0:
        return total_load, total_prefetch_load
    should_count = count_from_token_idx is None or target_token_idx >= count_from_token_idx
    for target_layer_idx, expert_ids in predictions.items():
        cache = layer_caches.setdefault(target_layer_idx, [])
        for expert_id in expert_ids:
            if expert_id not in cache and should_count:
                load_count = torch.tensor(1.0, device=device, dtype=torch.float64)
                total_load = total_load + load_count
                total_prefetch_load = total_prefetch_load + load_count
            _touch_layer_cache(cache, expert_id, cache_size)
    return total_load, total_prefetch_load


def infer_promoe_prefetch_budget(
    config: CachePolicyConfig,
    cache_size: int,
) -> int:
    cache_capacity = max(0, int(cache_size))
    if cache_capacity <= 0:
        return 0

    requested_budget = int(config.prefetch_budget)
    if requested_budget == 0:
        return 0
    if requested_budget > 0:
        return min(requested_budget, cache_capacity)

    predictor_config = config.promoe_predictor.get("config", {}) if config.promoe_predictor else {}
    predictor_budget = int(predictor_config.get("num_predict_expert_per_layer", 0) or 0)
    if predictor_budget <= 0:
        return 0
    return min(predictor_budget, cache_capacity)


def _infer_promoe_prefetch_budget(
    config: CachePolicyConfig,
    cache_size: int,
) -> int:
    return infer_promoe_prefetch_budget(config=config, cache_size=cache_size)


def infer_finemoe_prefetch_budget(config: CachePolicyConfig, cache_size: int) -> int:
    cache_capacity = max(0, int(cache_size))
    if cache_capacity <= 0:
        return 0
    requested_budget = int(config.prefetch_budget)
    if requested_budget == 0:
        return 0
    if requested_budget > 0:
        return min(requested_budget, cache_capacity)
    return cache_capacity


def _probability_like_scores(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.detach().to(device="cpu", dtype=torch.float32)
    if scores.numel() == 0:
        return scores
    if float(scores.min().item()) < 0.0 or float(scores.max().item()) > 1.0:
        return torch.softmax(scores, dim=-1)
    row_sums = scores.sum(dim=-1, keepdim=True)
    if float(row_sums.max().item()) > 1.5:
        return scores / row_sums.clamp_min(1e-12)
    return scores


def _sparse_probability_from_selected(token_selected: List[int], num_experts: int) -> torch.Tensor:
    probs = torch.zeros((max(0, int(num_experts)),), dtype=torch.float32)
    if probs.numel() <= 0 or not token_selected:
        return probs
    value = 1.0 / float(len(token_selected))
    for expert_id in token_selected:
        if 0 <= int(expert_id) < probs.numel():
            probs[int(expert_id)] += value
    return probs


def _select_finemoe_prefetch_experts(
    scores: torch.Tensor,
    threshold: float,
    budget: int,
    min_count: int,
) -> List[int]:
    if budget <= 0 or scores.numel() <= 0:
        return []
    k_max = min(max(int(budget), 0), int(scores.numel()))
    if k_max <= 0:
        return []
    sorted_scores, sorted_indices = torch.sort(scores.detach().to(dtype=torch.float32), descending=True)
    if float(threshold) > 0.0:
        cumulative = torch.cumsum(sorted_scores.clamp_min(0.0), dim=0)
        meets = torch.nonzero(cumulative >= float(threshold), as_tuple=False)
        target_count = int(meets[0].item()) + 1 if meets.numel() > 0 else k_max
    else:
        target_count = k_max
    target_count = min(k_max, max(int(min_count), target_count))
    return [int(item) for item in sorted_indices[:target_count].tolist()]


def _finemoe_evict_if_needed(
    cache: set[int],
    freq: Dict[int, float],
    predicted_scores: Optional[torch.Tensor],
    cache_size: int,
    protected: set[int],
    frequency_weight: float,
    probability_weight: float,
) -> None:
    while len(cache) > max(0, int(cache_size)):
        candidates = [expert_id for expert_id in cache if expert_id not in protected]
        if not candidates:
            candidates = list(cache)
        if not candidates:
            return

        def score(expert_id: int) -> tuple[float, float]:
            prob = 0.0
            if predicted_scores is not None and 0 <= expert_id < predicted_scores.numel():
                prob = float(predicted_scores[expert_id].item())
            return (
                float(frequency_weight) * float(freq.get(expert_id, 0.0))
                + float(probability_weight) * prob,
                float(freq.get(expert_id, 0.0)),
            )

        victim = min(candidates, key=score)
        cache.remove(victim)


def _simulate_finemoe_metrics(
    grouped_selected: Dict[int, List[torch.Tensor]],
    grouped_inputs: Dict[int, List[torch.Tensor]],
    grouped_scores: Dict[int, List[torch.Tensor]],
    cache_size: int,
    device: torch.device,
    config: CachePolicyConfig,
    count_from_token_idx: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    if config.finemoe_store is None:
        raise ValueError("--cache-policy finemoe requires --finemoe-store-path.")

    zero = torch.zeros((), device=device, dtype=torch.float64)
    selected_sequences = _build_selected_sequences(grouped_selected)
    if not selected_sequences:
        raise ValueError("--cache-policy finemoe requires recorded selected_experts.")

    input_sequences: Dict[int, torch.Tensor] = {}
    for layer_idx, chunks in grouped_inputs.items():
        if chunks:
            inputs = torch.cat(chunks, dim=1)[0].to(dtype=torch.float32, device="cpu")
            if inputs.size(0) > 0:
                input_sequences[layer_idx] = inputs

    score_sequences: Dict[int, torch.Tensor] = {}
    for layer_idx, chunks in grouped_scores.items():
        if chunks:
            scores = torch.cat(chunks, dim=1)[0].to(dtype=torch.float32, device="cpu")
            if scores.size(0) > 0:
                score_sequences[layer_idx] = _probability_like_scores(scores)

    if not input_sequences:
        raise ValueError(
            "--cache-policy finemoe requires recorded promoe_inputs. "
            "Use it with a method profile that records router inputs, such as baseline_lru."
        )

    layer_ids = sorted(selected_sequences)
    layer_positions = {layer_idx: pos for pos, layer_idx in enumerate(layer_ids)}
    layer_count = len(layer_ids)
    source_layer_idx = min(input_sequences)
    source_inputs = input_sequences[source_layer_idx]
    max_tokens = max((int(selected.size(0)) for selected in selected_sequences.values()), default=0)
    selected_width = max((int(selected_sequences[layer_idx].size(-1)) for layer_idx in layer_ids), default=0)
    prefetch_budget = infer_finemoe_prefetch_budget(config, cache_size=cache_size)
    prefetch_distance = max(1, int(config.prefetch_lookahead))

    selected_by_layer = {
        layer_idx: selected_sequences[layer_idx].detach().to(device="cpu", dtype=torch.long).tolist()
        for layer_idx in layer_ids
    }
    max_experts = max(
        [
            int(score_sequences[layer_idx].size(-1))
            for layer_idx in score_sequences
            if score_sequences[layer_idx].dim() == 2
        ]
        + [
            int(selected_sequences[layer_idx].max().item()) + 1
            for layer_idx in selected_sequences
            if selected_sequences[layer_idx].numel() > 0
        ],
        default=0,
    )

    total_hit_rate = zero.clone()
    total_overlap = zero.clone()
    total_access = zero.clone()
    total_load = zero.clone()
    total_demand_miss = zero.clone()
    total_prefetch_load = zero.clone()
    counted_steps = 0

    layer_caches: Dict[int, set[int]] = {layer_idx: set() for layer_idx in layer_ids}
    layer_freq: Dict[int, Dict[int, float]] = {layer_idx: {} for layer_idx in layer_ids}
    last_predicted_scores: Dict[int, torch.Tensor] = {}

    def token_score(layer_idx: int, token_idx: int, token_experts: Optional[List[int]] = None) -> torch.Tensor:
        scores = score_sequences.get(layer_idx)
        if scores is not None and token_idx < scores.size(0):
            return scores[token_idx]
        return _sparse_probability_from_selected(token_experts or [], max_experts)

    def insert_experts(
        layer_idx: int,
        expert_ids: List[int],
        predicted_scores: Optional[torch.Tensor],
        protected: set[int],
        *,
        count_load: bool,
    ) -> int:
        if cache_size <= 0:
            return 0
        cache = layer_caches.setdefault(layer_idx, set())
        freq = layer_freq.setdefault(layer_idx, {})
        load_count = 0
        for expert_id in expert_ids:
            expert_id = int(expert_id)
            if expert_id not in cache:
                if count_load:
                    load_count += 1
                cache.add(expert_id)
            _finemoe_evict_if_needed(
                cache=cache,
                freq=freq,
                predicted_scores=predicted_scores,
                cache_size=cache_size,
                protected=protected,
                frequency_weight=config.frequency_weight,
                probability_weight=config.finemoe_eviction_probability_weight,
            )
        return load_count

    def prefetch_for_layer(
        layer_idx: int,
        token_idx: int,
        semantic_candidates: Tuple[torch.Tensor, torch.Tensor],
        observed_maps: Dict[int, torch.Tensor],
    ) -> None:
        nonlocal total_load, total_prefetch_load
        if prefetch_budget <= 0 or cache_size <= 0:
            return
        predicted_scores, confidence = finemoe_predict_expert_scores(
            config.finemoe_store,
            semantic_candidates,
            target_layer_idx=layer_idx,
            observed_maps=observed_maps,
            top_k=config.finemoe_top_k,
            semantic_weight=config.finemoe_semantic_weight,
            trajectory_weight=config.finemoe_trajectory_weight,
        )
        if predicted_scores is None or predicted_scores.numel() <= 0:
            return
        threshold = finemoe_dynamic_threshold(
            confidence,
            fixed_threshold=config.finemoe_prefetch_threshold,
            min_threshold=config.finemoe_min_prefetch_threshold,
            max_threshold=config.finemoe_max_prefetch_threshold,
        )
        expert_ids = _select_finemoe_prefetch_experts(
            predicted_scores,
            threshold=threshold,
            budget=prefetch_budget,
            min_count=min(selected_width, prefetch_budget),
        )
        if not expert_ids:
            return
        protected = set(expert_ids)
        should_count = count_from_token_idx is None or token_idx >= count_from_token_idx
        prefetch_load_count = insert_experts(
            layer_idx=layer_idx,
            expert_ids=expert_ids,
            predicted_scores=predicted_scores,
            protected=protected,
            count_load=should_count,
        )
        last_predicted_scores[layer_idx] = predicted_scores
        if should_count and prefetch_load_count > 0:
            load = torch.tensor(float(prefetch_load_count), device=device, dtype=torch.float64)
            total_prefetch_load = total_prefetch_load + load
            total_load = total_load + load

    for token_idx in range(max_tokens):
        if token_idx < source_inputs.size(0):
            semantic_candidates = finemoe_semantic_candidates(
                config.finemoe_store,
                source_inputs[token_idx],
                candidate_pool=config.finemoe_candidate_pool,
            )
        else:
            semantic_candidates = (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32))

        observed_maps: Dict[int, torch.Tensor] = {}
        for initial_layer_pos in range(min(prefetch_distance, layer_count)):
            prefetch_for_layer(
                layer_ids[initial_layer_pos],
                token_idx,
                semantic_candidates,
                observed_maps,
            )

        for current_layer_pos, layer_idx in enumerate(layer_ids):
            selected = selected_by_layer[layer_idx]
            if token_idx >= len(selected):
                continue

            token_experts = [int(expert) for expert in selected[token_idx]]
            unique_experts = set(token_experts)
            cache = layer_caches.setdefault(layer_idx, set())
            hit_count_int = sum(1 for expert_id in token_experts if expert_id in cache)
            access_count_int = len(token_experts)
            should_count = count_from_token_idx is None or token_idx >= count_from_token_idx
            if access_count_int > 0 and should_count:
                access_count = torch.tensor(float(access_count_int), device=device, dtype=torch.float64)
                hit_count = torch.tensor(float(hit_count_int), device=device, dtype=torch.float64)
                demand_miss = access_count - hit_count
                total_overlap = total_overlap + hit_count
                total_hit_rate = total_hit_rate + hit_count / access_count
                total_access = total_access + access_count
                total_demand_miss = total_demand_miss + demand_miss
                total_load = total_load + demand_miss
                counted_steps += 1

            freq = layer_freq.setdefault(layer_idx, {})
            for expert_id in token_experts:
                freq[expert_id] = float(freq.get(expert_id, 0.0)) + 1.0
            demand_scores = last_predicted_scores.get(layer_idx)
            insert_experts(
                layer_idx=layer_idx,
                expert_ids=list(unique_experts),
                predicted_scores=demand_scores,
                protected=unique_experts,
                count_load=False,
            )

            observed_maps[layer_idx] = token_score(layer_idx, token_idx, token_experts)
            target_layer_pos = current_layer_pos + prefetch_distance
            if target_layer_pos < layer_count:
                prefetch_for_layer(
                    layer_ids[target_layer_pos],
                    token_idx,
                    semantic_candidates,
                    observed_maps,
                )

    return {
        "hard_hit_rate_sum": total_hit_rate,
        "hard_overlap_sum": total_overlap,
        "hard_step_count": torch.tensor(float(counted_steps), device=device, dtype=torch.float64),
        "hard_access_count": total_access,
        "hard_miss_count": total_load,
        "hard_demand_miss_count": total_demand_miss,
        "hard_prefetch_load_count": total_prefetch_load,
    }


def _build_future_prefetch_cache(
    priority_scores: torch.Tensor,
    cache_size: int,
) -> torch.Tensor:
    capacity = max(0, min(int(cache_size), int(priority_scores.size(-1))))
    if capacity <= 0:
        return torch.empty(0, device=priority_scores.device, dtype=torch.long)
    ranked = torch.argsort(priority_scores.detach(), descending=True).to(dtype=torch.long)
    return ranked[:capacity]


def _future_prefetch_priority_scores(
    current_scores: Optional[torch.Tensor],
    future_scores: torch.Tensor,
    priority_mode: str,
) -> torch.Tensor:
    mode = str(priority_mode or "current_plus_future").lower()
    if mode in {"future_only", "future", "mlp_only", "predictor_only"}:
        return future_scores
    if mode not in {"current_plus_future", "current+future", "hybrid", "sum"}:
        raise ValueError(f"Unsupported future_cache_priority_mode: {priority_mode!r}")
    if current_scores is None:
        return future_scores
    return current_scores + future_scores


def _simulate_future_prefetch_metrics(
    grouped_selected: Dict[int, List[torch.Tensor]],
    grouped_current: Dict[int, List[torch.Tensor]],
    grouped_future: Dict[int, List[torch.Tensor]],
    cache_size: int,
    device: torch.device,
    future_cache_priority_mode: str = "current_plus_future",
    count_from_token_idx: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    zero = torch.zeros((), device=device, dtype=torch.float64)
    selected_sequences = _build_selected_sequences(grouped_selected)
    current_sequences: Dict[int, torch.Tensor] = {}
    future_sequences: Dict[int, torch.Tensor] = {}
    for layer_idx, chunks in grouped_current.items():
        if chunks:
            current_sequences[layer_idx] = torch.cat(chunks, dim=1)[0].to(device=device, dtype=torch.float32)
    for layer_idx, chunks in grouped_future.items():
        if chunks:
            future_sequences[layer_idx] = torch.cat(chunks, dim=1)[0].to(device=device, dtype=torch.float32)

    total_hit_rate = zero.clone()
    total_overlap = zero.clone()
    total_access = zero.clone()
    total_load = zero.clone()
    total_demand_miss = zero.clone()
    total_prefetch_load = zero.clone()
    counted_steps = 0
    valid_layer_count = 0

    for layer_idx in sorted(selected_sequences):
        selected = selected_sequences[layer_idx]
        current = current_sequences.get(layer_idx)
        future = future_sequences.get(layer_idx)
        if future is None:
            continue
        if current is None and str(future_cache_priority_mode or "").lower() not in {
            "future_only",
            "future",
            "mlp_only",
            "predictor_only",
        }:
            continue
        max_tokens = min(
            int(selected.size(0)),
            int(future.size(0)),
            int(current.size(0)) if current is not None else int(future.size(0)),
        )
        if max_tokens <= 0:
            continue
        valid_layer_count += 1
        cache = torch.empty(0, device=device, dtype=torch.long)

        for token_idx in range(max_tokens):
            token_selected = selected[token_idx].detach().to(device=device, dtype=torch.long)
            selected_unique = torch.unique(token_selected)
            hit_count = _contains_any(token_selected, cache).to(dtype=torch.float64).sum()
            access_count = torch.tensor(float(token_selected.numel()), device=device, dtype=torch.float64)
            demand_miss = access_count - hit_count
            should_count = count_from_token_idx is None or token_idx >= count_from_token_idx
            if should_count:
                total_overlap = total_overlap + hit_count
                total_hit_rate = total_hit_rate + hit_count / access_count
                total_access = total_access + access_count
                total_demand_miss = total_demand_miss + demand_miss
                total_load = total_load + demand_miss
                counted_steps += 1

            if token_idx + 1 >= max_tokens:
                continue

            priority_scores = _future_prefetch_priority_scores(
                current_scores=current[token_idx] if current is not None else None,
                future_scores=future[token_idx],
                priority_mode=future_cache_priority_mode,
            )
            next_cache = _build_future_prefetch_cache(
                priority_scores=priority_scores,
                cache_size=cache_size,
            )
            cache_after_demand = torch.unique(torch.cat([cache, selected_unique])) if cache.numel() else selected_unique
            prefetch_new = next_cache[~_contains_any(next_cache, cache_after_demand)]
            target_token_idx = token_idx + 1
            should_count_prefetch = count_from_token_idx is None or target_token_idx >= count_from_token_idx
            if should_count_prefetch and prefetch_new.numel() > 0:
                prefetch_load = torch.tensor(float(prefetch_new.numel()), device=device, dtype=torch.float64)
                total_prefetch_load = total_prefetch_load + prefetch_load
                total_load = total_load + prefetch_load
            cache = next_cache

    if valid_layer_count == 0:
        raise ValueError(
            "--cache-policy future_prefetch requires recorded current_probs and future_probs. "
            "Use it with a future-router/cache_moe checkpoint and method profile."
        )

    return {
        "hard_hit_rate_sum": total_hit_rate,
        "hard_overlap_sum": total_overlap,
        "hard_step_count": torch.tensor(float(counted_steps), device=device, dtype=torch.float64),
        "hard_access_count": total_access,
        "hard_miss_count": total_load,
        "hard_demand_miss_count": total_demand_miss,
        "hard_prefetch_load_count": total_prefetch_load,
    }


def _simulate_promoe_metrics(
    grouped_selected: Dict[int, List[torch.Tensor]],
    grouped_inputs: Dict[int, List[torch.Tensor]],
    cache_size: int,
    device: torch.device,
    config: CachePolicyConfig,
    count_from_token_idx: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    if config.promoe_predictor is None:
        raise ValueError("--cache-policy promoe requires --promoe-predictor-path.")

    predictor = load_promoe_predictor_runtime(config.promoe_predictor)
    zero = torch.zeros((), device=device, dtype=torch.float64)
    selected_sequences = _build_selected_sequences(grouped_selected)
    input_sequences: Dict[int, torch.Tensor] = {}
    for layer_idx, chunks in grouped_inputs.items():
        if chunks:
            inputs = torch.cat(chunks, dim=1)[0].to(dtype=torch.float32, device="cpu")
            if inputs.size(0) > 0:
                input_sequences[layer_idx] = inputs

    prefetch_budget = _infer_promoe_prefetch_budget(config, cache_size=cache_size)
    layer_ids = sorted(selected_sequences)
    num_layers = len(layer_ids)
    layer_pos_to_idx = {pos: layer_idx for pos, layer_idx in enumerate(layer_ids)}
    layer_idx_to_pos = {layer_idx: pos for pos, layer_idx in enumerate(layer_ids)}
    mapping = build_promoe_layer_mapping(
        num_layers=num_layers,
        interval=config.promoe_layer_predict_interval,
        max_window=config.prefetch_lookahead,
        replace_first_input_with_last_output=config.promoe_use_last_output,
    )
    max_tokens = max((int(selected.size(0)) for selected in selected_sequences.values()), default=0)

    total_hit_rate = zero.clone()
    total_overlap = zero.clone()
    total_access = zero.clone()
    total_load = zero.clone()
    total_demand_miss = zero.clone()
    total_prefetch_load = zero.clone()
    counted_steps = 0
    layer_caches: Dict[int, List[int]] = {layer_idx: [] for layer_idx in layer_ids}

    for token_idx in range(max_tokens):
        if config.promoe_use_last_output and token_idx > 0 and layer_ids:
            last_layer_idx = layer_ids[-1]
            if last_layer_idx in input_sequences and token_idx - 1 < input_sequences[last_layer_idx].size(0):
                target_positions = mapping.get(num_layers, [])
                target_layer_ids = [layer_pos_to_idx[pos] for pos in target_positions if pos in layer_pos_to_idx]
                predictions = _predict_promoe_layers(
                    predictor=predictor,
                    source_layer_idx=num_layers,
                    source_input=input_sequences[last_layer_idx][token_idx - 1],
                    target_layer_ids=target_layer_ids,
                    budget=prefetch_budget,
                )
                total_load, total_prefetch_load = _apply_promoe_prefetch(
                    layer_caches=layer_caches,
                    predictions=predictions,
                    cache_size=cache_size,
                    device=device,
                    total_load=total_load,
                    total_prefetch_load=total_prefetch_load,
                    count_from_token_idx=count_from_token_idx,
                    target_token_idx=token_idx,
                )

        for layer_idx in layer_ids:
            selected = selected_sequences[layer_idx]
            if token_idx >= selected.size(0):
                continue

            token_selected = selected[token_idx]
            token_experts = [int(expert) for expert in token_selected.detach().to(device="cpu", dtype=torch.long).tolist()]
            cache = layer_caches.setdefault(layer_idx, [])
            hit_count_int = sum(1 for expert_id in token_experts if expert_id in cache)
            demand_miss_count_int = len(token_experts) - hit_count_int
            should_count = count_from_token_idx is None or token_idx >= count_from_token_idx
            if should_count:
                access_count = torch.tensor(float(len(token_experts)), device=device, dtype=torch.float64)
                hit_count = torch.tensor(float(hit_count_int), device=device, dtype=torch.float64)
                demand_miss_count = torch.tensor(float(demand_miss_count_int), device=device, dtype=torch.float64)
                total_overlap = total_overlap + hit_count
                total_hit_rate = total_hit_rate + hit_count / access_count
                total_access = total_access + access_count
                total_demand_miss = total_demand_miss + demand_miss_count
                total_load = total_load + demand_miss_count
                counted_steps += 1

            for expert_id in token_experts:
                _touch_layer_cache(cache, expert_id, cache_size)

            layer_pos = layer_idx_to_pos[layer_idx]
            if layer_idx not in input_sequences or token_idx >= input_sequences[layer_idx].size(0):
                continue
            target_positions = mapping.get(layer_pos, [])
            target_layer_ids = [layer_pos_to_idx[pos] for pos in target_positions if pos in layer_pos_to_idx]
            if not target_layer_ids:
                continue
            predictions = _predict_promoe_layers(
                predictor=predictor,
                source_layer_idx=layer_pos,
                source_input=input_sequences[layer_idx][token_idx],
                target_layer_ids=target_layer_ids,
                budget=prefetch_budget,
            )
            total_load, total_prefetch_load = _apply_promoe_prefetch(
                layer_caches=layer_caches,
                predictions=predictions,
                cache_size=cache_size,
                device=device,
                total_load=total_load,
                total_prefetch_load=total_prefetch_load,
                count_from_token_idx=count_from_token_idx,
                target_token_idx=token_idx,
            )

    return {
        "hard_hit_rate_sum": total_hit_rate,
        "hard_overlap_sum": total_overlap,
        "hard_step_count": torch.tensor(float(counted_steps), device=device, dtype=torch.float64),
        "hard_access_count": total_access,
        "hard_miss_count": total_load,
        "hard_demand_miss_count": total_demand_miss,
        "hard_prefetch_load_count": total_prefetch_load,
    }


def _group_cache_policy_layer_stats(
    layer_stats: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> tuple[
    Dict[int, List[torch.Tensor]],
    Dict[int, List[torch.Tensor]],
    Dict[int, List[torch.Tensor]],
    Dict[int, List[torch.Tensor]],
    Dict[int, List[torch.Tensor]],
]:
    grouped_selected: Dict[int, List[torch.Tensor]] = {}
    grouped_inputs: Dict[int, List[torch.Tensor]] = {}
    grouped_current: Dict[int, List[torch.Tensor]] = {}
    grouped_future: Dict[int, List[torch.Tensor]] = {}
    grouped_router_scores: Dict[int, List[torch.Tensor]] = {}
    for layer in layer_stats:
        selected = layer.get("selected_experts")
        if selected is None:
            continue
        layer_idx = int(layer["layer_idx"])
        grouped_selected.setdefault(layer_idx, []).append(selected.detach().to(device=device, dtype=torch.long))
        promoe_inputs = layer.get("promoe_inputs")
        if promoe_inputs is not None:
            grouped_inputs.setdefault(layer_idx, []).append(promoe_inputs.detach().to(device="cpu", dtype=torch.float32))
        current_probs = layer.get("current_probs")
        if current_probs is not None:
            current_probs = current_probs.detach().to(device=device, dtype=torch.float32)
            grouped_current.setdefault(layer_idx, []).append(current_probs)
            grouped_router_scores.setdefault(layer_idx, []).append(current_probs)
        future_probs = layer.get("future_probs")
        if future_probs is not None:
            grouped_future.setdefault(layer_idx, []).append(future_probs.detach().to(device=device, dtype=torch.float32))
        router_logits = layer.get("router_logits")
        if router_logits is not None and current_probs is None:
            router_scores = router_logits.detach().to(device=device, dtype=torch.float32)
            grouped_router_scores.setdefault(layer_idx, []).append(router_scores)
    return grouped_selected, grouped_inputs, grouped_current, grouped_future, grouped_router_scores


def simulate_cache_policy_metrics(
    layer_stats: List[Dict[str, torch.Tensor]],
    cache_size: int,
    device: torch.device,
    config: CachePolicyConfig,
    count_from_token_idx: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    zero = torch.zeros((), device=device, dtype=torch.float64)
    (
        grouped_selected,
        grouped_inputs,
        grouped_current,
        grouped_future,
        grouped_router_scores,
    ) = _group_cache_policy_layer_stats(layer_stats=layer_stats, device=device)

    total_hit_rate = zero.clone()
    total_overlap = zero.clone()
    total_access = zero.clone()
    total_miss = zero.clone()
    counted_steps = 0
    policy = config.policy.lower()

    if policy == "least_stale":
        return _simulate_least_stale_metrics(
            grouped=grouped_selected,
            cache_size=cache_size,
            device=device,
            count_from_token_idx=count_from_token_idx,
        )
    if policy == "promoe":
        return _simulate_promoe_metrics(
            grouped_selected=grouped_selected,
            grouped_inputs=grouped_inputs,
            cache_size=cache_size,
            device=device,
            config=config,
            count_from_token_idx=count_from_token_idx,
        )
    if policy == "finemoe":
        return _simulate_finemoe_metrics(
            grouped_selected=grouped_selected,
            grouped_inputs=grouped_inputs,
            grouped_scores=grouped_router_scores,
            cache_size=cache_size,
            device=device,
            config=config,
            count_from_token_idx=count_from_token_idx,
        )
    if policy in {"future_prefetch", "future_router_prefetch"}:
        return _simulate_future_prefetch_metrics(
            grouped_selected=grouped_selected,
            grouped_current=grouped_current,
            grouped_future=grouped_future,
            cache_size=cache_size,
            device=device,
            future_cache_priority_mode=config.future_cache_priority_mode,
            count_from_token_idx=count_from_token_idx,
        )
    if policy in {"specmd_prefetch", "specmd"}:
        return _simulate_specmd_prefetch_metrics(
            grouped_selected=grouped_selected,
            grouped_scores=grouped_router_scores,
            cache_size=cache_size,
            device=device,
            config=config,
            count_from_token_idx=count_from_token_idx,
        )

    for selected_chunks in grouped_selected.values():
        if not selected_chunks:
            continue
        selected = torch.cat(selected_chunks, dim=1)[0]
        if selected.size(0) <= 0:
            continue
        num_experts = int(selected.max().item()) + 1 if selected.numel() > 0 else 0
        num_experts = max(num_experts, cache_size)
        cache = torch.empty(0, device=device, dtype=selected.dtype)
        freq = torch.zeros(num_experts, device=device, dtype=torch.float32)
        crf = torch.zeros(num_experts, device=device, dtype=torch.float32)
        lrfu_last_update = torch.full((num_experts,), -1.0, device=device, dtype=torch.float32)

        for token_idx in range(selected.size(0)):
            token_selected = selected[token_idx]
            unique_selected = torch.unique(token_selected.detach())
            hit_count = _contains_any(token_selected, cache).to(dtype=torch.float64).sum()
            should_count = count_from_token_idx is None or token_idx >= count_from_token_idx
            if should_count:
                access_count = torch.tensor(float(token_selected.numel()), device=device, dtype=torch.float64)
                total_overlap = total_overlap + hit_count
                total_hit_rate = total_hit_rate + hit_count / access_count
                total_access = total_access + access_count
                total_miss = total_miss + (access_count - hit_count)
                counted_steps += 1

            if policy == "lru":
                for expert in token_selected:
                    cache = _touch_lru(cache, expert.detach(), cache_size)
                continue

            selected_long = unique_selected.to(dtype=torch.long)
            max_selected = int(selected_long.max().item()) if selected_long.numel() > 0 else -1
            if max_selected >= freq.numel():
                grow = max_selected + 1 - freq.numel()
                freq = torch.cat([freq, torch.zeros(grow, device=device, dtype=freq.dtype)])
                crf = torch.cat([crf, torch.zeros(grow, device=device, dtype=crf.dtype)])
                lrfu_last_update = torch.cat(
                    [
                        lrfu_last_update,
                        torch.full((grow,), -1.0, device=device, dtype=lrfu_last_update.dtype),
                    ]
                )

            freq[selected_long] += 1.0
            if policy == "lrfu":
                selected_score = _score_lrfu(
                    crf=crf[selected_long],
                    last_update=lrfu_last_update[selected_long],
                    step=token_idx,
                    lrfu_lambda=config.lrfu_lambda,
                )
                crf[selected_long] = selected_score + 1.0
                lrfu_last_update[selected_long] = float(token_idx)
            misses = unique_selected[~_contains_any(unique_selected, cache)]
            if misses.numel() > 0:
                cache = torch.cat([cache, misses])
            if policy == "lfu":
                score = _score_lfu(freq)
            elif policy == "lrfu":
                score = _score_lrfu(
                    crf=crf,
                    last_update=lrfu_last_update,
                    step=token_idx,
                    lrfu_lambda=config.lrfu_lambda,
                )
            else:
                raise ValueError(f"Unsupported cache policy baseline: {config.policy!r}")
            cache = _evict_by_score(
                cache=torch.unique(cache),
                score=score,
                cache_size=cache_size,
                protected=unique_selected,
            )

    return {
        "hard_hit_rate_sum": total_hit_rate,
        "hard_overlap_sum": total_overlap,
        "hard_step_count": torch.tensor(float(counted_steps), device=device, dtype=torch.float64),
        "hard_access_count": total_access,
        "hard_miss_count": total_miss,
    }


def simulate_cache_policy_metrics_pair(
    layer_stats: List[Dict[str, torch.Tensor]],
    cache_size: int,
    device: torch.device,
    config: CachePolicyConfig,
    decode_count_from_token_idx: Optional[int],
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    policy = config.policy.lower()
    if policy not in {"specmd_prefetch", "specmd"}:
        return (
            simulate_cache_policy_metrics(
                layer_stats=layer_stats,
                cache_size=cache_size,
                device=device,
                config=config,
            ),
            simulate_cache_policy_metrics(
                layer_stats=layer_stats,
                cache_size=cache_size,
                device=device,
                config=config,
                count_from_token_idx=decode_count_from_token_idx,
            ),
        )

    (
        grouped_selected,
        _grouped_inputs,
        _grouped_current,
        _grouped_future,
        grouped_router_scores,
    ) = _group_cache_policy_layer_stats(layer_stats=layer_stats, device=device)
    full_metrics, decode_metrics = _simulate_specmd_prefetch_metrics_many(
        grouped_selected=grouped_selected,
        grouped_scores=grouped_router_scores,
        cache_size=cache_size,
        device=device,
        config=config,
        count_from_token_indices=[None, decode_count_from_token_idx],
    )
    return full_metrics, decode_metrics
