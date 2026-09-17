from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


FINEMOE_STORE_FORMAT = "finemoe_expert_map_store_v1"
PROMOE_TRACE_FORMATS = {"promoe_trace_v1", "promoe_trace_shard_v1"}


def _normalize_rows(values: torch.Tensor) -> torch.Tensor:
    values = values.to(dtype=torch.float32, device="cpu")
    if values.dim() == 1:
        values = values.unsqueeze(0)
    norms = torch.norm(values, p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    return values / norms


def _normalize_vector(value: torch.Tensor) -> torch.Tensor:
    normalized = _normalize_rows(value.reshape(1, -1))[0]
    return normalized.to(device="cpu", dtype=torch.float32)


def _tensor_bytes(value: Any) -> int:
    if not isinstance(value, torch.Tensor):
        return 0
    return int(value.numel() * value.element_size())


def _entry_tensor_bytes(entry: Dict[str, Any]) -> int:
    return _tensor_bytes(entry.get("semantic")) + _tensor_bytes(entry.get("expert_maps"))


def finemoe_store_storage_bytes(store: Optional[Dict[str, Any]]) -> int:
    if store is None:
        return 0
    total = 0
    for entry in store.get("entries", []):
        if isinstance(entry, dict):
            total += _entry_tensor_bytes(entry)
    return total


def finemoe_store_summary(store: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if store is None:
        return {
            "finemoe_store_entries": 0,
            "finemoe_store_storage_MB": 0.0,
        }
    metadata = store.get("metadata", {}) if isinstance(store, dict) else {}
    entry_count = int(metadata.get("num_entries", len(store.get("entries", []))))
    storage_bytes = int(metadata.get("storage_bytes", finemoe_store_storage_bytes(store)))
    return {
        "finemoe_store_entries": entry_count,
        "finemoe_store_storage_MB": storage_bytes / 1_000_000.0,
        "finemoe_store_num_layers": int(metadata.get("num_layers", len(store.get("layer_ids", [])))),
        "finemoe_store_num_experts": int(metadata.get("num_experts", store.get("num_experts", 0) or 0)),
    }


def load_trace_samples(paths: Sequence[str | Path]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for path in paths:
        payload = torch.load(Path(path), map_location="cpu")
        if isinstance(payload, dict) and payload.get("format") in PROMOE_TRACE_FORMATS:
            samples.extend(payload.get("samples", []))
        elif isinstance(payload, dict) and "samples" in payload:
            samples.extend(payload.get("samples", []))
        elif isinstance(payload, list):
            samples.extend(item for item in payload if isinstance(item, dict))
        else:
            raise ValueError(f"Unsupported trace file format: {path}")
    samples.sort(key=lambda item: int(item.get("index", 0)))
    return samples


def _layer_probability_maps(layer: Dict[str, Any]) -> Optional[torch.Tensor]:
    expert_maps = layer.get("expert_maps")
    if expert_maps is not None:
        maps = expert_maps.detach().to(device="cpu", dtype=torch.float32)
        if maps.dim() == 3:
            maps = maps.reshape(-1, maps.size(-1))
        return maps

    selected = layer.get("selected_experts")
    if selected is None:
        return None
    selected = selected.detach().to(device="cpu", dtype=torch.long)
    if selected.dim() == 3:
        selected = selected.reshape(-1, selected.size(-1))
    if selected.dim() != 2 or selected.size(0) == 0:
        return None

    num_experts = int(layer.get("num_experts", 0) or 0)
    if num_experts <= 0 and selected.numel() > 0:
        num_experts = int(selected.max().item()) + 1
    if num_experts <= 0:
        return None

    maps = torch.zeros((selected.size(0), num_experts), dtype=torch.float32)
    width = max(int(selected.size(-1)), 1)
    maps.scatter_add_(1, selected, torch.full_like(selected, 1.0 / float(width), dtype=torch.float32))
    return maps


def _layer_inputs(layer: Dict[str, Any]) -> Optional[torch.Tensor]:
    inputs = layer.get("promoe_inputs")
    if inputs is None:
        return None
    inputs = inputs.detach().to(device="cpu", dtype=torch.float32)
    if inputs.dim() == 3:
        inputs = inputs.reshape(-1, inputs.size(-1))
    if inputs.dim() != 2 or inputs.size(0) == 0:
        return None
    return inputs


def _dedup_semantic_match(existing_semantics: List[torch.Tensor], semantic: torch.Tensor, threshold: float) -> bool:
    if threshold <= 0.0 or not existing_semantics:
        return False
    existing = torch.stack(existing_semantics, dim=0)
    score = torch.mv(existing, semantic).max().item()
    return float(score) >= float(threshold)


def build_finemoe_store_from_trace_samples(
    samples: Sequence[Dict[str, Any]],
    *,
    max_entries: int = 1000,
    max_tokens_per_sample: Optional[int] = None,
    semantic_layer_idx: Optional[int] = None,
    dedup_similarity_threshold: float = 0.0,
) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    stored_semantics: List[torch.Tensor] = []
    global_layer_ids: List[int] = []
    global_num_experts = 0

    for sample in samples:
        if max_entries > 0 and len(entries) >= max_entries:
            break
        layers = [layer for layer in sample.get("layers", []) if isinstance(layer, dict)]
        if not layers:
            continue
        layers.sort(key=lambda item: int(item.get("layer_idx", 0)))
        layer_ids = [int(layer.get("layer_idx", 0)) for layer in layers]
        if not global_layer_ids:
            global_layer_ids = layer_ids
        else:
            global_layer_ids = sorted(set(global_layer_ids).union(layer_ids))

        layer_maps_by_idx: Dict[int, torch.Tensor] = {}
        layer_inputs_by_idx: Dict[int, torch.Tensor] = {}
        sample_num_experts = 0
        for layer in layers:
            layer_idx = int(layer.get("layer_idx", 0))
            maps = _layer_probability_maps(layer)
            if maps is not None and maps.size(0) > 0:
                layer_maps_by_idx[layer_idx] = maps
                sample_num_experts = max(sample_num_experts, int(maps.size(-1)))
            inputs = _layer_inputs(layer)
            if inputs is not None:
                layer_inputs_by_idx[layer_idx] = inputs
        if not layer_maps_by_idx:
            continue

        source_layer_idx = semantic_layer_idx
        if source_layer_idx is None or source_layer_idx not in layer_inputs_by_idx:
            source_layer_idx = min(layer_inputs_by_idx) if layer_inputs_by_idx else None
        if source_layer_idx is None:
            continue
        source_inputs = layer_inputs_by_idx[source_layer_idx]
        token_count = min(
            [int(source_inputs.size(0))]
            + [int(layer_map.size(0)) for layer_map in layer_maps_by_idx.values()]
        )
        if max_tokens_per_sample is not None and max_tokens_per_sample > 0:
            token_count = min(token_count, int(max_tokens_per_sample))
        if token_count <= 0:
            continue

        sample_layer_ids = sorted(layer_maps_by_idx)
        layer_pos = {layer_idx: pos for pos, layer_idx in enumerate(sample_layer_ids)}
        global_num_experts = max(global_num_experts, sample_num_experts)

        for token_idx in range(token_count):
            if max_entries > 0 and len(entries) >= max_entries:
                break
            semantic = _normalize_vector(source_inputs[token_idx])
            if _dedup_semantic_match(stored_semantics, semantic, dedup_similarity_threshold):
                continue

            expert_maps = torch.zeros((len(sample_layer_ids), sample_num_experts), dtype=torch.float32)
            for layer_idx, layer_map in layer_maps_by_idx.items():
                current = layer_map[token_idx]
                if current.numel() < sample_num_experts:
                    padded = torch.zeros((sample_num_experts,), dtype=torch.float32)
                    padded[: current.numel()] = current
                    current = padded
                expert_maps[layer_pos[layer_idx]] = current.to(dtype=torch.float32)

            entries.append(
                {
                    "sample_index": int(sample.get("index", len(entries))),
                    "token_idx": int(token_idx),
                    "semantic": semantic.to(dtype=torch.float16),
                    "layer_ids": torch.tensor(sample_layer_ids, dtype=torch.int16),
                    "expert_maps": expert_maps.to(dtype=torch.float16),
                }
            )
            stored_semantics.append(semantic)

    storage_bytes = sum(_entry_tensor_bytes(entry) for entry in entries)
    store = {
        "format": FINEMOE_STORE_FORMAT,
        "version": 1,
        "config": {
            "max_entries": int(max_entries),
            "max_tokens_per_sample": max_tokens_per_sample,
            "semantic_layer_idx": semantic_layer_idx,
            "dedup_similarity_threshold": float(dedup_similarity_threshold),
        },
        "layer_ids": global_layer_ids,
        "num_experts": int(global_num_experts),
        "entries": entries,
        "metadata": {
            "num_entries": len(entries),
            "num_layers": len(global_layer_ids),
            "num_experts": int(global_num_experts),
            "storage_bytes": int(storage_bytes),
        },
    }
    return store


def _prepare_finemoe_runtime(store: Dict[str, Any]) -> Dict[str, Any]:
    runtime = store.get("_runtime")
    if isinstance(runtime, dict):
        return runtime

    entries = [entry for entry in store.get("entries", []) if isinstance(entry, dict)]
    if not entries:
        runtime = {
            "semantic": torch.empty((0, 0), dtype=torch.float32),
            "expert_maps": torch.empty((0, 0, 0), dtype=torch.float32),
            "layer_ids": [],
            "layer_pos": {},
        }
        store["_runtime"] = runtime
        return runtime

    layer_ids = sorted(
        {
            int(layer_idx)
            for entry in entries
            for layer_idx in entry.get("layer_ids", torch.empty(0, dtype=torch.int16)).tolist()
        }
    )
    layer_pos = {layer_idx: pos for pos, layer_idx in enumerate(layer_ids)}
    num_experts = max(int(entry.get("expert_maps").size(-1)) for entry in entries if entry.get("expert_maps") is not None)

    semantics: List[torch.Tensor] = []
    maps: List[torch.Tensor] = []
    metadata: List[Tuple[int, int]] = []
    for entry in entries:
        semantic = entry.get("semantic")
        expert_maps = entry.get("expert_maps")
        entry_layer_ids = entry.get("layer_ids")
        if semantic is None or expert_maps is None or entry_layer_ids is None:
            continue
        semantics.append(_normalize_vector(semantic))
        packed = torch.zeros((len(layer_ids), num_experts), dtype=torch.float32)
        entry_layer_ids_list = [int(item) for item in entry_layer_ids.tolist()]
        expert_maps = expert_maps.detach().to(device="cpu", dtype=torch.float32)
        for src_pos, layer_idx in enumerate(entry_layer_ids_list):
            dst_pos = layer_pos.get(layer_idx)
            if dst_pos is None or src_pos >= expert_maps.size(0):
                continue
            current = expert_maps[src_pos]
            packed[dst_pos, : current.numel()] = current
        maps.append(packed)
        metadata.append((int(entry.get("sample_index", -1)), int(entry.get("token_idx", -1))))

    if semantics:
        semantic_tensor = torch.stack(semantics, dim=0)
        map_tensor = torch.stack(maps, dim=0)
    else:
        semantic_tensor = torch.empty((0, 0), dtype=torch.float32)
        map_tensor = torch.empty((0, len(layer_ids), num_experts), dtype=torch.float32)

    runtime = {
        "semantic": semantic_tensor,
        "expert_maps": map_tensor,
        "layer_ids": layer_ids,
        "layer_pos": layer_pos,
        "metadata": metadata,
    }
    store["_runtime"] = runtime
    return runtime


def load_finemoe_store(path: Optional[str | Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != FINEMOE_STORE_FORMAT:
        raise ValueError(f"Invalid FineMoE Expert Map Store file: {path}")
    _prepare_finemoe_runtime(payload)
    return payload


def finemoe_semantic_candidates(
    store: Dict[str, Any],
    query_semantic: torch.Tensor,
    candidate_pool: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    runtime = _prepare_finemoe_runtime(store)
    semantic_matrix = runtime["semantic"]
    if semantic_matrix.numel() == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32)
    query = _normalize_vector(query_semantic)
    if query.numel() != semantic_matrix.size(-1):
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32)
    scores = torch.mv(semantic_matrix, query)
    k = min(max(int(candidate_pool), 1), int(scores.numel()))
    values, indices = torch.topk(scores, k=k, largest=True, sorted=True)
    return indices.to(dtype=torch.long), values.to(dtype=torch.float32)


def finemoe_predict_expert_scores(
    store: Dict[str, Any],
    semantic_candidates: Tuple[torch.Tensor, torch.Tensor],
    target_layer_idx: int,
    observed_maps: Dict[int, torch.Tensor],
    *,
    top_k: int,
    semantic_weight: float,
    trajectory_weight: float,
) -> Tuple[Optional[torch.Tensor], float]:
    runtime = _prepare_finemoe_runtime(store)
    layer_pos = runtime["layer_pos"].get(int(target_layer_idx))
    if layer_pos is None:
        return None, 0.0

    candidate_indices, semantic_scores = semantic_candidates
    if candidate_indices.numel() <= 0:
        return None, 0.0

    expert_maps = runtime["expert_maps"]
    candidate_maps = expert_maps[candidate_indices]
    combined_scores = float(semantic_weight) * semantic_scores.to(dtype=torch.float32)

    observed_layers = [layer_idx for layer_idx in sorted(observed_maps) if layer_idx in runtime["layer_pos"]]
    if observed_layers and float(trajectory_weight) != 0.0:
        observed_parts: List[torch.Tensor] = []
        candidate_parts: List[torch.Tensor] = []
        for layer_idx in observed_layers:
            pos = runtime["layer_pos"][layer_idx]
            observed = observed_maps[layer_idx].detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            num_experts = candidate_maps.size(-1)
            if observed.numel() < num_experts:
                padded = torch.zeros((num_experts,), dtype=torch.float32)
                padded[: observed.numel()] = observed
                observed = padded
            observed_parts.append(observed[:num_experts])
            candidate_parts.append(candidate_maps[:, pos, :])
        observed_query = _normalize_vector(torch.cat(observed_parts, dim=0))
        candidate_matrix = _normalize_rows(torch.cat(candidate_parts, dim=1))
        trajectory_scores = torch.mv(candidate_matrix, observed_query)
        combined_scores = combined_scores + float(trajectory_weight) * trajectory_scores

    k = min(max(int(top_k), 1), int(candidate_indices.numel()))
    top_values, top_positions = torch.topk(combined_scores, k=k, largest=True, sorted=True)
    selected_maps = candidate_maps[top_positions, layer_pos, :].to(dtype=torch.float32)
    weights = torch.softmax(top_values.to(dtype=torch.float32), dim=0)
    expert_scores = torch.sum(selected_maps * weights.unsqueeze(-1), dim=0)

    score_scale = max(abs(float(semantic_weight)) + abs(float(trajectory_weight)), 1e-6)
    confidence = float(torch.clamp((top_values[0] / score_scale + 1.0) * 0.5, 0.0, 1.0).item())
    return expert_scores, confidence


def finemoe_dynamic_threshold(
    confidence: float,
    *,
    fixed_threshold: float,
    min_threshold: float,
    max_threshold: float,
) -> float:
    if fixed_threshold >= 0.0:
        return float(fixed_threshold)
    confidence = max(0.0, min(1.0, float(confidence)))
    low = float(min_threshold)
    high = max(low, float(max_threshold))
    return high - confidence * (high - low)


def iter_finemoe_store_trace_paths(paths: Iterable[str | Path]) -> List[Path]:
    return [Path(path) for path in paths]
