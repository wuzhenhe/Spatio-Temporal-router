from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedModel, Trainer


def _unwrap_cache_model(model):
    candidates = [model]
    visited = set()
    while candidates:
        candidate = candidates.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if (
            hasattr(candidate, "compute_cache_moe_aux")
            or hasattr(candidate, "reset_cache_moe_state")
            or hasattr(candidate, "compute_window_cache_aux")
            or hasattr(candidate, "reset_window_cache_state")
            or hasattr(candidate, "compute_lru_cache_metrics")
            or hasattr(candidate, "reset_lru_cache_metric_state")
            or hasattr(candidate, "compute_temporal_option_moe_aux")
            or hasattr(candidate, "reset_temporal_option_moe_state")
        ):
            return candidate
        for attr in ("module", "_orig_mod", "base_model", "model"):
            nested = getattr(candidate, attr, None)
            if nested is not None and id(nested) not in visited:
                candidates.append(nested)
    return model


class MoESFTTrainer(Trainer):
    def _save(self, output_dir=None, state_dict=None):
        """Keep fused MoE keys compatible with Trainer's state-dict loader.

        Transformers 5 otherwise exports Qwen experts in their original unfused
        format, which _load_from_checkpoint does not convert back.
        FSDP SHARDED_STATE_DICT saving remains handled by Accelerate.
        """
        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        if not isinstance(model, PreTrainedModel):
            return super()._save(output_dir, state_dict)
        destination = Path(output_dir or self.args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(destination, state_dict=state_dict, save_original_format=False)
        processor = self.processing_class or getattr(self.data_collator, "tokenizer", None)
        if processor is not None:
            processor.save_pretrained(destination)
        torch.save(self.args, destination / "training_args.bin")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._latest_train_cache_metrics: Dict[str, float] = {}
        self._eval_cache_metric_sums: Dict[str, float] = {}
        self._eval_cache_metric_weight = 0.0
        self._future_gate_only_split_clip_max_norm = 0.0
        self._future_gate_only_split_clip_enabled = False
        cache_model = _unwrap_cache_model(self.model)
        cache_cfg = getattr(cache_model, "_cache_moe_config", None)
        if (
            cache_cfg is not None
            and getattr(cache_cfg, "swap_loss_future_gate_only", False)
            and getattr(cache_cfg, "split_grad_clip", False)
        ):
            max_grad_norm = float(getattr(self.args, "max_grad_norm", 0.0) or 0.0)
            if max_grad_norm > 0.0:
                self._future_gate_only_split_clip_max_norm = max_grad_norm
                self._future_gate_only_split_clip_enabled = True
                # Disable Trainer's global clipping. It would couple future_gate gradients
                # back into backbone/router updates through the shared global norm.
                self.args.max_grad_norm = 0.0

    @staticmethod
    def _to_float(value: Optional[torch.Tensor]) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().float().mean().cpu().item())
        return float(value)

    @staticmethod
    def _infer_batch_size(inputs: Dict[str, Any]) -> int:
        for key in ("input_ids", "labels", "attention_mask"):
            value = inputs.get(key)
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                return int(value.shape[0])
        return 1

    def _distributed_mean_tensor(self, value: torch.Tensor) -> torch.Tensor:
        value = value.detach().float()
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return value
        reduced = value.clone()
        torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.SUM)
        reduced = reduced / torch.distributed.get_world_size()
        return reduced

    def _is_world_process_zero(self) -> bool:
        if hasattr(self, "is_world_process_zero"):
            return bool(self.is_world_process_zero())
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    @staticmethod
    def _to_float_list(value: Optional[torch.Tensor]) -> Optional[List[float]]:
        if value is None or not isinstance(value, torch.Tensor) or value.numel() == 0:
            return None
        return [float(item) for item in value.detach().float().cpu().tolist()]

    @staticmethod
    def _summary(values: List[float]) -> Dict[str, float]:
        if not values:
            return {}
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        return {
            "mean": mean_value,
            "min": min(values),
            "max": max(values),
            "std": variance ** 0.5,
        }

    @staticmethod
    def _clip_weights(weights: List[float], min_weight: float, max_weight: float) -> List[float]:
        return [max(min_weight, min(max_weight, float(value))) for value in weights]

    def _append_layer_adaptive_history(self, cache_cfg, record: Dict[str, Any]) -> None:
        if not getattr(cache_cfg, "layer_adaptive_history", True):
            return
        if not self._is_world_process_zero():
            return
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "layer_adaptive_history.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        latest_path = output_dir / "layer_adaptive_state_latest.json"
        with latest_path.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)

    def _append_layer_adaptive_compact_log(self, logs: Dict[str, float]) -> None:
        if not self._is_world_process_zero():
            return
        cache_model = _unwrap_cache_model(self.model)
        cache_cfg = getattr(cache_model, "_cache_moe_config", None)
        if cache_cfg is None or not getattr(cache_cfg, "layer_adaptive_swap_weight", False):
            return

        interesting_keys = {
            "loss",
            "eval_loss",
            "grad_norm",
            "learning_rate",
            "moe_native_loss",
            "eval_moe_native_loss",
            "total_loss",
            "eval_total_loss",
            "swap_loss",
            "eval_swap_loss",
            "weighted_swap_loss",
            "eval_weighted_swap_loss",
            "cache_hit_rate",
            "eval_cache_hit_rate",
            "hybrid_cache_hit_rate",
            "eval_hybrid_cache_hit_rate",
            "temporal_cache_hit_rate",
            "eval_temporal_cache_hit_rate",
            "current_cache_hit_rate",
            "eval_current_cache_hit_rate",
            "spatio_cache_hit_rate",
            "eval_spatio_cache_hit_rate",
            "cache_retention_rate",
            "eval_cache_retention_rate",
            "future_entropy",
            "eval_future_entropy",
            "spatio_entropy",
            "eval_spatio_entropy",
            "hard_cache_hit_rate",
            "eval_hard_cache_hit_rate",
            "hard_cache_miss_rate",
            "eval_hard_cache_miss_rate",
            "hard_cache_overlap_count",
            "eval_hard_cache_overlap_count",
            "per_layer_swap_loss_min",
            "per_layer_swap_loss_max",
            "per_layer_swap_loss_std",
            "eval_per_layer_swap_loss_min",
            "eval_per_layer_swap_loss_max",
            "eval_per_layer_swap_loss_std",
            "per_layer_cache_hit_rate_min",
            "per_layer_cache_hit_rate_max",
            "per_layer_cache_hit_rate_std",
            "eval_per_layer_cache_hit_rate_min",
            "eval_per_layer_cache_hit_rate_max",
            "eval_per_layer_cache_hit_rate_std",
            "per_layer_future_entropy_min",
            "per_layer_future_entropy_max",
            "per_layer_future_entropy_std",
            "eval_per_layer_future_entropy_min",
            "eval_per_layer_future_entropy_max",
            "eval_per_layer_future_entropy_std",
            "layer_adaptive_update_count",
            "layer_adaptive_last_did_update",
            "layer_adaptive_weight_mean",
            "layer_adaptive_weight_min",
            "layer_adaptive_weight_max",
            "layer_adaptive_weight_std",
            "layer_adaptive_score_mean",
            "layer_adaptive_score_min",
            "layer_adaptive_score_max",
            "layer_adaptive_score_std",
            "layer_adaptive_cosine_valid_count",
            "split_clip_base_grad_norm",
            "split_clip_future_gate_grad_norm",
            "split_clip_base_clip_coef",
            "split_clip_future_gate_clip_coef",
            "temporal_option_loss",
            "weighted_temporal_option_loss",
            "option_selection_loss",
            "option_termination_loss",
            "option_deliberation_loss",
            "option_entropy",
            "option_switch_rate",
            "option_target_switch_rate",
            "option_teacher_coverage",
            "option_router_overlap",
            "option_size",
        }
        record: Dict[str, Any] = {
            "step": int(getattr(self.state, "global_step", 0)),
            "epoch": None if self.state.epoch is None else float(self.state.epoch),
            "is_eval": any(key.startswith("eval_") for key in logs),
        }
        for key in sorted(interesting_keys):
            if key not in logs:
                continue
            value = logs[key]
            try:
                record[key] = float(value)
            except (TypeError, ValueError):
                record[key] = value

        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "layer_adaptive_compact_log.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _layer_adaptive_should_update(self, cache_cfg, training: bool) -> bool:
        if not training or not getattr(cache_cfg, "layer_adaptive_swap_weight", False):
            return False
        update_count = int(getattr(cache_cfg, "_layer_adaptive_update_count", 0))
        warmup_steps = int(getattr(cache_cfg, "layer_adaptive_warmup_steps", 0))
        update_interval = max(1, int(getattr(cache_cfg, "layer_adaptive_update_interval", 1)))
        return update_count >= warmup_steps and (update_count - warmup_steps) % update_interval == 0

    def _consume_pending_layer_adaptive_cosine(self, cache_cfg, layer_count: int) -> Optional[List[float]]:
        pending = getattr(cache_cfg, "_layer_adaptive_pending_cosine", None)
        if not pending:
            return None
        device = self.args.device if self.args is not None else torch.device("cpu")
        values = [float(pending.get(layer_idx, 0.0)) for layer_idx in range(layer_count)]
        valid_count = sum(1 for layer_idx in range(layer_count) if layer_idx in pending)
        tensor = torch.tensor(values, device=device, dtype=torch.float32)
        tensor = self._distributed_mean_tensor(tensor)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            count_tensor = torch.tensor(float(valid_count), device=device, dtype=torch.float32)
            torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.SUM)
            valid_count = int(round(count_tensor.item() / torch.distributed.get_world_size()))
        cache_cfg._layer_adaptive_pending_cosine = {}
        cache_cfg._layer_adaptive_grad_probe = {}
        cache_cfg._layer_adaptive_cosine_valid_count = valid_count
        return self._to_float_list(tensor)

    def _update_layer_adaptive_weights(
        self,
        cache_cfg,
        cache_aux: Dict[str, torch.Tensor],
        training: bool,
        cosine_values: Optional[List[float]] = None,
    ) -> None:
        if not training or not getattr(cache_cfg, "layer_adaptive_swap_weight", False):
            return
        raw_swap = cache_aux.get("per_layer_swap_loss")
        raw_hit = cache_aux.get("per_layer_cache_hit_rate")
        if raw_swap is None:
            return
        per_layer_swap = self._to_float_list(self._distributed_mean_tensor(raw_swap))
        per_layer_hit = self._to_float_list(self._distributed_mean_tensor(raw_hit)) if raw_hit is not None else None
        if per_layer_swap is None:
            return

        layer_count = len(per_layer_swap)
        if not hasattr(cache_cfg, "_layer_adaptive_weights") or len(cache_cfg._layer_adaptive_weights) != layer_count:
            cache_cfg._layer_adaptive_weights = [1.0] * layer_count
            cache_cfg._layer_adaptive_update_count = 0
            cache_cfg._layer_adaptive_cosine_ema = None
            cache_cfg._layer_adaptive_last_cosine = None
            cache_cfg._layer_adaptive_last_score = None

        update_count = int(getattr(cache_cfg, "_layer_adaptive_update_count", 0))
        cosine_ema = getattr(cache_cfg, "_layer_adaptive_cosine_ema", None)
        if cosine_values is not None and len(cosine_values) == layer_count:
            cache_cfg._layer_adaptive_last_cosine = [float(value) for value in cosine_values]
            cosine_alpha = float(getattr(cache_cfg, "layer_adaptive_cosine_ema", 0.95))
            if cosine_ema is None or len(cosine_ema) != layer_count:
                cosine_ema = list(cosine_values)
            else:
                cosine_ema = [
                    cosine_alpha * old + (1.0 - cosine_alpha) * current
                    for old, current in zip(cosine_ema, cosine_values)
                ]

        if cosine_ema is None or len(cosine_ema) != layer_count:
            score = [0.0 for _ in range(layer_count)]
        else:
            cosine_scale = float(getattr(cache_cfg, "layer_adaptive_cosine_scale", 1.0))
            cosine_deadband = max(0.0, float(getattr(cache_cfg, "layer_adaptive_cosine_deadband", 0.0)))
            score = [
                0.0 if abs(float(value)) < cosine_deadband else float(value) * cosine_scale
                for value in cosine_ema
            ]
        should_update = cosine_values is not None and len(cosine_values) == layer_count
        weights = list(cache_cfg._layer_adaptive_weights)
        if should_update:
            lr = float(cache_cfg.layer_adaptive_lr)
            weights = [weight * math.exp(lr * value) for weight, value in zip(weights, score)]
            weights = self._clip_weights(
                weights,
                min_weight=float(cache_cfg.layer_adaptive_weight_min),
                max_weight=float(cache_cfg.layer_adaptive_weight_max),
            )

        cache_cfg._layer_adaptive_weights = [float(value) for value in weights]
        cache_cfg._layer_adaptive_cosine_ema = (
            None if cosine_ema is None else [float(value) for value in cosine_ema]
        )
        cache_cfg._layer_adaptive_last_score = [float(value) for value in score]
        cache_cfg._layer_adaptive_last_hit = None if per_layer_hit is None else [float(value) for value in per_layer_hit]
        cache_cfg._layer_adaptive_last_did_update = bool(should_update)
        cache_cfg._layer_adaptive_update_count = update_count + 1

        if should_update:
            record = {
                "step": int(getattr(self.state, "global_step", 0)),
                "epoch": None if self.state.epoch is None else float(self.state.epoch),
                "update_count": update_count,
                "did_update": True,
                "weights": cache_cfg._layer_adaptive_weights,
                "score": cache_cfg._layer_adaptive_last_score,
                "cosine": getattr(cache_cfg, "_layer_adaptive_last_cosine", None),
                "cosine_ema": cache_cfg._layer_adaptive_cosine_ema,
                "per_layer_swap_loss": per_layer_swap,
                "per_layer_cache_hit_rate": per_layer_hit,
            }
            self._append_layer_adaptive_history(cache_cfg, record)

    def _layer_adaptive_metric_dict(self, cache_cfg) -> Dict[str, float]:
        if not getattr(cache_cfg, "layer_adaptive_swap_weight", False):
            return {}
        weights = [float(value) for value in getattr(cache_cfg, "_layer_adaptive_weights", [])]
        score = [float(value) for value in getattr(cache_cfg, "_layer_adaptive_last_score", []) or []]
        metrics: Dict[str, float] = {
            "layer_adaptive_update_count": float(getattr(cache_cfg, "_layer_adaptive_update_count", 0)),
            "layer_adaptive_last_did_update": float(bool(getattr(cache_cfg, "_layer_adaptive_last_did_update", False))),
            "layer_adaptive_cosine_valid_count": float(getattr(cache_cfg, "_layer_adaptive_cosine_valid_count", 0)),
        }
        summary_items = [
            ("layer_adaptive_weight", weights),
            ("layer_adaptive_score", score),
        ]
        for prefix, values in summary_items:
            for key, value in self._summary(values).items():
                metrics[f"{prefix}_{key}"] = float(value)
        return metrics

    @classmethod
    def _build_cache_metric_dict(
        cls,
        moe_native_loss: torch.Tensor,
        total_loss: torch.Tensor,
        balance_loss: Optional[torch.Tensor],
        swap_loss: torch.Tensor,
        cache_aux: Dict[str, torch.Tensor],
        balance_weight: float,
        swap_weight: float,
        include_hard_metrics: bool,
    ) -> Dict[str, float]:
        weighted_balance_loss = balance_weight * balance_loss
        weighted_swap_loss = swap_weight * swap_loss
        metrics = {
            "moe_native_loss": cls._to_float(moe_native_loss),
            "total_loss": cls._to_float(total_loss),
            "swap_loss": cls._to_float(swap_loss),
            "weighted_swap_loss": cls._to_float(weighted_swap_loss),
            "cache_hit_rate": cls._to_float(cache_aux["cache_hit_rate"]),
            "cache_retention_rate": cls._to_float(cache_aux["cache_retention_rate"]),
            "future_entropy": cls._to_float(cache_aux["future_entropy"]),
        }
        for metric_name in (
            "hybrid_cache_hit_rate",
            "temporal_cache_hit_rate",
            "current_cache_hit_rate",
            "spatio_cache_hit_rate",
            "spatio_entropy",
        ):
            value = cache_aux.get(metric_name)
            if value is not None:
                metrics[metric_name] = cls._to_float(value)
        if balance_weight != 0.0:
            metrics["balance_loss"] = cls._to_float(balance_loss)
            metrics["weighted_balance_loss"] = cls._to_float(weighted_balance_loss)
        per_layer_swap_weight = cache_aux.get("per_layer_swap_weight")
        if isinstance(per_layer_swap_weight, torch.Tensor) and per_layer_swap_weight.numel() > 0:
            detached = per_layer_swap_weight.detach().float()
            metrics.update(
                {
                    "layer_swap_weight_min": cls._to_float(detached.min()),
                    "layer_swap_weight_max": cls._to_float(detached.max()),
                    "layer_swap_weight_std": cls._to_float(detached.std(unbiased=False)),
                }
            )
        for metric_name, tensor_name in (
            ("per_layer_swap_loss", "per_layer_swap_loss"),
            ("per_layer_cache_hit_rate", "per_layer_cache_hit_rate"),
            ("per_layer_future_entropy", "per_layer_future_entropy"),
        ):
            value = cache_aux.get(tensor_name)
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                detached = value.detach().float()
                metrics[f"{metric_name}_min"] = cls._to_float(detached.min())
                metrics[f"{metric_name}_max"] = cls._to_float(detached.max())
                metrics[f"{metric_name}_std"] = cls._to_float(detached.std(unbiased=False))
        if include_hard_metrics:
            metrics.update(
                {
                    "hard_cache_hit_rate": cls._to_float(cache_aux["hard_cache_hit_rate"]),
                    "hard_cache_miss_rate": cls._to_float(cache_aux["hard_cache_miss_rate"]),
                    "hard_cache_overlap_count": cls._to_float(cache_aux["hard_cache_overlap_count"]),
                }
            )
        return {key: value for key, value in metrics.items() if value is not None}

    @classmethod
    def _build_lru_metric_dict(
        cls,
        moe_native_loss: torch.Tensor,
        lru_metrics: Dict[str, torch.Tensor],
        include_hard_metrics: bool,
    ) -> Dict[str, float]:
        metrics = {
            "moe_native_loss": cls._to_float(moe_native_loss),
            "total_loss": cls._to_float(moe_native_loss),
        }
        if include_hard_metrics:
            metrics.update(
                {
                    "hard_cache_hit_rate": cls._to_float(lru_metrics["hard_cache_hit_rate"]),
                    "hard_cache_miss_rate": cls._to_float(lru_metrics["hard_cache_miss_rate"]),
                    "hard_cache_overlap_count": cls._to_float(lru_metrics["hard_cache_overlap_count"]),
                }
            )
        return {key: value for key, value in metrics.items() if value is not None}

    @classmethod
    def _build_window_cache_metric_dict(
        cls,
        moe_native_loss: torch.Tensor,
        total_loss: torch.Tensor,
        window_metrics: Dict[str, torch.Tensor],
        avg_swap_coeff: float,
        peak_swap_coeff: float,
        mtp_swap_coeff: float,
        include_hard_metrics: bool,
    ) -> Dict[str, float]:
        avg_swap_loss = window_metrics["avg_swap_loss"]
        peak_swap_loss = window_metrics["peak_swap_loss"]
        mtp_swap_loss = window_metrics["mtp_swap_loss"]
        metrics = {
            "moe_native_loss": cls._to_float(moe_native_loss),
            "total_loss": cls._to_float(total_loss),
            "cache_avg_swap_metric": cls._to_float(window_metrics["avg_swap_metric"]),
            "cache_max_swap_metric": cls._to_float(window_metrics["peak_swap_metric"]),
            "cache_avg_swap_loss": cls._to_float(avg_swap_loss),
            "cache_peak_swap_loss": cls._to_float(peak_swap_loss),
            "cache_mtp_swap_loss": cls._to_float(mtp_swap_loss),
            "weighted_cache_avg_swap_loss": cls._to_float(avg_swap_coeff * avg_swap_loss),
            "weighted_cache_peak_swap_loss": cls._to_float(peak_swap_coeff * peak_swap_loss),
            "weighted_cache_mtp_swap_loss": cls._to_float(mtp_swap_coeff * mtp_swap_loss),
            "cache_cache_utilization": cls._to_float(window_metrics["cache_utilization"]),
            "soft_cache_hit_rate": cls._to_float(window_metrics["soft_cache_hit_rate"]),
            "soft_cache_miss_rate": cls._to_float(window_metrics["soft_cache_miss_rate"]),
            "avg_expected_active_experts": cls._to_float(window_metrics["avg_expected_active_experts"]),
        }
        if include_hard_metrics:
            metrics.update(
                {
                    "hard_cache_hit_rate": cls._to_float(window_metrics["hard_cache_hit_rate"]),
                    "hard_cache_miss_rate": cls._to_float(window_metrics["hard_cache_miss_rate"]),
                    "hard_cache_overlap_count": cls._to_float(window_metrics["hard_cache_overlap_count"]),
                    "hard_cache_swap_count": cls._to_float(window_metrics["hard_cache_swap_count"]),
                }
            )
        return {key: value for key, value in metrics.items() if value is not None}

    @classmethod
    def _build_temporal_option_metric_dict(
        cls,
        moe_native_loss: torch.Tensor,
        total_loss: torch.Tensor,
        option_metrics: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        temporal_option_loss = option_metrics["temporal_option_loss"]
        metrics = {
            "moe_native_loss": cls._to_float(moe_native_loss),
            "total_loss": cls._to_float(total_loss),
            "temporal_option_loss": cls._to_float(temporal_option_loss),
            "weighted_temporal_option_loss": cls._to_float(temporal_option_loss),
            "option_selection_loss": cls._to_float(option_metrics["option_selection_loss"]),
            "option_termination_loss": cls._to_float(option_metrics["option_termination_loss"]),
            "option_deliberation_loss": cls._to_float(option_metrics["option_deliberation_loss"]),
            "option_entropy": cls._to_float(option_metrics["option_entropy"]),
            "option_switch_rate": cls._to_float(option_metrics["option_switch_rate"]),
            "option_target_switch_rate": cls._to_float(option_metrics["option_target_switch_rate"]),
            "option_teacher_coverage": cls._to_float(option_metrics["option_teacher_coverage"]),
            "option_router_overlap": cls._to_float(option_metrics["option_router_overlap"]),
            "option_size": cls._to_float(option_metrics["option_size"]),
        }
        return {key: value for key, value in metrics.items() if value is not None}

    def _distributed_sum(self, value: float) -> float:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return value
        device = self.args.device if self.args is not None else torch.device("cpu")
        tensor = torch.tensor(value, device=device, dtype=torch.float64)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return float(tensor.cpu().item())

    def _accumulate_eval_cache_metrics(self, metrics: Dict[str, float], batch_size: int) -> None:
        if not metrics:
            return
        weight = float(max(batch_size, 1))
        for key, value in metrics.items():
            self._eval_cache_metric_sums[key] = self._eval_cache_metric_sums.get(key, 0.0) + value * weight
        self._eval_cache_metric_weight += weight

    def _get_eval_cache_metrics(self, prefix: str) -> Dict[str, float]:
        if self._eval_cache_metric_weight == 0:
            return {}
        total_weight = self._distributed_sum(self._eval_cache_metric_weight)
        if total_weight == 0:
            return {}
        return {
            f"{prefix}_{key}": self._distributed_sum(value) / total_weight
            for key, value in self._eval_cache_metric_sums.items()
        }

    def _reset_eval_cache_metrics(self) -> None:
        self._eval_cache_metric_sums = {}
        self._eval_cache_metric_weight = 0.0

    def _distributed_grad_norm(self, parameters: List[torch.nn.Parameter], norm_type: float = 2.0) -> torch.Tensor:
        grads = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
        device = self.args.device if self.args is not None else torch.device("cpu")
        if not grads:
            return torch.zeros((), device=device, dtype=torch.float32)
        if norm_type == float("inf"):
            local_norm = torch.stack([grad.detach().abs().max().float() for grad in grads]).max()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(local_norm, op=torch.distributed.ReduceOp.MAX)
            return local_norm
        norm_type = float(norm_type)
        local_sum = torch.zeros((), device=grads[0].device, dtype=torch.float32)
        for grad in grads:
            local_sum = local_sum + grad.detach().float().norm(norm_type).pow(norm_type)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(local_sum, op=torch.distributed.ReduceOp.SUM)
        return local_sum.pow(1.0 / norm_type)

    @staticmethod
    def _apply_clip_coef(parameters: List[torch.nn.Parameter], clip_coef: torch.Tensor) -> None:
        coef = clip_coef.to(dtype=torch.float32)
        if coef.item() >= 1.0:
            return
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.detach().mul_(coef.to(device=parameter.grad.device, dtype=parameter.grad.dtype))

    def _future_gate_only_parameter_groups(self, model) -> tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
        future_gate_params: List[torch.nn.Parameter] = []
        base_params: List[torch.nn.Parameter] = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if "future_gate" in name or "spatio_gate" in name:
                future_gate_params.append(parameter)
            else:
                base_params.append(parameter)
        return base_params, future_gate_params

    def _clip_future_gate_only_gradients(self, model) -> None:
        max_norm = float(self._future_gate_only_split_clip_max_norm)
        if max_norm <= 0.0:
            return
        if getattr(self, "do_grad_scaling", False):
            self.scaler.unscale_(self.optimizer)
        elif hasattr(self, "accelerator") and hasattr(self.accelerator, "unscale_gradients"):
            self.accelerator.unscale_gradients()

        base_params, future_gate_params = self._future_gate_only_parameter_groups(model)
        base_norm = self._distributed_grad_norm(base_params)
        future_gate_norm = self._distributed_grad_norm(future_gate_params)
        max_norm_tensor = torch.tensor(max_norm, device=base_norm.device, dtype=torch.float32)
        eps = torch.tensor(1e-6, device=base_norm.device, dtype=torch.float32)
        base_coef = torch.clamp(max_norm_tensor / (base_norm + eps), max=1.0)
        future_gate_coef = torch.clamp(max_norm_tensor / (future_gate_norm.to(base_norm.device) + eps), max=1.0)
        self._apply_clip_coef(base_params, base_coef)
        self._apply_clip_coef(future_gate_params, future_gate_coef.to(future_gate_norm.device))
        self._latest_train_cache_metrics.update(
            {
                "split_clip_base_grad_norm": self._to_float(base_norm),
                "split_clip_future_gate_grad_norm": self._to_float(future_gate_norm),
                "split_clip_base_clip_coef": self._to_float(base_coef),
                "split_clip_future_gate_clip_coef": self._to_float(future_gate_coef),
            }
        )

    def training_step(self, model, inputs: Dict[str, Any], *args, **kwargs) -> torch.Tensor:
        loss = super().training_step(model, inputs, *args, **kwargs)
        accelerator = getattr(self, "accelerator", None)
        if self._future_gate_only_split_clip_enabled and getattr(accelerator, "sync_gradients", True):
            self._clip_future_gate_only_gradients(model)
        return loss

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        cache_model = _unwrap_cache_model(model)
        cache_cfg = getattr(cache_model, "_cache_moe_config", None)
        if cache_cfg is not None:
            cache_cfg._layer_adaptive_capture_cosine = (
                self._layer_adaptive_should_update(cache_cfg, bool(model.training))
            )
        if hasattr(cache_model, "reset_cache_moe_state"):
            cache_model.reset_cache_moe_state()
        if hasattr(cache_model, "reset_window_cache_state"):
            cache_model.reset_window_cache_state()
        if hasattr(cache_model, "reset_lru_cache_metric_state"):
            cache_model.reset_lru_cache_metric_state()
        if hasattr(cache_model, "reset_temporal_option_moe_state"):
            cache_model.reset_temporal_option_moe_state()

        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        moe_native_loss = loss

        balance_loss = None
        if isinstance(outputs, dict):
            balance_loss = outputs.get("aux_loss")
        else:
            balance_loss = getattr(outputs, "aux_loss", None)

        compute_hard_metrics = not model.training

        if hasattr(cache_model, "compute_temporal_option_moe_aux"):
            option_metrics = cache_model.compute_temporal_option_moe_aux(
                inputs.get("attention_mask"),
                compute_hard_metrics=compute_hard_metrics,
            )
            temporal_option_loss = option_metrics["temporal_option_loss"]
            total_loss = loss + temporal_option_loss
            detached_metrics = self._build_temporal_option_metric_dict(
                moe_native_loss=moe_native_loss,
                total_loss=total_loss,
                option_metrics=option_metrics,
            )
            if model.training:
                self._latest_train_cache_metrics = detached_metrics
            else:
                self._accumulate_eval_cache_metrics(detached_metrics, self._infer_batch_size(inputs))

            if isinstance(outputs, dict):
                outputs["moe_native_loss"] = moe_native_loss.detach()
                outputs["total_loss"] = total_loss.detach()
                for metric_name, metric_value in option_metrics.items():
                    outputs[metric_name] = metric_value.detach()
            else:
                setattr(outputs, "moe_native_loss", moe_native_loss.detach())
                setattr(outputs, "total_loss", total_loss.detach())
                for metric_name, metric_value in option_metrics.items():
                    setattr(outputs, metric_name, metric_value.detach())
            loss = total_loss
        elif hasattr(cache_model, "compute_cache_moe_aux"):
            cache_aux = cache_model.compute_cache_moe_aux(
                inputs.get("attention_mask"),
                compute_hard_metrics=compute_hard_metrics,
            )
            swap_loss = cache_aux["swap_loss"]
            cache_cfg = cache_model._cache_moe_config
            effective_swap_weight = float(cache_cfg.swap_weight)
            cosine_values = None
            per_layer_swap = cache_aux.get("per_layer_swap_loss")
            if isinstance(per_layer_swap, torch.Tensor):
                cosine_values = self._consume_pending_layer_adaptive_cosine(
                    cache_cfg,
                    int(per_layer_swap.numel()),
                )
            self._update_layer_adaptive_weights(
                cache_cfg,
                cache_aux,
                training=bool(model.training),
                cosine_values=cosine_values,
            )
            if balance_loss is None:
                balance_loss = torch.zeros_like(swap_loss)
            total_loss = loss + cache_cfg.balance_weight * balance_loss + effective_swap_weight * swap_loss
            detached_metrics = self._build_cache_metric_dict(
                moe_native_loss=moe_native_loss,
                total_loss=total_loss,
                balance_loss=balance_loss,
                swap_loss=swap_loss,
                cache_aux=cache_aux,
                balance_weight=cache_cfg.balance_weight,
                swap_weight=effective_swap_weight,
                include_hard_metrics=compute_hard_metrics,
            )
            if getattr(cache_cfg, "layer_adaptive_swap_weight", False):
                for key in (
                    "layer_swap_weight_min",
                    "layer_swap_weight_max",
                    "layer_swap_weight_std",
                ):
                    detached_metrics.pop(key, None)
            detached_metrics.update(self._layer_adaptive_metric_dict(cache_cfg))
            if model.training:
                self._latest_train_cache_metrics = detached_metrics
            else:
                self._accumulate_eval_cache_metrics(detached_metrics, self._infer_batch_size(inputs))

            if isinstance(outputs, dict):
                outputs["moe_native_loss"] = moe_native_loss.detach()
                outputs["total_loss"] = total_loss.detach()
                outputs["swap_loss"] = swap_loss.detach()
                outputs["weighted_swap_loss"] = (effective_swap_weight * swap_loss).detach()
                outputs["cache_hit_rate"] = cache_aux["cache_hit_rate"].detach()
                outputs["cache_retention_rate"] = cache_aux["cache_retention_rate"].detach()
                outputs["layer_swap_weight"] = cache_aux["layer_swap_weight"].detach()
                outputs["effective_swap_weight"] = torch.tensor(
                    effective_swap_weight,
                    device=swap_loss.device,
                    dtype=swap_loss.dtype,
                )
                if compute_hard_metrics:
                    outputs["hard_cache_hit_rate"] = cache_aux["hard_cache_hit_rate"].detach()
                    outputs["hard_cache_miss_rate"] = cache_aux["hard_cache_miss_rate"].detach()
                    outputs["hard_cache_overlap_count"] = cache_aux["hard_cache_overlap_count"].detach()
                outputs["future_entropy"] = cache_aux["future_entropy"].detach()
                for metric_name in (
                    "hybrid_cache_hit_rate",
                    "temporal_cache_hit_rate",
                    "current_cache_hit_rate",
                    "spatio_cache_hit_rate",
                    "spatio_entropy",
                ):
                    if metric_name in cache_aux:
                        outputs[metric_name] = cache_aux[metric_name].detach()
                outputs["balance_loss"] = balance_loss.detach()
                outputs["weighted_balance_loss"] = (cache_cfg.balance_weight * balance_loss).detach()
            else:
                setattr(outputs, "moe_native_loss", moe_native_loss.detach())
                setattr(outputs, "total_loss", total_loss.detach())
                setattr(outputs, "swap_loss", swap_loss.detach())
                setattr(outputs, "weighted_swap_loss", (effective_swap_weight * swap_loss).detach())
                setattr(outputs, "cache_hit_rate", cache_aux["cache_hit_rate"].detach())
                setattr(outputs, "cache_retention_rate", cache_aux["cache_retention_rate"].detach())
                setattr(outputs, "layer_swap_weight", cache_aux["layer_swap_weight"].detach())
                setattr(
                    outputs,
                    "effective_swap_weight",
                    torch.tensor(effective_swap_weight, device=swap_loss.device, dtype=swap_loss.dtype),
                )
                if compute_hard_metrics:
                    setattr(outputs, "hard_cache_hit_rate", cache_aux["hard_cache_hit_rate"].detach())
                    setattr(outputs, "hard_cache_miss_rate", cache_aux["hard_cache_miss_rate"].detach())
                    setattr(outputs, "hard_cache_overlap_count", cache_aux["hard_cache_overlap_count"].detach())
                setattr(outputs, "future_entropy", cache_aux["future_entropy"].detach())
                for metric_name in (
                    "hybrid_cache_hit_rate",
                    "temporal_cache_hit_rate",
                    "current_cache_hit_rate",
                    "spatio_cache_hit_rate",
                    "spatio_entropy",
                ):
                    if metric_name in cache_aux:
                        setattr(outputs, metric_name, cache_aux[metric_name].detach())
                setattr(outputs, "balance_loss", balance_loss.detach())
                setattr(outputs, "weighted_balance_loss", (cache_cfg.balance_weight * balance_loss).detach())
            loss = total_loss
        elif hasattr(cache_model, "compute_window_cache_aux"):
            window_metrics = cache_model.compute_window_cache_aux(
                inputs.get("attention_mask"),
                compute_hard_metrics=compute_hard_metrics,
            )
            window_cfg = cache_model._window_cache_config
            avg_swap_loss = window_metrics["avg_swap_loss"]
            peak_swap_loss = window_metrics["peak_swap_loss"]
            mtp_swap_loss = window_metrics["mtp_swap_loss"]

            total_loss = loss
            if window_cfg.training_mode == "cache_aware":
                total_loss = (
                    loss
                    + window_cfg.avg_swap_coeff * avg_swap_loss
                    + window_cfg.peak_swap_coeff * peak_swap_loss
                    + window_cfg.mtp_swap_coeff * mtp_swap_loss
                )

            detached_metrics = self._build_window_cache_metric_dict(
                moe_native_loss=moe_native_loss,
                total_loss=total_loss,
                window_metrics=window_metrics,
                avg_swap_coeff=window_cfg.avg_swap_coeff,
                peak_swap_coeff=window_cfg.peak_swap_coeff,
                mtp_swap_coeff=window_cfg.mtp_swap_coeff,
                include_hard_metrics=compute_hard_metrics,
            )
            if model.training:
                self._latest_train_cache_metrics = detached_metrics
            else:
                self._accumulate_eval_cache_metrics(detached_metrics, self._infer_batch_size(inputs))

            if isinstance(outputs, dict):
                outputs["moe_native_loss"] = moe_native_loss.detach()
                outputs["total_loss"] = total_loss.detach()
                outputs["cache_avg_swap_metric"] = window_metrics["avg_swap_metric"].detach()
                outputs["cache_max_swap_metric"] = window_metrics["peak_swap_metric"].detach()
                outputs["cache_avg_swap_loss"] = avg_swap_loss.detach()
                outputs["cache_peak_swap_loss"] = peak_swap_loss.detach()
                outputs["cache_mtp_swap_loss"] = mtp_swap_loss.detach()
                outputs["weighted_cache_avg_swap_loss"] = (window_cfg.avg_swap_coeff * avg_swap_loss).detach()
                outputs["weighted_cache_peak_swap_loss"] = (window_cfg.peak_swap_coeff * peak_swap_loss).detach()
                outputs["weighted_cache_mtp_swap_loss"] = (window_cfg.mtp_swap_coeff * mtp_swap_loss).detach()
                outputs["cache_cache_utilization"] = window_metrics["cache_utilization"].detach()
                outputs["soft_cache_hit_rate"] = window_metrics["soft_cache_hit_rate"].detach()
                outputs["soft_cache_miss_rate"] = window_metrics["soft_cache_miss_rate"].detach()
                outputs["avg_expected_active_experts"] = window_metrics["avg_expected_active_experts"].detach()
                if compute_hard_metrics:
                    outputs["hard_cache_hit_rate"] = window_metrics["hard_cache_hit_rate"].detach()
                    outputs["hard_cache_miss_rate"] = window_metrics["hard_cache_miss_rate"].detach()
                    outputs["hard_cache_overlap_count"] = window_metrics["hard_cache_overlap_count"].detach()
                    outputs["hard_cache_swap_count"] = window_metrics["hard_cache_swap_count"].detach()
            else:
                setattr(outputs, "moe_native_loss", moe_native_loss.detach())
                setattr(outputs, "total_loss", total_loss.detach())
                setattr(outputs, "cache_avg_swap_metric", window_metrics["avg_swap_metric"].detach())
                setattr(outputs, "cache_max_swap_metric", window_metrics["peak_swap_metric"].detach())
                setattr(outputs, "cache_avg_swap_loss", avg_swap_loss.detach())
                setattr(outputs, "cache_peak_swap_loss", peak_swap_loss.detach())
                setattr(outputs, "cache_mtp_swap_loss", mtp_swap_loss.detach())
                setattr(outputs, "weighted_cache_avg_swap_loss", (window_cfg.avg_swap_coeff * avg_swap_loss).detach())
                setattr(outputs, "weighted_cache_peak_swap_loss", (window_cfg.peak_swap_coeff * peak_swap_loss).detach())
                setattr(outputs, "weighted_cache_mtp_swap_loss", (window_cfg.mtp_swap_coeff * mtp_swap_loss).detach())
                setattr(outputs, "cache_cache_utilization", window_metrics["cache_utilization"].detach())
                setattr(outputs, "soft_cache_hit_rate", window_metrics["soft_cache_hit_rate"].detach())
                setattr(outputs, "soft_cache_miss_rate", window_metrics["soft_cache_miss_rate"].detach())
                setattr(outputs, "avg_expected_active_experts", window_metrics["avg_expected_active_experts"].detach())
                if compute_hard_metrics:
                    setattr(outputs, "hard_cache_hit_rate", window_metrics["hard_cache_hit_rate"].detach())
                    setattr(outputs, "hard_cache_miss_rate", window_metrics["hard_cache_miss_rate"].detach())
                    setattr(outputs, "hard_cache_overlap_count", window_metrics["hard_cache_overlap_count"].detach())
                    setattr(outputs, "hard_cache_swap_count", window_metrics["hard_cache_swap_count"].detach())
            loss = total_loss
        elif hasattr(cache_model, "compute_lru_cache_metrics"):
            lru_metrics = cache_model.compute_lru_cache_metrics(
                inputs.get("attention_mask"),
                compute_hard_metrics=compute_hard_metrics,
            )
            detached_metrics = self._build_lru_metric_dict(
                moe_native_loss=moe_native_loss,
                lru_metrics=lru_metrics,
                include_hard_metrics=compute_hard_metrics,
            )
            if model.training:
                self._latest_train_cache_metrics = detached_metrics
            else:
                self._accumulate_eval_cache_metrics(detached_metrics, self._infer_batch_size(inputs))

            if isinstance(outputs, dict):
                outputs["moe_native_loss"] = moe_native_loss.detach()
                outputs["total_loss"] = moe_native_loss.detach()
                if compute_hard_metrics:
                    outputs["hard_cache_hit_rate"] = lru_metrics["hard_cache_hit_rate"].detach()
                    outputs["hard_cache_miss_rate"] = lru_metrics["hard_cache_miss_rate"].detach()
                    outputs["hard_cache_overlap_count"] = lru_metrics["hard_cache_overlap_count"].detach()
            else:
                setattr(outputs, "moe_native_loss", moe_native_loss.detach())
                setattr(outputs, "total_loss", moe_native_loss.detach())
                if compute_hard_metrics:
                    setattr(outputs, "hard_cache_hit_rate", lru_metrics["hard_cache_hit_rate"].detach())
                    setattr(outputs, "hard_cache_miss_rate", lru_metrics["hard_cache_miss_rate"].detach())
                    setattr(outputs, "hard_cache_overlap_count", lru_metrics["hard_cache_overlap_count"].detach())

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        if any(key.startswith("eval_") for key in logs):
            logs.update(self._get_eval_cache_metrics(prefix="eval"))
        else:
            logs.update(self._latest_train_cache_metrics)
        self._append_layer_adaptive_compact_log(logs)
        super().log(logs, *args, **kwargs)

    def evaluate(self, *args, **kwargs):
        self._reset_eval_cache_metrics()
        return super().evaluate(*args, **kwargs)

    def save_cache_moe_layer_adaptive_state(self, output_dir: Optional[str] = None) -> None:
        cache_model = _unwrap_cache_model(self.model)
        cache_cfg = getattr(cache_model, "_cache_moe_config", None)
        if cache_cfg is None or not getattr(cache_cfg, "layer_adaptive_swap_weight", False):
            return
        if not self._is_world_process_zero():
            return
        output_path = Path(output_dir or self.args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": int(getattr(self.state, "global_step", 0)),
            "epoch": None if self.state.epoch is None else float(self.state.epoch),
            "weights": [float(value) for value in getattr(cache_cfg, "_layer_adaptive_weights", [])],
            "score": [float(value) for value in getattr(cache_cfg, "_layer_adaptive_last_score", []) or []],
            "cosine": [float(value) for value in getattr(cache_cfg, "_layer_adaptive_last_cosine", []) or []],
            "cosine_ema": [float(value) for value in getattr(cache_cfg, "_layer_adaptive_cosine_ema", []) or []],
            "cache_hit_rate": [float(value) for value in getattr(cache_cfg, "_layer_adaptive_last_hit", []) or []],
            "update_count": int(getattr(cache_cfg, "_layer_adaptive_update_count", 0)),
            "config": {
                "global_swap_weight": float(cache_cfg.swap_weight),
                "weight_min": float(cache_cfg.layer_adaptive_weight_min),
                "weight_max": float(cache_cfg.layer_adaptive_weight_max),
                "warmup_steps": int(cache_cfg.layer_adaptive_warmup_steps),
                "update_interval": int(cache_cfg.layer_adaptive_update_interval),
                "lr": float(cache_cfg.layer_adaptive_lr),
                "cosine_ema": float(getattr(cache_cfg, "layer_adaptive_cosine_ema", 0.95)),
                "cosine_scale": float(getattr(cache_cfg, "layer_adaptive_cosine_scale", 1.0)),
                "cosine_deadband": float(getattr(cache_cfg, "layer_adaptive_cosine_deadband", 0.0)),
            },
        }
        with (output_path / "layer_adaptive_state.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
