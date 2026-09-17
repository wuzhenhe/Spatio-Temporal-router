import argparse
import json
import math
import os
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from transformers import TrainingArguments

from cache_policy_baselines import (
    CachePolicyConfig,
    build_promoe_trace_sample,
    infer_finemoe_prefetch_budget,
    infer_promoe_prefetch_budget,
    infer_specmd_prefetch_budget,
    simulate_cache_policy_metrics,
    simulate_cache_policy_metrics_pair,
)
from cache_moe import _contains_any, _initial_inference_cache, _top_priority_cache, _update_inference_cache
from collator import SupervisedDataCollator
from config import load_config, resolve_cache_size
from data import _build_messages, _load_raw_dataset, _render_messages, _tokenize_example
from checkpoints import resolve_checkpoint_path
from finemoe_store import finemoe_store_summary, load_finemoe_store
from promoe_predictor import (
    PROMOE_PREDICTOR_FORMAT,
    build_promoe_layer_mapping,
    promoe_predictor_macs_per_model,
    promoe_predictor_model_count,
    promoe_predictor_parameter_count,
)
from temporal_option_moe import _decoder_layers, temporal_option_parameter_summary
from train import build_model, build_tokenizer
from trainer import MoESFTTrainer
from trainer import _unwrap_cache_model


FUTURE_ONLY_PRIORITY_MODES = {"future_only", "future", "mlp_only", "predictor_only"}
ASYNC_FUTURE_PREFETCH_POLICIES = {"future_prefetch", "future_router_prefetch"}
CACHE_POLICY_CHOICES = [
    "auto",
    "lru",
    "lfu",
    "lrfu",
    "least_stale",
    "specmd_prefetch",
    "specmd",
    "promoe",
    "finemoe",
    "future_prefetch",
    "future_router_prefetch",
]
CACHE_POLICY_ALIASES = {
    "specmd_least_stale": "specmd_prefetch",
    "specmd-least-stale": "specmd_prefetch",
    "specmd/least-stale": "specmd_prefetch",
    "specmd/least_stale": "specmd_prefetch",
}


def is_future_only_priority_mode(mode: str) -> bool:
    return str(mode or "").lower() in FUTURE_ONLY_PRIORITY_MODES


def validate_future_only_cache_policy_binding(cache_policy_config: CachePolicyConfig) -> None:
    if not is_future_only_priority_mode(cache_policy_config.future_cache_priority_mode):
        return
    if cache_policy_config.policy in ASYNC_FUTURE_PREFETCH_POLICIES:
        return
    raise ValueError(
        "cache_moe.future_cache_priority_mode=future_only is the async-prefetch "
        "future-router baseline. Its training, inference scoring, and hard-cache "
        "metrics are bound together: evaluate it with --cache-policy future_prefetch "
        "(or future_router_prefetch), which prefetches the top cache_size experts from "
        "future_probs only. Do not use --cache-policy auto or normal cache policies; "
        "those belong to the regular current_plus_future future-router setting."
    )


def normalize_cache_policy_name(policy: str) -> str:
    normalized = str(policy or "").strip().lower()
    normalized = CACHE_POLICY_ALIASES.get(normalized, normalized)
    if normalized not in CACHE_POLICY_CHOICES:
        valid = ", ".join(CACHE_POLICY_CHOICES)
        raise ValueError(f"Unsupported cache policy {policy!r}. Expected one of: {valid}.")
    return normalized


def parse_cache_policy_list(raw_value: Optional[str]) -> List[str]:
    if raw_value is None:
        return []
    policies: List[str] = []
    seen = set()
    for item in str(raw_value).split(","):
        item = item.strip()
        if not item:
            continue
        policy = normalize_cache_policy_name(item)
        if policy not in seen:
            policies.append(policy)
            seen.add(policy)
    return policies


def parse_spatio_temporal_refine_budget_list(raw_value: Optional[str]) -> List[int]:
    if raw_value is None:
        return []
    budgets: List[int] = []
    seen = set()
    for item in str(raw_value).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            budget = int(item)
        except ValueError as exc:
            raise ValueError(
                f"Invalid spatio-temporal refine budget {item!r}; expected comma-separated integers."
            ) from exc
        if budget < 0:
            raise ValueError("Spatio-temporal refine budgets must be non-negative.")
        if budget not in seen:
            budgets.append(budget)
            seen.add(budget)
    if not budgets:
        raise ValueError("--spatio-temporal-refine-budgets did not contain any integers.")
    return budgets


def build_cache_policy_config(
    args: argparse.Namespace,
    config: Dict[str, Any],
    policy: str,
    promoe_predictor: Optional[Dict[str, Any]],
    finemoe_store: Optional[Dict[str, Any]],
) -> CachePolicyConfig:
    return CachePolicyConfig(
        policy=normalize_cache_policy_name(policy),
        recency_weight=args.cache_policy_recency_weight,
        frequency_weight=args.cache_policy_frequency_weight,
        recency_decay=args.cache_policy_recency_decay,
        lrfu_lambda=args.cache_policy_lrfu_lambda,
        prefetch_budget=args.cache_policy_prefetch_budget,
        prefetch_lookahead=args.cache_policy_prefetch_lookahead,
        future_cache_priority_mode=config.get("cache_moe", {}).get(
            "future_cache_priority_mode",
            "current_plus_future",
        ),
        promoe_predictor=promoe_predictor,
        spatio_temporal_refine_mode=args.spatio_temporal_refine_mode,
        spatio_temporal_refine_budget=args.spatio_temporal_refine_budget,
        finemoe_store=finemoe_store,
        finemoe_top_k=args.finemoe_top_k,
        finemoe_candidate_pool=args.finemoe_candidate_pool,
        finemoe_semantic_weight=args.finemoe_semantic_weight,
        finemoe_trajectory_weight=args.finemoe_trajectory_weight,
        finemoe_prefetch_threshold=args.finemoe_prefetch_threshold,
        finemoe_min_prefetch_threshold=args.finemoe_min_prefetch_threshold,
        finemoe_max_prefetch_threshold=args.finemoe_max_prefetch_threshold,
        finemoe_eviction_probability_weight=args.finemoe_eviction_probability_weight,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GSM8K answers and compute exact numeric accuracy.")
    parser.add_argument("--config", action="append", required=True, help="YAML config/profile path.")
    parser.add_argument("--model-path", default=None, help="Run output dir or checkpoint-* dir.")
    parser.add_argument(
        "--skip-checkpoint-load",
        action="store_true",
        help="Evaluate the base model loaded from config.model without loading a fine-tuned checkpoint.",
    )
    parser.add_argument("--output-dir", default="./eval_outputs/gsm8k_accuracy", help="Output directory.")
    parser.add_argument("--predictions-output", default=None, help="Optional JSONL prediction path.")
    parser.add_argument("--metrics-output", default=None, help="Optional JSON metrics path.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Limit examples for smoke tests.")
    parser.add_argument("--eval-split", default=None, help="Override dataset.eval_split, e.g. train for trace collection.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation budget.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Use 0 for greedy decoding.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--compute-generation-nll",
        action="store_true",
        help="Also compute NLL of model-generated tokens. Disabled by default to avoid storing vocab scores.",
    )
    parser.add_argument(
        "--skip-reference-loss",
        action="store_true",
        help="Skip the extra supervised reference-loss forward pass; accuracy/cache/load metrics are still computed.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Print progress every N local samples per rank. Use 0 to disable progress logs.",
    )
    parser.add_argument(
        "--expert-usage-output",
        default=None,
        help="Optional torch .pt path for averaged per-layer expert usage counts.",
    )
    parser.add_argument(
        "--no-generation-cache",
        action="store_true",
        help="Disable KV cache during generation. Expert usage recording is cleaner with cache enabled.",
    )
    parser.add_argument(
        "--expert-size-mb",
        type=float,
        default=None,
        help="Override routed expert weight size in decimal MB for expert-load estimates.",
    )
    parser.add_argument(
        "--expert-load-bandwidth-gbps",
        type=float,
        default=0.0,
        help=(
            "Optional effective expert-load bandwidth in decimal GB/s. "
            "When positive, report load-only TPOT proxy in ms per generated token."
        ),
    )
    parser.add_argument(
        "--predictor-effective-tflops",
        type=float,
        default=0.0,
        help=(
            "Optional effective compute throughput for predictor overhead estimates. "
            "When positive, report predictor_overhead_ms_per_token."
        ),
    )
    parser.add_argument(
        "--cache-policy",
        choices=CACHE_POLICY_CHOICES,
        default="auto",
        help=(
            "Hard-cache simulation policy. 'auto' keeps the method's native behavior "
            "(future-router cache for cache_moe, LRU otherwise). "
            "lfu/lrfu/least_stale/specmd_prefetch/promoe/finemoe/future_prefetch "
            "are isolated replay baselines and do not affect model generation. "
            "future_cache_priority_mode=future_only must use future_prefetch."
        ),
    )
    parser.add_argument(
        "--extra-cache-policies",
        default=None,
        help=(
            "Comma-separated additional hard-cache replay policies to evaluate on the same generation trace, "
            "for example lru,lfu,lrfu,specmd_prefetch. These policies do not trigger extra generation."
        ),
    )
    parser.add_argument("--cache-policy-recency-weight", type=float, default=1.0)
    parser.add_argument("--cache-policy-frequency-weight", type=float, default=1.0)
    parser.add_argument("--cache-policy-recency-decay", type=float, default=128.0)
    parser.add_argument(
        "--cache-policy-lrfu-lambda",
        type=float,
        default=0.5,
        help="For lrfu, original LRFU CRF decay parameter lambda. 0 approximates LFU; larger values favor recency.",
    )
    parser.add_argument(
        "--cache-policy-prefetch-budget",
        type=int,
        default=-1,
        help=(
            "For promoe/specmd_prefetch, number of experts to prefetch per predicted layer. "
            "For promoe, -1 uses the predictor's trained num_predict_expert_per_layer. "
            "For specmd_prefetch, -1 uses cache_size. "
            "0 disables prefetch; positive values are capped at cache_size."
        ),
    )
    parser.add_argument(
        "--cache-policy-prefetch-lookahead",
        type=int,
        default=3,
        help="For promoe/specmd_prefetch, layer lookahead window; the paper/official ProMoE best config uses 3.",
    )
    parser.add_argument(
        "--spatio-temporal-refine-mode",
        choices=["topk", "replace_lowest"],
        default="topk",
        help=(
            "For spatio_temporal_enabled + --cache-policy auto only. "
            "'topk' replaces the layer cache with top cache_size experts by full priority. "
            "'replace_lowest' takes the top --spatio-temporal-refine-budget experts by full priority "
            "and only inserts the missing ones, evicting the lowest-priority resident experts."
        ),
    )
    parser.add_argument(
        "--spatio-temporal-refine-budget",
        type=int,
        default=-1,
        help=(
            "For --spatio-temporal-refine-mode replace_lowest, number of full-priority top experts "
            "to consider during the second-stage spatio-temporal refinement. "
            "-1 uses cache_size; pass 8 to only conditionally insert missing top-8 experts."
        ),
    )
    parser.add_argument(
        "--spatio-temporal-refine-budgets",
        default=None,
        help=(
            "Optional comma-separated replace_lowest budgets to replay from one generation trace, "
            "for example 4,8,12. This is valid only for cache-policy=auto with a spatial router and "
            "spatio-temporal-refine-mode=replace_lowest. The first value is the primary metric; the "
            "remaining values are reported under spatio_temporal_refine_budget_metrics without "
            "running model.generate again."
        ),
    )
    parser.add_argument(
        "--promoe-predictor-path",
        default=None,
        help="Torch .pt predictor from src/train_promoe_predictor.py. Required for --cache-policy promoe.",
    )
    parser.add_argument(
        "--promoe-trace-output",
        default=None,
        help="Optional torch .pt path to save selected_experts traces for training a ProMoE-style predictor.",
    )
    parser.add_argument(
        "--finemoe-store-path",
        default=None,
        help="Torch .pt Expert Map Store from src/train_finemoe_store.py. Required for --cache-policy finemoe.",
    )
    parser.add_argument(
        "--finemoe-top-k",
        type=int,
        default=8,
        help="Number of retrieved FineMoE expert maps to aggregate after semantic/trajectory scoring.",
    )
    parser.add_argument(
        "--finemoe-candidate-pool",
        type=int,
        default=64,
        help="Semantic candidate pool size before FineMoE trajectory re-ranking.",
    )
    parser.add_argument("--finemoe-semantic-weight", type=float, default=1.0)
    parser.add_argument("--finemoe-trajectory-weight", type=float, default=1.0)
    parser.add_argument(
        "--finemoe-prefetch-threshold",
        type=float,
        default=-1.0,
        help="Fixed cumulative probability threshold. Negative enables FineMoE-style dynamic thresholding.",
    )
    parser.add_argument("--finemoe-min-prefetch-threshold", type=float, default=0.05)
    parser.add_argument("--finemoe-max-prefetch-threshold", type=float, default=0.35)
    parser.add_argument("--finemoe-eviction-probability-weight", type=float, default=1.0)
    return parser.parse_args()


def build_eval_args(config: Dict[str, Any], output_dir: Path) -> TrainingArguments:
    train_cfg = config["training"]
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=1,
        bf16=train_cfg["bf16"],
        fp16=train_cfg["fp16"],
        tf32=train_cfg["tf32"],
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=["labels"],
    )


def ignore_fsdp_state_restore_error(exc: KeyError) -> bool:
    if exc.args != (None,):
        return False
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return False
    rank = torch.distributed.get_rank()
    if rank == 0:
        print(
            "Ignored FSDP state_dict_type restore KeyError(None) after checkpoint load. "
            "This is a torch/accelerate compatibility issue during context cleanup."
        )
    return True


def load_trained_weights(trainer: MoESFTTrainer, checkpoint_path: Path) -> None:
    load_from_checkpoint = getattr(trainer, "_load_from_checkpoint", None)
    if load_from_checkpoint is None:
        raise RuntimeError("This Transformers Trainer version does not expose _load_from_checkpoint.")

    StateDictType = None
    fsdp_plugin = getattr(getattr(trainer, "accelerator", None), "state", None)
    fsdp_plugin = getattr(fsdp_plugin, "fsdp_plugin", None)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        from torch.distributed.fsdp import FullyShardedDataParallel, StateDictType

        original_set_state_dict_type = FullyShardedDataParallel.set_state_dict_type

        def safe_set_state_dict_type(
            module,
            state_dict_type,
            state_dict_config=None,
            optim_state_dict_config=None,
        ):
            if state_dict_type is None:
                state_dict_type = StateDictType.SHARDED_STATE_DICT
            return original_set_state_dict_type(
                module,
                state_dict_type,
                state_dict_config,
                optim_state_dict_config,
            )

        FullyShardedDataParallel.set_state_dict_type = staticmethod(safe_set_state_dict_type)

    if fsdp_plugin is not None:
        if getattr(fsdp_plugin, "state_dict_type", None) is None:
            if StateDictType is None:
                raise RuntimeError("FSDP plugin is active, but torch.distributed is not initialized.")
            fsdp_plugin.state_dict_type = StateDictType.SHARDED_STATE_DICT

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        FullyShardedDataParallel.set_state_dict_type(
            trainer.model,
            StateDictType.SHARDED_STATE_DICT,
        )

    try:
        load_from_checkpoint(str(checkpoint_path), model=trainer.model)
    except KeyError as exc:
        if not ignore_fsdp_state_restore_error(exc):
            raise
    except RuntimeError as exc:
        message = str(exc)
        if "Missing key in checkpoint state_dict" in message and "spatio_gate" in message:
            raise RuntimeError(
                "The selected checkpoint does not contain spatio_gate weights, but the eval "
                "config enables cache_moe.spatio_temporal_enabled=true. This usually means "
                "the --model-path points to an old/non-spatio checkpoint, or the training run "
                "did not enable configs/methods/spatio-temporal.yaml. Use the final output "
                "directory from the matching run, or retrain in a new output directory. "
                f"Checkpoint path: {checkpoint_path}"
            ) from exc
        raise
    except TypeError:
        try:
            load_from_checkpoint(str(checkpoint_path))
        except KeyError as exc:
            if not ignore_fsdp_state_restore_error(exc):
                raise
        except RuntimeError as exc:
            message = str(exc)
            if "Missing key in checkpoint state_dict" in message and "spatio_gate" in message:
                raise RuntimeError(
                    "The selected checkpoint does not contain spatio_gate weights, but the eval "
                    "config enables cache_moe.spatio_temporal_enabled=true. This usually means "
                    "the --model-path points to an old/non-spatio checkpoint, or the training run "
                    "did not enable configs/methods/spatio-temporal.yaml. Use the final output "
                    "directory from the matching run, or retrain in a new output directory. "
                    f"Checkpoint path: {checkpoint_path}"
                ) from exc
            raise


def future_router_parameter_summary(model) -> Dict[str, Any]:
    future_names = []
    spatio_names = []
    future_parameter_count = 0
    spatio_parameter_count = 0
    for name, parameter in model.named_parameters():
        if "future_gate" in name:
            future_names.append(name)
            future_parameter_count += int(parameter.numel())
        if "spatio_gate" in name:
            spatio_names.append(name)
            spatio_parameter_count += int(parameter.numel())
    names = future_names + spatio_names
    return {
        "future_router_tensor_count": len(names),
        "future_router_parameter_count": future_parameter_count + spatio_parameter_count,
        "first_future_router_tensors": names[:8],
        "future_gate_tensor_count": len(future_names),
        "future_gate_parameter_count": future_parameter_count,
        "spatio_gate_tensor_count": len(spatio_names),
        "spatio_gate_parameter_count": spatio_parameter_count,
        **temporal_option_parameter_summary(model),
    }


def validate_future_router_loaded(model, config: Dict[str, Any], rank: int) -> Dict[str, Any]:
    summary = future_router_parameter_summary(model)
    expects_future_router = bool(config.get("cache_moe", {}).get("enabled", False))
    if expects_future_router and summary["future_router_tensor_count"] == 0:
        raise RuntimeError(
            "cache_moe.enabled=true, but no cache-router predictor parameters were found after checkpoint load. "
            "The requested router is not installed or the wrong method config was used."
        )
    cache_cfg = config.get("cache_moe", {})
    expects_temporal = bool(expects_future_router and not cache_cfg.get("spatio_only_enabled", False))
    if expects_temporal and summary["future_gate_tensor_count"] == 0:
        raise RuntimeError(
            "The selected method requires a temporal/future gate, but no future_gate parameters "
            "were found after checkpoint load."
        )
    expects_spatio = bool(
        cache_cfg.get("spatio_temporal_enabled", False) or cache_cfg.get("spatio_only_enabled", False)
    )
    if expects_spatio and summary["spatio_gate_tensor_count"] == 0:
        raise RuntimeError(
            "A spatial cache-router mode is enabled, but no spatio_gate parameters were found "
            "after checkpoint load. Use a checkpoint trained with the matching spatial method profile."
        )
    expects_temporal_option = bool(config.get("temporal_option_moe", {}).get("enabled", False))
    if expects_temporal_option and summary["temporal_option_tensor_count"] == 0:
        raise RuntimeError(
            "temporal_option_moe.enabled=true, but no temporal option controller parameters were found "
            "after checkpoint load. Use a checkpoint trained with the temporal option profile."
        )
    if rank == 0:
        print(
            "Future router parameter check: "
            f"tensors={summary['future_router_tensor_count']}, "
            f"params={summary['future_router_parameter_count']}, "
            f"future_gate_tensors={summary['future_gate_tensor_count']}, "
            f"spatio_gate_tensors={summary['spatio_gate_tensor_count']}, "
            f"temporal_option_tensors={summary['temporal_option_tensor_count']}, "
            f"examples={summary['first_future_router_tensors']}",
            flush=True,
        )
    return summary


def normalize_number(text: str) -> Optional[str]:
    if text is None:
        return None
    cleaned = text.replace(",", "").strip()
    matches = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", cleaned)
    if not matches:
        return None
    value = matches[-1]
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            number = float(numerator) / float(denominator)
        else:
            number = float(value)
    except ValueError:
        return value
    if math.isfinite(number) and abs(number - round(number)) < 1e-6:
        return str(int(round(number)))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def extract_gsm8k_answer(text: str) -> Optional[str]:
    if text is None:
        return None
    marker_matches = re.findall(r"####\s*([^\n\r]+)", text)
    if marker_matches:
        return normalize_number(marker_matches[-1])
    return normalize_number(text)


def exact_match(prediction: Optional[str], reference: Optional[str]) -> bool:
    return prediction is not None and reference is not None and prediction == reference


def distributed_info() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def resolve_generation_device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def input_embedding_device(model) -> torch.device:
    candidate = model
    visited = set()
    while id(candidate) not in visited:
        visited.add(id(candidate))
        if hasattr(candidate, "get_input_embeddings"):
            embeddings = candidate.get_input_embeddings()
            if embeddings is not None:
                try:
                    return next(embeddings.parameters()).device
                except StopIteration:
                    pass
        if hasattr(candidate, "module"):
            candidate = candidate.module
            continue
        if hasattr(candidate, "_orig_mod"):
            candidate = candidate._orig_mod
            continue
        break
    return next(model.parameters()).device


def move_model_for_generation(model, generation_device: torch.device, rank: int):
    before_device = input_embedding_device(model)
    if before_device != generation_device:
        if generation_device.type == "cuda":
            model.to(generation_device)
            torch.cuda.empty_cache()
        else:
            model.to(generation_device)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval()
    after_device = input_embedding_device(model)
    if rank == 0:
        print(
            f"Generation device setup: embedding {before_device} -> {after_device}; "
            f"requested_device={generation_device}."
        )
    return model, after_device


def disable_training_router_outputs_for_generation(model) -> None:
    candidate = model
    visited = set()
    while id(candidate) not in visited:
        visited.add(id(candidate))
        config = getattr(candidate, "config", None)
        if config is not None and hasattr(config, "output_router_logits"):
            config.output_router_logits = False
        generation_config = getattr(candidate, "generation_config", None)
        if generation_config is not None and hasattr(generation_config, "output_router_logits"):
            generation_config.output_router_logits = False
        if hasattr(candidate, "module"):
            candidate = candidate.module
            continue
        if hasattr(candidate, "_orig_mod"):
            candidate = candidate._orig_mod
            continue
        break


def gather_predictions_from_rank_files(
    local_predictions: List[Dict[str, Any]],
    output_dir: Path,
    world_size: int,
    rank: int,
) -> List[Dict[str, Any]]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return sorted(local_predictions, key=lambda item: item["index"])
    if rank != 0:
        return []

    predictions: List[Dict[str, Any]] = []
    for shard_rank in range(world_size):
        shard_path = output_dir / f"predictions.rank{shard_rank}.jsonl.tmp"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing prediction shard from rank {shard_rank}: {shard_path}")
        with shard_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    predictions.append(json.loads(line))
    return sorted(predictions, key=lambda item: item["index"])


def _rank_file_timeout_seconds() -> float:
    return float(os.environ.get("EVAL_RANK_SHARD_TIMEOUT_SECONDS", "86400"))


def wait_for_rank_files(
    paths: List[Path],
    description: str,
    timeout_seconds: Optional[float] = None,
) -> None:
    timeout = _rank_file_timeout_seconds() if timeout_seconds is None else float(timeout_seconds)
    deadline = time.time() + timeout
    last_log = 0.0
    while True:
        missing = [path for path in paths if not path.exists()]
        if not missing:
            return
        now = time.time()
        if now >= deadline:
            missing_text = ", ".join(str(path) for path in missing[:8])
            if len(missing) > 8:
                missing_text += f", ... ({len(missing)} missing)"
            raise TimeoutError(f"Timed out waiting for {description}: {missing_text}")
        if now - last_log >= 60.0:
            print(
                f"[rank0] waiting for {len(missing)}/{len(paths)} {description} shards...",
                flush=True,
            )
            last_log = now
        time.sleep(10.0)


def atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def gather_metric_tensors_from_rank_files(
    output_dir: Path,
    world_size: int,
    rank: int,
    usage_counts: torch.Tensor,
    token_counts: torch.Tensor,
    routed_slot_counts: torch.Tensor,
    generation_metric_sums: torch.Tensor,
    extra_cache_policy_metric_sums: Optional[Dict[str, torch.Tensor]] = None,
    return_extra_cache_policy_metric_sums: bool = False,
):
    extra_cache_policy_metric_sums = extra_cache_policy_metric_sums or {}
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        result = (usage_counts, token_counts, routed_slot_counts, generation_metric_sums)
        if return_extra_cache_policy_metric_sums:
            return (*result, extra_cache_policy_metric_sums)
        return result

    shard_path = output_dir / f"metrics.rank{rank}.pt"
    atomic_torch_save(
        {
            "rank": int(rank),
            "usage_counts": usage_counts.detach().cpu(),
            "token_counts": token_counts.detach().cpu(),
            "routed_slot_counts": routed_slot_counts.detach().cpu(),
            "generation_metric_sums": generation_metric_sums.detach().cpu(),
            "extra_cache_policy_metric_sums": {
                policy: metric_sums.detach().cpu()
                for policy, metric_sums in extra_cache_policy_metric_sums.items()
            },
        },
        shard_path,
    )

    if rank != 0:
        result = (usage_counts, token_counts, routed_slot_counts, generation_metric_sums)
        if return_extra_cache_policy_metric_sums:
            return (*result, extra_cache_policy_metric_sums)
        return result

    shard_paths = [output_dir / f"metrics.rank{shard_rank}.pt" for shard_rank in range(world_size)]
    wait_for_rank_files(shard_paths, description="metric")

    usage_sum: Optional[torch.Tensor] = None
    token_sum: Optional[torch.Tensor] = None
    routed_sum: Optional[torch.Tensor] = None
    metric_sum: Optional[torch.Tensor] = None
    extra_metric_sums: Dict[str, torch.Tensor] = {}
    for path in shard_paths:
        payload = torch.load(path, map_location="cpu")
        shard_usage = payload["usage_counts"].to(dtype=torch.float64)
        shard_tokens = payload["token_counts"].to(dtype=torch.float64)
        shard_routed = payload["routed_slot_counts"].to(dtype=torch.float64)
        shard_metrics = payload["generation_metric_sums"].to(dtype=torch.float64)
        usage_sum = shard_usage if usage_sum is None else usage_sum + shard_usage
        token_sum = shard_tokens if token_sum is None else token_sum + shard_tokens
        routed_sum = shard_routed if routed_sum is None else routed_sum + shard_routed
        metric_sum = shard_metrics if metric_sum is None else metric_sum + shard_metrics
        for policy, shard_extra_metrics in payload.get("extra_cache_policy_metric_sums", {}).items():
            shard_extra_metrics = shard_extra_metrics.to(dtype=torch.float64)
            extra_metric_sums[policy] = (
                shard_extra_metrics
                if policy not in extra_metric_sums
                else extra_metric_sums[policy] + shard_extra_metrics
            )

    if usage_sum is None or token_sum is None or routed_sum is None or metric_sum is None:
        raise RuntimeError("No rank metric shards were loaded.")
    result = (usage_sum, token_sum, routed_sum, metric_sum)
    if return_extra_cache_policy_metric_sums:
        return (*result, extra_metric_sums)
    return result


def load_promoe_predictor(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != PROMOE_PREDICTOR_FORMAT or "models" not in payload:
        raise ValueError(f"Invalid ProMoE predictor file: {path}")
    return payload


def save_promoe_trace_output(
    trace_output: Optional[str],
    local_traces: List[Dict[str, Any]],
    rank: int,
    world_size: int,
) -> None:
    if trace_output is None:
        return

    output = Path(trace_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix or ".pt"
    shard_path = output.with_name(f"{output.stem}.rank{rank}{suffix}")
    atomic_torch_save(
        {
            "format": "promoe_trace_shard_v1",
            "rank": rank,
            "world_size": world_size,
            "samples": local_traces,
        },
        shard_path,
    )

    if rank == 0:
        shard_paths = [output.with_name(f"{output.stem}.rank{shard_rank}{suffix}") for shard_rank in range(world_size)]
        wait_for_rank_files(shard_paths, description="ProMoE trace")
        merged: List[Dict[str, Any]] = []
        for shard in shard_paths:
            payload = torch.load(shard, map_location="cpu")
            merged.extend(payload.get("samples", []))
        merged.sort(key=lambda item: int(item.get("index", 0)))
        atomic_torch_save(
            {
                "format": "promoe_trace_v1",
                "world_size": world_size,
                "samples": merged,
            },
            output,
        )
        print(f"Saved ProMoE trace: path={output}, samples={len(merged)}", flush=True)


def infer_moe_shape(model) -> tuple[int, int]:
    cache_model = _unwrap_cache_model(model)
    decoder_layers = _decoder_layers(cache_model)
    if decoder_layers is None:
        raise ValueError("Expert usage recording expects a decoder model with model.layers.")

    moe_layers = 0
    num_experts = None
    for decoder_layer in decoder_layers:
        mlp = getattr(decoder_layer, "mlp", None)
        if mlp is None:
            continue

        candidate_num_experts = getattr(mlp, "num_experts", None)
        if candidate_num_experts is None and hasattr(mlp, "gate"):
            candidate_num_experts = getattr(mlp.gate, "num_experts", None)
        if candidate_num_experts is None and hasattr(mlp, "gate") and hasattr(mlp.gate, "weight"):
            candidate_num_experts = int(mlp.gate.weight.shape[0])
        if candidate_num_experts is None and hasattr(mlp, "router") and hasattr(mlp.router, "weight"):
            candidate_num_experts = int(mlp.router.weight.shape[0])
        if candidate_num_experts is None and hasattr(mlp, "original_block"):
            original_gate = getattr(mlp.original_block, "gate", None)
            if original_gate is not None and hasattr(original_gate, "weight"):
                candidate_num_experts = int(original_gate.weight.shape[0])
        if candidate_num_experts is None and hasattr(mlp, "original_block"):
            original_router = getattr(mlp.original_block, "router", None)
            if original_router is not None and hasattr(original_router, "weight"):
                candidate_num_experts = int(original_router.weight.shape[0])

        if candidate_num_experts is not None:
            moe_layers += 1
            num_experts = int(candidate_num_experts)

    if moe_layers <= 0 or num_experts is None:
        raise ValueError("Could not infer MoE layer/expert shape for expert usage recording.")
    return moe_layers, num_experts


def _dtype_num_bytes(dtype: Any) -> int:
    if isinstance(dtype, torch.dtype):
        if dtype in (torch.float16, torch.bfloat16):
            return 2
        if dtype in (torch.float32, torch.int32):
            return 4
        if dtype in (torch.float64, torch.int64):
            return 8
        if dtype in (torch.int8, torch.uint8, torch.bool):
            return 1
    dtype_text = str(dtype).lower()
    if "bfloat16" in dtype_text or "float16" in dtype_text or "fp16" in dtype_text or "bf16" in dtype_text:
        return 2
    if "float32" in dtype_text or "fp32" in dtype_text:
        return 4
    if "float64" in dtype_text or "fp64" in dtype_text:
        return 8
    return 2


def infer_routed_expert_size_bytes(model, override_mb: Optional[float] = None) -> int:
    if override_mb is not None:
        return int(float(override_mb) * 1_000_000)

    cache_model = _unwrap_cache_model(model)
    config = getattr(cache_model, "config", None)
    if config is not None:
        hidden_size = getattr(config, "hidden_size", None)
        intermediate_size = (
            getattr(config, "moe_intermediate_size", None)
            or getattr(config, "intermediate_size_mlp", None)
            or getattr(config, "intermediate_size", None)
            or getattr(config, "ffn_dim", None)
        )
        if hidden_size is not None and intermediate_size is not None:
            dtype = getattr(config, "torch_dtype", None)
            if dtype is None:
                try:
                    dtype = next(cache_model.parameters()).dtype
                except StopIteration:
                    dtype = torch.bfloat16
            # Routed experts in Qwen/DeepSeek-style gated MLPs use gate/up/down projections.
            return int(3 * int(hidden_size) * int(intermediate_size) * _dtype_num_bytes(dtype))

    decoder_layers = _decoder_layers(cache_model)
    if decoder_layers is not None:
        for decoder_layer in decoder_layers:
            mlp = getattr(decoder_layer, "mlp", None)
            experts = getattr(mlp, "experts", None)
            if experts is None and hasattr(mlp, "original_block"):
                experts = getattr(mlp.original_block, "experts", None)
            if experts is None:
                continue
            first_expert = None
            if isinstance(experts, torch.nn.ModuleList) and len(experts) > 0:
                first_expert = experts[0]
            elif hasattr(experts, "__getitem__"):
                try:
                    first_expert = experts[0]
                except Exception:
                    first_expert = None
            if first_expert is not None:
                size = sum(int(parameter.numel() * parameter.element_size()) for parameter in first_expert.parameters())
                if size > 0:
                    return size

    return 0


def reset_recorded_expert_stats(model) -> None:
    cache_model = _unwrap_cache_model(model)
    if hasattr(cache_model, "reset_cache_moe_state"):
        cache_model.reset_cache_moe_state()
    if hasattr(cache_model, "reset_window_cache_state"):
        cache_model.reset_window_cache_state()
    if hasattr(cache_model, "reset_lru_cache_metric_state"):
        cache_model.reset_lru_cache_metric_state()
    if hasattr(cache_model, "reset_temporal_option_moe_state"):
        cache_model.reset_temporal_option_moe_state()


def get_recorded_layer_stats(model) -> List[Dict[str, torch.Tensor]]:
    cache_model = _unwrap_cache_model(model)
    for attr in (
        "_cache_moe_layer_stats",
        "_window_cache_layer_stats",
        "_lru_cache_metric_layer_stats",
        "_temporal_option_moe_layer_stats",
    ):
        stats = getattr(cache_model, attr, None)
        if stats:
            return stats
    return []


def temporal_option_generation_summary(model, device: torch.device) -> torch.Tensor:
    stats = get_recorded_layer_stats(model)
    total_switch = 0.0
    total_target_switch = 0.0
    total_coverage = 0.0
    total_overlap = 0.0
    counted = 0.0
    for layer in stats:
        option_masks = layer.get("option_masks")
        switch_flags = layer.get("switch_flags")
        switch_targets = layer.get("switch_targets")
        teacher_selected = layer.get("teacher_selected_experts")
        selected = layer.get("selected_experts")
        if (
            option_masks is None
            or switch_flags is None
            or switch_targets is None
            or teacher_selected is None
            or selected is None
        ):
            continue
        option_masks_cpu = option_masks.detach().to(device="cpu", dtype=torch.bool)
        switch_flags_cpu = switch_flags.detach().to(device="cpu", dtype=torch.float32)
        switch_targets_cpu = switch_targets.detach().to(device="cpu", dtype=torch.float32)
        teacher_cpu = teacher_selected.detach().to(device="cpu", dtype=torch.long)
        selected_cpu = selected.detach().to(device="cpu", dtype=torch.long)
        batch_size, seq_len = switch_flags_cpu.shape
        for batch_idx in range(batch_size):
            for token_idx in range(seq_len):
                mask = option_masks_cpu[batch_idx, token_idx]
                teacher = teacher_cpu[batch_idx, token_idx]
                routed = selected_cpu[batch_idx, token_idx]
                total_switch += float(switch_flags_cpu[batch_idx, token_idx].item())
                total_target_switch += float(switch_targets_cpu[batch_idx, token_idx].item())
                total_coverage += float(_contains_any(teacher, mask.nonzero().flatten()).float().mean().item())
                total_overlap += float(_contains_any(routed, teacher).float().mean().item())
                counted += 1.0
    return torch.tensor(
        [total_switch, total_target_switch, total_coverage, total_overlap, counted],
        device=device,
        dtype=torch.float64,
    )


def accumulate_expert_usage(
    model,
    usage_counts: torch.Tensor,
    token_counts: torch.Tensor,
    routed_slot_counts: torch.Tensor,
) -> None:
    for layer in get_recorded_layer_stats(model):
        selected = layer.get("selected_experts")
        if selected is None:
            continue
        layer_idx = int(layer["layer_idx"])
        selected_flat = selected.detach().to(device=usage_counts.device, dtype=torch.long).reshape(-1)
        if selected_flat.numel() == 0:
            continue
        bincount = torch.bincount(selected_flat, minlength=usage_counts.size(1)).to(dtype=usage_counts.dtype)
        usage_counts[layer_idx] += bincount[: usage_counts.size(1)]
        routed_slot_counts[layer_idx] += float(selected_flat.numel())
        token_counts[layer_idx] += float(selected.numel() // selected.size(-1))


def generation_nll_from_scores(scores, generated_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not scores or generated_tokens.numel() == 0:
        device = generated_tokens.device
        return torch.zeros((), device=device, dtype=torch.float64), torch.zeros((), device=device, dtype=torch.float64)

    loss_sum = torch.zeros((), device=generated_tokens.device, dtype=torch.float64)
    token_count = min(len(scores), int(generated_tokens.numel()))
    for step_idx in range(token_count):
        step_scores = scores[step_idx][0].to(device=generated_tokens.device, dtype=torch.float32)
        token_id = generated_tokens[step_idx].to(device=generated_tokens.device)
        log_probs = torch.log_softmax(step_scores, dim=-1)
        loss_sum = loss_sum - log_probs[token_id].to(dtype=torch.float64)
    return loss_sum, torch.tensor(float(token_count), device=generated_tokens.device, dtype=torch.float64)


def generation_hard_cache_metrics(
    model,
    cache_size: int,
    device: torch.device,
    count_from_token_idx: Optional[int] = None,
    spatio_temporal_refine_mode: str = "topk",
    spatio_temporal_refine_budget: int = -1,
) -> Dict[str, torch.Tensor]:
    zero = torch.zeros((), device=device, dtype=torch.float64)
    cache_model = _unwrap_cache_model(model)
    cache_moe_config = getattr(cache_model, "_cache_moe_config", None)
    grouped: Dict[int, Dict[str, List[torch.Tensor]]] = {}
    for layer in get_recorded_layer_stats(model):
        selected = layer.get("selected_experts")
        if selected is None:
            continue
        layer_idx = int(layer["layer_idx"])
        group = grouped.setdefault(layer_idx, {"selected": [], "future": [], "current": []})
        group["selected"].append(selected.detach().to(device=device, dtype=torch.long))
        future_probs = layer.get("future_probs")
        if future_probs is not None:
            group["future"].append(future_probs.detach().to(device=device, dtype=torch.float32))
        current_probs = layer.get("current_probs")
        if current_probs is not None:
            group["current"].append(current_probs.detach().to(device=device, dtype=torch.float32))
        spatio_probs = layer.get("spatio_probs")
        if spatio_probs is not None:
            group.setdefault("spatio", []).append(spatio_probs.detach().to(device=device, dtype=torch.float32))
        option_masks = layer.get("option_masks")
        if option_masks is not None:
            group.setdefault("option", []).append(option_masks.detach().to(device=device, dtype=torch.bool))
        switch_flags = layer.get("switch_flags")
        if switch_flags is not None:
            group.setdefault("switch", []).append(switch_flags.detach().to(device=device, dtype=torch.float32))

    if getattr(cache_model, "_temporal_option_moe_enabled", False):
        return generation_temporal_option_working_set_cache_metrics(
            grouped=grouped,
            cache_size=cache_size,
            device=device,
            count_from_token_idx=count_from_token_idx,
        )

    if cache_moe_config is not None and getattr(cache_moe_config, "spatio_only_enabled", False):
        return generation_spatio_temporal_hard_cache_metrics(
            grouped=grouped,
            cache_size=cache_size,
            device=device,
            count_from_token_idx=count_from_token_idx,
            refine_mode=spatio_temporal_refine_mode,
            refine_budget=spatio_temporal_refine_budget,
            spatio_only=True,
        )

    if cache_moe_config is not None and getattr(cache_moe_config, "spatio_temporal_enabled", False):
        return generation_spatio_temporal_hard_cache_metrics(
            grouped=grouped,
            cache_size=cache_size,
            device=device,
            count_from_token_idx=count_from_token_idx,
            refine_mode=spatio_temporal_refine_mode,
            refine_budget=spatio_temporal_refine_budget,
        )

    total_hit_rate = zero.clone()
    total_overlap = zero.clone()
    total_access = zero.clone()
    total_miss = zero.clone()
    counted_steps = 0

    def touch_lru(cache: torch.Tensor, expert: torch.Tensor) -> torch.Tensor:
        if cache.numel() > 0:
            cache = cache[cache.ne(expert)]
        cache = torch.cat([cache, expert.detach().reshape(1)])
        if cache.numel() > cache_size:
            cache = cache[-cache_size:]
        return cache

    for group in grouped.values():
        selected_chunks = group["selected"]
        if not selected_chunks:
            continue
        selected = torch.cat(selected_chunks, dim=1)[0]
        if selected.size(0) <= 0:
            continue

        future_chunks = group["future"]
        current_chunks = group["current"]
        if future_chunks and len(future_chunks) == len(selected_chunks):
            future = torch.cat(future_chunks, dim=1)[0]
            current = torch.cat(current_chunks, dim=1)[0] if len(current_chunks) == len(selected_chunks) else None
            cache = _initial_inference_cache(
                selected_experts=selected[0],
                future_scores=future[0],
                cache_size=cache_size,
                current_scores=current[0] if current is not None else None,
                cache_moe_config=cache_moe_config,
            )
            start_idx = 1
        else:
            future = None
            current = None
            cache = torch.empty(0, device=device, dtype=selected.dtype)
            start_idx = 0

        for token_idx in range(start_idx, selected.size(0)):
            token_selected = selected[token_idx]
            hit_count = _contains_any(token_selected, cache).to(dtype=torch.float64).sum()
            should_count = count_from_token_idx is None or token_idx >= count_from_token_idx
            if should_count:
                access_count = torch.tensor(float(token_selected.numel()), device=device, dtype=torch.float64)
                total_overlap = total_overlap + hit_count
                total_hit_rate = total_hit_rate + hit_count / access_count
                total_access = total_access + access_count
                total_miss = total_miss + (access_count - hit_count)
                counted_steps += 1
            if future is not None:
                cache = _update_inference_cache(
                    cache=cache,
                    selected_experts=token_selected,
                    future_scores=future[token_idx],
                    cache_size=cache_size,
                    current_scores=current[token_idx] if current is not None else None,
                    cache_moe_config=cache_moe_config,
                )
            else:
                for expert in token_selected:
                    cache = touch_lru(cache, expert)

    step_count = torch.tensor(float(counted_steps), device=device, dtype=torch.float64)
    return {
        "hard_hit_rate_sum": total_hit_rate,
        "hard_overlap_sum": total_overlap,
        "hard_step_count": step_count,
        "hard_access_count": total_access,
        "hard_miss_count": total_miss,
    }


def generation_temporal_option_working_set_cache_metrics(
    grouped: Dict[int, Dict[str, List[torch.Tensor]]],
    cache_size: int,
    device: torch.device,
    count_from_token_idx: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Evaluate Temporal Option as a resident expert working set.

    This metric is intentionally only used for Temporal Option with
    --cache-policy auto. The option set itself is treated as the layer cache:
    a non-switch token reuses the previous option set, while a switch loads
    exactly the experts in new_option - old_option. Routed experts are selected
    from the active option set, so demand misses are measured against that active
    set. Switch churn is counted separately as prefetch/switch load so raw hit
    rate and load-adjusted hit rate remain distinguishable.
    """

    zero = torch.zeros((), device=device, dtype=torch.float64)
    if not grouped:
        return {
            "hard_hit_rate_sum": zero,
            "hard_overlap_sum": zero,
            "hard_step_count": zero,
            "hard_access_count": zero,
            "hard_miss_count": zero,
            "hard_demand_miss_count": zero,
            "hard_prefetch_load_count": zero,
        }

    total_hit_rate = zero.clone()
    total_overlap = zero.clone()
    total_access = zero.clone()
    total_miss = zero.clone()
    total_demand_miss = zero.clone()
    total_prefetch_load = zero.clone()
    counted_steps = 0

    for layer_idx, group in grouped.items():
        selected_chunks = group.get("selected", [])
        option_chunks = group.get("option", [])
        switch_chunks = group.get("switch", [])
        if not selected_chunks or not option_chunks or not switch_chunks:
            raise ValueError(
                "Temporal Option working-set cache metrics require selected_experts, option_masks, and switch_flags "
                f"for every recorded layer; missing them at layer_idx={layer_idx}."
            )
        if len(selected_chunks) != len(option_chunks) or len(option_chunks) != len(switch_chunks):
            raise ValueError(
                "Recorded Temporal Option stats have mismatched selected_experts, option_mask, and switch_flag chunks "
                f"at layer_idx={layer_idx}."
            )

        selected = torch.cat(selected_chunks, dim=1)[0].to(device=device, dtype=torch.long)
        option_masks = torch.cat(option_chunks, dim=1)[0].to(device=device, dtype=torch.bool)
        switch_flags = torch.cat(switch_chunks, dim=1)[0].to(device=device, dtype=torch.float32)
        max_tokens = min(int(selected.size(0)), int(option_masks.size(0)), int(switch_flags.size(0)))
        if max_tokens <= 0:
            continue

        prev_option = torch.zeros(option_masks.size(-1), device=device, dtype=torch.bool)
        has_prev = False
        for token_idx in range(max_tokens):
            current_option = option_masks[token_idx]
            selected_access = torch.tensor(
                float(selected[token_idx].numel()),
                device=device,
                dtype=torch.float64,
            )
            if selected_access.item() <= 0.0:
                continue

            should_switch = (not has_prev) or bool(switch_flags[token_idx].item() >= 0.5)
            if should_switch:
                load_count = (current_option & ~prev_option).to(dtype=torch.float64).sum()
                prev_option = current_option
                has_prev = True
            else:
                load_count = zero.clone()

            should_count = count_from_token_idx is None or token_idx >= count_from_token_idx
            if not should_count:
                continue

            token_selected = selected[token_idx]
            hit_count = current_option[token_selected].to(dtype=torch.float64).sum()
            demand_miss_count = selected_access - hit_count
            total_overlap = total_overlap + hit_count
            total_hit_rate = total_hit_rate + hit_count / selected_access
            total_access = total_access + selected_access
            total_demand_miss = total_demand_miss + demand_miss_count
            total_prefetch_load = total_prefetch_load + load_count
            total_miss = total_miss + demand_miss_count + load_count
            counted_steps += 1

    step_count = torch.tensor(float(counted_steps), device=device, dtype=torch.float64)
    return {
        "hard_hit_rate_sum": total_hit_rate,
        "hard_overlap_sum": total_overlap,
        "hard_step_count": step_count,
        "hard_access_count": total_access,
        "hard_miss_count": total_miss,
        "hard_demand_miss_count": total_demand_miss,
        "hard_prefetch_load_count": total_prefetch_load,
    }


def generation_spatio_temporal_hard_cache_metrics(
    grouped: Dict[int, Dict[str, List[torch.Tensor]]],
    cache_size: int,
    device: torch.device,
    count_from_token_idx: Optional[int] = None,
    refine_mode: str = "topk",
    refine_budget: int = -1,
    spatio_only: bool = False,
) -> Dict[str, torch.Tensor]:
    zero = torch.zeros((), device=device, dtype=torch.float64)
    if not grouped:
        return {
            "hard_hit_rate_sum": zero,
            "hard_overlap_sum": zero,
            "hard_step_count": zero,
            "hard_access_count": zero,
            "hard_miss_count": zero,
            "hard_demand_miss_count": zero,
            "hard_prefetch_load_count": zero,
        }

    layer_ids = sorted(grouped)
    layer_count = max(layer_ids) + 1
    if len(layer_ids) != layer_count:
        raise ValueError("Spatio-temporal Future Router expects contiguous MoE layer_idx values.")

    selected_sequences: Dict[int, torch.Tensor] = {}
    current_sequences: Dict[int, torch.Tensor] = {}
    temporal_sequences: Dict[int, torch.Tensor] = {}
    spatio_sequences: Dict[int, torch.Tensor] = {}
    for layer_idx in layer_ids:
        group = grouped[layer_idx]
        selected_chunks = group.get("selected", [])
        current_chunks = group.get("current", [])
        temporal_chunks = group.get("future", [])
        spatio_chunks = group.get("spatio", [])
        missing_temporal = not spatio_only and not temporal_chunks
        if not selected_chunks or not current_chunks or missing_temporal or not spatio_chunks:
            raise ValueError(
                "Spatial cache-router evaluation requires selected_experts, current_probs, and "
                "spatio_probs for every MoE layer; spatio-temporal mode also requires future_probs."
            )
        if (
            len(current_chunks) != len(selected_chunks)
            or len(spatio_chunks) != len(selected_chunks)
            or (not spatio_only and len(temporal_chunks) != len(selected_chunks))
        ):
            raise ValueError("Recorded spatial cache-router layer stats have mismatched chunk counts.")
        selected_sequences[layer_idx] = torch.cat(selected_chunks, dim=1)[0].to(device=device, dtype=torch.long)
        current_sequences[layer_idx] = torch.cat(current_chunks, dim=1)[0].to(device=device, dtype=torch.float32)
        if temporal_chunks:
            temporal_sequences[layer_idx] = torch.cat(temporal_chunks, dim=1)[0].to(
                device=device,
                dtype=torch.float32,
            )
        spatio_sequences[layer_idx] = torch.cat(spatio_chunks, dim=1)[0].to(device=device, dtype=torch.float32)

    total_hit_rate = zero.clone()
    total_overlap = zero.clone()
    total_access = zero.clone()
    total_miss = zero.clone()
    total_demand_miss = zero.clone()
    total_prefetch_load = zero.clone()
    counted_steps = 0
    refine_mode = str(refine_mode or "topk").lower()
    if refine_mode not in {"topk", "replace_lowest"}:
        raise ValueError(
            f"Unsupported spatio-temporal refine mode: {refine_mode!r}. "
            "Expected 'topk' or 'replace_lowest'."
        )

    def refine_cache(cache: torch.Tensor, priority_scores: torch.Tensor) -> torch.Tensor:
        if refine_mode == "topk":
            return _top_priority_cache(priority_scores, cache_size=cache_size)

        capacity = max(0, min(int(cache_size), int(priority_scores.size(-1))))
        if capacity <= 0:
            return torch.empty(0, device=priority_scores.device, dtype=torch.long)

        cache = cache[:capacity]
        budget = int(refine_budget)
        if budget < 0:
            budget = capacity
        budget = max(0, min(budget, int(priority_scores.size(-1)), capacity))
        if budget <= 0:
            return cache

        # Conditional top-K insertion: only full-priority top-K experts are allowed
        # to enter during this second-stage refinement. If a top-K expert is already
        # resident, it causes no load; missing top-K experts evict the lowest-scored
        # current residents one by one.
        candidates = torch.topk(priority_scores.detach(), k=budget, largest=True).indices.to(
            device=cache.device, dtype=cache.dtype
        )
        for candidate in candidates:
            if bool(_contains_any(candidate.reshape(1), cache).item()):
                continue
            if cache.numel() < capacity:
                cache = torch.cat([cache, candidate.reshape(1)], dim=0)
                continue
            if cache.numel() == 0:
                cache = candidate.reshape(1)
                continue
            resident_scores = priority_scores[cache]
            victim_pos = torch.argmin(resident_scores)
            if priority_scores[candidate] <= resident_scores[victim_pos]:
                continue
            cache = cache.clone()
            cache[victim_pos] = candidate
        return cache[:capacity]

    last_layer_idx = layer_ids[-1]
    for layer_idx in layer_ids:
        selected = selected_sequences[layer_idx]
        current = current_sequences[layer_idx]
        temporal = temporal_sequences.get(layer_idx)
        spatio_source = spatio_sequences[last_layer_idx if layer_idx == 0 else layer_idx - 1]
        sequence_lengths = [
            int(selected.size(0)),
            int(current.size(0)),
            int(spatio_source.size(0)),
        ]
        if temporal is not None:
            sequence_lengths.append(int(temporal.size(0)))
        max_tokens = min(sequence_lengths)
        if max_tokens <= 1:
            continue

        def carry_priority_after_token(token_idx: int) -> torch.Tensor:
            if spatio_only:
                return current[token_idx]
            return current[token_idx] + temporal[token_idx]

        def priority_for_target(token_idx: int) -> torch.Tensor:
            # Target is (token_idx, layer_idx), token_idx >= 1.
            carry_priority = carry_priority_after_token(token_idx - 1)
            if layer_idx == 0:
                spatio_priority = spatio_source[token_idx - 1]
            else:
                spatio_priority = spatio_source[token_idx]
            return carry_priority + spatio_priority

        cache = _initial_inference_cache(
            selected_experts=selected[0],
            future_scores=carry_priority_after_token(0),
            cache_size=cache_size,
            cache_moe_config=None,
        )
        for token_idx in range(1, max_tokens):
            cache_before_refine = cache
            cache = refine_cache(cache, priority_for_target(token_idx))
            prefetch_load_count = (~_contains_any(cache, cache_before_refine)).to(dtype=torch.float64).sum()
            token_selected = selected[token_idx]
            hit_count = _contains_any(token_selected, cache).to(dtype=torch.float64).sum()
            should_count = count_from_token_idx is None or token_idx >= count_from_token_idx
            if should_count:
                access_count = torch.tensor(float(token_selected.numel()), device=device, dtype=torch.float64)
                demand_miss_count = access_count - hit_count
                total_overlap = total_overlap + hit_count
                total_hit_rate = total_hit_rate + hit_count / access_count
                total_access = total_access + access_count
                total_demand_miss = total_demand_miss + demand_miss_count
                total_prefetch_load = total_prefetch_load + prefetch_load_count
                total_miss = total_miss + demand_miss_count + prefetch_load_count
                counted_steps += 1

            if not spatio_only and token_idx + 1 < max_tokens:
                cache = _update_inference_cache(
                    cache=cache,
                    selected_experts=token_selected,
                    future_scores=carry_priority_after_token(token_idx),
                    cache_size=cache_size,
                    cache_moe_config=None,
                )

    return {
        "hard_hit_rate_sum": total_hit_rate,
        "hard_overlap_sum": total_overlap,
        "hard_step_count": torch.tensor(float(counted_steps), device=device, dtype=torch.float64),
        "hard_access_count": total_access,
        "hard_miss_count": total_miss,
        "hard_demand_miss_count": total_demand_miss,
        "hard_prefetch_load_count": total_prefetch_load,
    }


def evaluate_hard_cache_metrics(
    model,
    cache_size: int,
    device: torch.device,
    cache_policy_config: CachePolicyConfig,
    count_from_token_idx: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    validate_future_only_cache_policy_binding(cache_policy_config)
    if cache_policy_config.policy == "auto":
        return generation_hard_cache_metrics(
            model,
            cache_size=cache_size,
            device=device,
            count_from_token_idx=count_from_token_idx,
            spatio_temporal_refine_mode=cache_policy_config.spatio_temporal_refine_mode,
            spatio_temporal_refine_budget=cache_policy_config.spatio_temporal_refine_budget,
        )
    return simulate_cache_policy_metrics(
        get_recorded_layer_stats(model),
        cache_size=cache_size,
        device=device,
        config=cache_policy_config,
        count_from_token_idx=count_from_token_idx,
    )


def evaluate_hard_cache_metrics_pair(
    model,
    cache_size: int,
    device: torch.device,
    cache_policy_config: CachePolicyConfig,
    decode_count_from_token_idx: Optional[int],
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    validate_future_only_cache_policy_binding(cache_policy_config)
    if cache_policy_config.policy == "auto":
        return (
            evaluate_hard_cache_metrics(
                model,
                cache_size=cache_size,
                device=device,
                cache_policy_config=cache_policy_config,
            ),
            evaluate_hard_cache_metrics(
                model,
                cache_size=cache_size,
                device=device,
                cache_policy_config=cache_policy_config,
                count_from_token_idx=decode_count_from_token_idx,
            ),
        )
    return simulate_cache_policy_metrics_pair(
        get_recorded_layer_stats(model),
        cache_size=cache_size,
        device=device,
        config=cache_policy_config,
        decode_count_from_token_idx=decode_count_from_token_idx,
    )


def hard_cache_metric_mode(model, cache_policy_config: CachePolicyConfig) -> str:
    cache_model = _unwrap_cache_model(model)
    if cache_policy_config.policy == "auto" and getattr(cache_model, "_temporal_option_moe_enabled", False):
        return "temporal_option_working_set"
    cache_moe_config = getattr(cache_model, "_cache_moe_config", None)
    if (
        cache_policy_config.policy == "auto"
        and cache_moe_config is not None
        and getattr(cache_moe_config, "spatio_only_enabled", False)
    ):
        return "spatio_only_prefetch_replay"
    if cache_policy_config.policy == "auto":
        return "selected_expert_replay"
    return f"{cache_policy_config.policy}_replay"


def _zero_metric_like(metric_sums: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=metric_sums.device, dtype=torch.float64)


def accumulate_cache_metric_pair(
    metric_sums: torch.Tensor,
    hard_metrics: Dict[str, torch.Tensor],
    decode_hard_metrics: Dict[str, torch.Tensor],
) -> None:
    zero = _zero_metric_like(metric_sums)
    metric_sums[2] += hard_metrics["hard_hit_rate_sum"]
    metric_sums[3] += hard_metrics["hard_overlap_sum"]
    metric_sums[4] += hard_metrics["hard_step_count"]
    metric_sums[8] += hard_metrics["hard_access_count"]
    metric_sums[9] += hard_metrics["hard_miss_count"]
    metric_sums[10] += decode_hard_metrics["hard_hit_rate_sum"]
    metric_sums[11] += decode_hard_metrics["hard_overlap_sum"]
    metric_sums[12] += decode_hard_metrics["hard_step_count"]
    metric_sums[13] += decode_hard_metrics["hard_access_count"]
    metric_sums[14] += decode_hard_metrics["hard_miss_count"]
    metric_sums[15] += hard_metrics.get("hard_demand_miss_count", hard_metrics["hard_miss_count"])
    metric_sums[16] += hard_metrics.get("hard_prefetch_load_count", zero)
    metric_sums[17] += decode_hard_metrics.get(
        "hard_demand_miss_count",
        decode_hard_metrics["hard_miss_count"],
    )
    metric_sums[18] += decode_hard_metrics.get("hard_prefetch_load_count", zero)


def hard_summary_from_metric_sums(
    metric_sums: torch.Tensor,
    *,
    hit_sum_idx: int,
    overlap_sum_idx: int,
    step_count_idx: int,
    access_count_idx: int,
    miss_count_idx: int,
    prefetch_load_count_idx: int,
) -> Dict[str, float]:
    hard_step_count = metric_sums[step_count_idx].clamp_min(1.0)
    hard_hit_rate = metric_sums[hit_sum_idx] / hard_step_count
    summary = {
        "hit_rate": float(hard_hit_rate.cpu().item()),
        "miss_rate": float((1.0 - hard_hit_rate).cpu().item()),
        "overlap_count": float((metric_sums[overlap_sum_idx] / hard_step_count).cpu().item()),
        "counted_steps": float(metric_sums[step_count_idx].cpu().item()),
        "access_count": float(metric_sums[access_count_idx].cpu().item()),
        "miss_count": float(metric_sums[miss_count_idx].cpu().item()),
        "prefetch_load_count": float(metric_sums[prefetch_load_count_idx].cpu().item()),
    }
    summary["load_adjusted_access_count"] = summary["access_count"] + summary["prefetch_load_count"]
    summary["load_adjusted_miss_rate"] = (
        summary["miss_count"] / summary["load_adjusted_access_count"]
        if summary["load_adjusted_access_count"] > 0.0
        else 0.0
    )
    summary["load_adjusted_hit_rate"] = 1.0 - summary["load_adjusted_miss_rate"]
    return summary


def summarize_hard_cache_metrics(hard_metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    step_count = float(hard_metrics["hard_step_count"].detach().cpu().item())
    access_count = float(hard_metrics["hard_access_count"].detach().cpu().item())
    miss_count = float(hard_metrics["hard_miss_count"].detach().cpu().item())
    prefetch_load_count = float(
        hard_metrics.get(
            "hard_prefetch_load_count",
            torch.zeros((), device=hard_metrics["hard_access_count"].device, dtype=torch.float64),
        )
        .detach()
        .cpu()
        .item()
    )
    load_adjusted_access_count = access_count + prefetch_load_count
    load_adjusted_miss_rate = (
        miss_count / load_adjusted_access_count if load_adjusted_access_count > 0.0 else 0.0
    )
    load_adjusted_hit_rate = 1.0 - load_adjusted_miss_rate
    if step_count <= 0:
        return {
            "hit_rate": 0.0,
            "miss_rate": 0.0,
            "load_adjusted_hit_rate": 0.0,
            "load_adjusted_miss_rate": 0.0,
            "overlap_count": 0.0,
            "counted_steps": 0.0,
            "access_count": 0.0,
            "miss_count": 0.0,
            "prefetch_load_count": 0.0,
            "load_adjusted_access_count": 0.0,
        }
    if hard_metrics.get("hard_hit_rate_mode") == "aggregate_access":
        hit_rate = (access_count - miss_count) / access_count if access_count > 0.0 else 0.0
        overlap_count = (access_count - miss_count) / step_count if step_count > 0.0 else 0.0
    else:
        hit_rate = float((hard_metrics["hard_hit_rate_sum"] / hard_metrics["hard_step_count"]).detach().cpu().item())
        overlap_count = float(
            (hard_metrics["hard_overlap_sum"] / hard_metrics["hard_step_count"]).detach().cpu().item()
        )
    return {
        "hit_rate": hit_rate,
        "miss_rate": 1.0 - hit_rate,
        "load_adjusted_hit_rate": load_adjusted_hit_rate,
        "load_adjusted_miss_rate": load_adjusted_miss_rate,
        "overlap_count": overlap_count,
        "counted_steps": step_count,
        "access_count": access_count,
        "miss_count": miss_count,
        "prefetch_load_count": prefetch_load_count,
        "load_adjusted_access_count": load_adjusted_access_count,
    }


def expert_load_summary(
    hard_summary: Dict[str, float],
    generated_token_count: float,
    expert_size_bytes: int,
    bandwidth_gbps: float = 0.0,
) -> Dict[str, float]:
    expert_size_mb = float(expert_size_bytes) / 1_000_000.0 if expert_size_bytes > 0 else 0.0
    load_mb = hard_summary["miss_count"] * expert_size_mb
    token_denom = max(float(generated_token_count), 1.0)
    load_mb_per_generated_token = load_mb / token_denom
    result = {
        "expert_size_MB": expert_size_mb,
        "expert_load_MB": load_mb,
        "load_MB_per_generated_token": load_mb_per_generated_token,
        "expert_misses_per_generated_token": hard_summary["miss_count"] / token_denom,
    }
    if bandwidth_gbps > 0.0:
        # Decimal units: 1 GB/s = 1000 MB/s, so MB/token divided by GB/s gives ms/token.
        result["estimated_load_TPOT_ms_per_generated_token"] = load_mb_per_generated_token / bandwidth_gbps
        result["expert_load_bandwidth_GBps"] = bandwidth_gbps
    return result


def effective_cache_policy_prefetch_budget(
    cache_policy_config: CachePolicyConfig,
    cache_size: int,
) -> int:
    if cache_policy_config.policy == "promoe":
        return infer_promoe_prefetch_budget(cache_policy_config, cache_size=cache_size)
    if cache_policy_config.policy == "finemoe":
        return infer_finemoe_prefetch_budget(cache_policy_config, cache_size=cache_size)
    if cache_policy_config.policy in {"specmd_prefetch", "specmd"}:
        return infer_specmd_prefetch_budget(cache_policy_config, cache_size=cache_size)
    return int(cache_policy_config.prefetch_budget)


def summarize_cache_policy_metric_sums(
    *,
    policy: str,
    metric_sums: torch.Tensor,
    metric_mode: str,
    cache_policy_config: CachePolicyConfig,
    cache_size: int,
    total: int,
    correct: int,
    generated_token_count: float,
    num_moe_layers: int,
    expert_size_bytes: int,
    expert_load_bandwidth_gbps: float,
) -> Dict[str, Any]:
    full_hard_summary = hard_summary_from_metric_sums(
        metric_sums,
        hit_sum_idx=2,
        overlap_sum_idx=3,
        step_count_idx=4,
        access_count_idx=8,
        miss_count_idx=9,
        prefetch_load_count_idx=16,
    )
    decode_hard_summary = hard_summary_from_metric_sums(
        metric_sums,
        hit_sum_idx=10,
        overlap_sum_idx=11,
        step_count_idx=12,
        access_count_idx=13,
        miss_count_idx=14,
        prefetch_load_count_idx=18,
    )
    full_load_summary = expert_load_summary(
        full_hard_summary,
        generated_token_count,
        expert_size_bytes,
        bandwidth_gbps=expert_load_bandwidth_gbps,
    )
    decode_output_token_count = decode_hard_summary["counted_steps"] / float(max(num_moe_layers, 1))
    decode_load_summary = expert_load_summary(
        decode_hard_summary,
        decode_output_token_count,
        expert_size_bytes,
        bandwidth_gbps=expert_load_bandwidth_gbps,
    )
    summary: Dict[str, Any] = {
        "cache_policy": policy,
        "hard_cache_metric_mode": metric_mode,
        "cache_size": cache_size,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "hard_cache_hit_rate": full_hard_summary["hit_rate"],
        "hard_cache_miss_rate": full_hard_summary["miss_rate"],
        "load_adjusted_hard_cache_hit_rate": full_hard_summary["load_adjusted_hit_rate"],
        "load_adjusted_hard_cache_miss_rate": full_hard_summary["load_adjusted_miss_rate"],
        "hard_cache_overlap_count": full_hard_summary["overlap_count"],
        "hard_cache_counted_steps": int(full_hard_summary["counted_steps"]),
        "hard_cache_access_count": int(full_hard_summary["access_count"]),
        "hard_cache_miss_count": full_hard_summary["miss_count"],
        "hard_cache_demand_miss_count": float(metric_sums[15].cpu().item()),
        "hard_cache_prefetch_load_count": full_hard_summary["prefetch_load_count"],
        "decode_hard_cache_hit_rate": decode_hard_summary["hit_rate"],
        "decode_hard_cache_miss_rate": decode_hard_summary["miss_rate"],
        "decode_load_adjusted_hard_cache_hit_rate": decode_hard_summary["load_adjusted_hit_rate"],
        "decode_load_adjusted_hard_cache_miss_rate": decode_hard_summary["load_adjusted_miss_rate"],
        "decode_hard_cache_overlap_count": decode_hard_summary["overlap_count"],
        "decode_hard_cache_counted_steps": int(decode_hard_summary["counted_steps"]),
        "decode_output_token_count": int(round(decode_output_token_count)),
        "decode_hard_cache_access_count": int(decode_hard_summary["access_count"]),
        "decode_hard_cache_miss_count": decode_hard_summary["miss_count"],
        "decode_hard_cache_demand_miss_count": float(metric_sums[17].cpu().item()),
        "decode_hard_cache_prefetch_load_count": decode_hard_summary["prefetch_load_count"],
        "expert_size_MB": decode_load_summary["expert_size_MB"],
        "load_MB_per_generated_token": decode_load_summary["load_MB_per_generated_token"],
        "decode_load_MB_per_generated_token": decode_load_summary["load_MB_per_generated_token"],
        "decode_load_MB_per_decode_token": decode_load_summary["load_MB_per_generated_token"],
        "decode_expert_load_MB": decode_load_summary["expert_load_MB"],
        "decode_expert_misses_per_generated_token": decode_load_summary["expert_misses_per_generated_token"],
        "decode_expert_misses_per_decode_token": decode_load_summary["expert_misses_per_generated_token"],
        "full_request_load_MB_per_generated_token": full_load_summary["load_MB_per_generated_token"],
        "full_request_expert_load_MB": full_load_summary["expert_load_MB"],
        "full_request_expert_misses_per_generated_token": full_load_summary["expert_misses_per_generated_token"],
        "cache_policy_recency_weight": cache_policy_config.recency_weight,
        "cache_policy_frequency_weight": cache_policy_config.frequency_weight,
        "cache_policy_recency_decay": cache_policy_config.recency_decay,
        "cache_policy_lrfu_lambda": cache_policy_config.lrfu_lambda,
        "cache_policy_prefetch_budget": cache_policy_config.prefetch_budget,
        "cache_policy_effective_prefetch_budget": effective_cache_policy_prefetch_budget(
            cache_policy_config,
            cache_size=cache_size,
        ),
        "cache_policy_prefetch_lookahead": cache_policy_config.prefetch_lookahead,
        "future_cache_priority_mode": cache_policy_config.future_cache_priority_mode,
        "spatio_temporal_refine_mode": cache_policy_config.spatio_temporal_refine_mode,
        "spatio_temporal_refine_budget": cache_policy_config.spatio_temporal_refine_budget,
        "finemoe_top_k": cache_policy_config.finemoe_top_k,
        "finemoe_candidate_pool": cache_policy_config.finemoe_candidate_pool,
        "finemoe_semantic_weight": cache_policy_config.finemoe_semantic_weight,
        "finemoe_trajectory_weight": cache_policy_config.finemoe_trajectory_weight,
        "finemoe_prefetch_threshold": cache_policy_config.finemoe_prefetch_threshold,
        "finemoe_min_prefetch_threshold": cache_policy_config.finemoe_min_prefetch_threshold,
        "finemoe_max_prefetch_threshold": cache_policy_config.finemoe_max_prefetch_threshold,
        "finemoe_eviction_probability_weight": cache_policy_config.finemoe_eviction_probability_weight,
    }
    if expert_load_bandwidth_gbps > 0.0:
        summary.update(
            {
                "expert_load_bandwidth_GBps": expert_load_bandwidth_gbps,
                "estimated_load_TPOT_ms_per_generated_token": decode_load_summary[
                    "estimated_load_TPOT_ms_per_generated_token"
                ],
                "decode_estimated_load_TPOT_ms_per_generated_token": decode_load_summary[
                    "estimated_load_TPOT_ms_per_generated_token"
                ],
                "full_request_estimated_load_TPOT_ms_per_generated_token": full_load_summary[
                    "estimated_load_TPOT_ms_per_generated_token"
                ],
            }
        )
    return summary


def predictor_overhead_summary(
    future_router_parameter_count: int,
    effective_tflops: float = 0.0,
    dtype_bytes: int = 2,
    external_predictor: Optional[Dict[str, Any]] = None,
    external_predictor_dtype_bytes: int = 4,
    num_moe_layers: int = 0,
    prefetch_lookahead: int = 3,
) -> Dict[str, float]:
    future_params = int(future_router_parameter_count or 0)
    external_params = promoe_predictor_parameter_count(external_predictor)
    external_model_count = promoe_predictor_model_count(external_predictor)
    external_hidden_dim = 0
    external_input_dim = 0
    external_num_experts = 0
    external_num_predict = 0
    external_models_invoked_per_token = 0.0
    external_macs_per_token = 0.0
    if external_predictor is not None:
        predictor_config = external_predictor.get("config", {})
        external_hidden_dim = int(predictor_config.get("hidden_dim", 0) or 0)
        external_input_dim = int(predictor_config.get("input_dim", 0) or 0)
        external_num_experts = int(predictor_config.get("num_experts", 0) or 0)
        external_num_predict = int(predictor_config.get("num_predict_expert_per_layer", 0) or 0)
        mapping_layers = int(predictor_config.get("num_layers", 0) or num_moe_layers or 0)
        if mapping_layers > 0:
            mapping = build_promoe_layer_mapping(
                num_layers=mapping_layers,
                interval=int(predictor_config.get("layer_predict_interval", 1) or 1),
                max_window=int(predictor_config.get("layer_predict_max_window", prefetch_lookahead) or prefetch_lookahead),
                replace_first_input_with_last_output=bool(
                    predictor_config.get("layer_predict_replace_first_input_with_last_output", True)
                ),
            )
            external_models_invoked_per_token = float(sum(len(targets) for targets in mapping.values()))
        elif external_model_count > 0:
            external_models_invoked_per_token = float(external_model_count)
        external_macs_per_token = external_models_invoked_per_token * float(
            promoe_predictor_macs_per_model(external_predictor)
        )
    params = future_params + external_params

    # A future_gate linear projection reads each weight once per decode token.
    # Count both MACs and 2-FLOP multiply-adds so papers can choose either convention.
    future_macs_per_token = float(future_params)
    macs_per_token = future_macs_per_token + external_macs_per_token
    flops_per_token = 2.0 * macs_per_token
    summary = {
        "predictor_overhead_params": params,
        "predictor_overhead_storage_MB": (
            future_params * dtype_bytes + external_params * external_predictor_dtype_bytes
        )
        / 1_000_000.0,
        "predictor_overhead_MACs_per_token": macs_per_token,
        "predictor_overhead_FLOPs_per_token": flops_per_token,
        "predictor_overhead_ops_per_token": flops_per_token,
        "predictor_overhead_ms_per_token": None,
        "future_router_predictor_params": future_params,
        "external_predictor_params": external_params,
        "external_predictor_model_count": external_model_count,
        "external_predictor_hidden_dim": external_hidden_dim,
        "external_predictor_input_dim": external_input_dim,
        "external_predictor_num_experts": external_num_experts,
        "external_predictor_num_predict_expert_per_layer": external_num_predict,
        "external_predictor_models_invoked_per_token": external_models_invoked_per_token,
        "external_predictor_MACs_per_token": external_macs_per_token,
        "external_predictor_FLOPs_per_token": 2.0 * external_macs_per_token,
        "external_predictor_ops_per_token": 2.0 * external_macs_per_token,
    }
    if effective_tflops > 0.0:
        summary["predictor_overhead_ms_per_token"] = (
            flops_per_token / (effective_tflops * 1_000_000_000_000.0) * 1000.0
        )
        summary["predictor_effective_TFLOPs"] = effective_tflops
    return summary


def emphasize_paper_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    highlighted = {
        "*paper_cache_policy": metrics.get("cache_policy"),
        "*paper_cache_size": metrics.get("cache_size"),
        "*paper_accuracy": metrics.get("accuracy"),
        "*paper_decode_hard_cache_hit_rate": metrics.get("decode_hard_cache_hit_rate"),
        "*paper_decode_load_adjusted_hard_cache_hit_rate": metrics.get(
            "decode_load_adjusted_hard_cache_hit_rate"
        ),
        "*paper_decode_load_MB_per_decode_token": metrics.get("decode_load_MB_per_decode_token"),
        "*paper_predictor_overhead_params": metrics.get("predictor_overhead_params"),
        "*paper_predictor_overhead_ms_per_token": metrics.get("predictor_overhead_ms_per_token"),
    }
    if "decode_estimated_load_TPOT_ms_per_generated_token" in metrics:
        highlighted["*paper_decode_estimated_load_TPOT_ms_per_token"] = metrics.get(
            "decode_estimated_load_TPOT_ms_per_generated_token"
        )
    highlighted["*paper_note"] = (
        "Core paper metrics: quality, decode cache hit rate, decode expert-load MB/token, "
        "predictor overhead, and optional load-only TPOT proxy."
    )
    return {**highlighted, **metrics}


def save_expert_usage(
    output_path: Path,
    usage_counts: torch.Tensor,
    token_counts: torch.Tensor,
    routed_slot_counts: torch.Tensor,
    sample_count: int,
    model_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_sample_count = max(sample_count, 1)
    safe_routed_slots = routed_slot_counts.clamp_min(1.0).unsqueeze(-1)
    payload = {
        "model_name": model_name,
        "sample_count": int(sample_count),
        "num_layers": int(usage_counts.size(0)),
        "num_experts": int(usage_counts.size(1)),
        "usage_counts": usage_counts.cpu(),
        "avg_counts_per_sample": (usage_counts / safe_sample_count).cpu(),
        "frequency": (usage_counts / safe_routed_slots).cpu(),
        "token_counts": token_counts.cpu(),
        "routed_slot_counts": routed_slot_counts.cpu(),
    }
    torch.save(payload, output_path)


def select_eval_examples(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    dataset_cfg = config["dataset"]
    raw = _load_raw_dataset(dataset_cfg["name"], dataset_cfg.get("subset"))
    split_name = dataset_cfg.get("eval_split") or "test"
    dataset = raw[split_name]
    max_eval_samples = dataset_cfg.get("max_eval_samples")
    if max_eval_samples:
        dataset = dataset.select(range(min(int(max_eval_samples), len(dataset))))
    return [dict(example) for example in dataset]


def render_prompt(config: Dict[str, Any], tokenizer, example: Dict[str, Any]) -> str:
    dataset_cfg = config["dataset"]
    full_messages = _build_messages(
        example,
        dataset_name=dataset_cfg["name"],
        system_prompt=dataset_cfg["system_prompt"],
        dataset_format=dataset_cfg.get("format", "auto"),
        messages_field=dataset_cfg.get("messages_field", "messages"),
        prepend_system_prompt=dataset_cfg.get("prepend_system_prompt", False),
    )
    prompt_messages = [message for message in full_messages if message["role"] != "assistant"]
    return _render_messages(tokenizer, prompt_messages, add_generation_prompt=True)


def tokenize_reference_example(config: Dict[str, Any], tokenizer, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    dataset_cfg = config["dataset"]
    tokenized = _tokenize_example(
        example,
        tokenizer,
        max_length=int(dataset_cfg["max_length"]),
        system_prompt=dataset_cfg["system_prompt"],
        dataset_name=dataset_cfg["name"],
        dataset_format=dataset_cfg.get("format", "auto"),
        messages_field=dataset_cfg.get("messages_field", "messages"),
        prepend_system_prompt=dataset_cfg.get("prepend_system_prompt", False),
        label_mask_strategy=dataset_cfg.get("label_mask_strategy", "token_length"),
    )
    if not tokenized["input_ids"]:
        raise ValueError(f"Tokenized reference example has no supervised labels: {example!r}")
    return {
        "input_ids": torch.tensor([tokenized["input_ids"]], dtype=torch.long),
        "attention_mask": torch.tensor([tokenized["attention_mask"]], dtype=torch.long),
        "labels": torch.tensor([tokenized["labels"]], dtype=torch.long),
    }


def reference_moe_loss_sum(
    model,
    config: Dict[str, Any],
    tokenizer,
    example: Dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    reset_recorded_expert_stats(model)
    batch = tokenize_reference_example(config, tokenizer, example)
    batch = {key: value.to(device) for key, value in batch.items()}
    labels = batch["labels"]
    shifted_valid_labels = labels[..., 1:].ne(-100)
    valid_token_count = shifted_valid_labels.sum().to(dtype=torch.float64)
    if valid_token_count.item() <= 0:
        return (
            torch.zeros((), device=device, dtype=torch.float64),
            torch.zeros((), device=device, dtype=torch.float64),
            0.0,
        )

    with torch.no_grad():
        try:
            outputs = model(
                **batch,
                use_cache=False,
                output_router_logits=False,
                return_dict=True,
            )
        except TypeError:
            outputs = model(
                **batch,
                use_cache=False,
                return_dict=True,
            )
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
    loss_sum = loss.detach().to(dtype=torch.float64) * valid_token_count
    reset_recorded_expert_stats(model)
    return loss_sum, valid_token_count, float(loss.detach().cpu().item())


def main() -> None:
    args = parse_args()
    args.cache_policy = normalize_cache_policy_name(args.cache_policy)
    requested_refine_budgets = parse_spatio_temporal_refine_budget_list(
        args.spatio_temporal_refine_budgets
    )
    extra_cache_policy_names = [
        policy for policy in parse_cache_policy_list(args.extra_cache_policies) if policy != args.cache_policy
    ]
    config = load_config(args.config)
    if args.eval_split is not None:
        config["dataset"]["eval_split"] = args.eval_split
    if args.max_eval_samples is not None:
        config["dataset"]["max_eval_samples"] = args.max_eval_samples

    output_dir = Path(args.output_dir)
    predictions_output = Path(args.predictions_output) if args.predictions_output else output_dir / "predictions.jsonl"
    metrics_output = Path(args.metrics_output) if args.metrics_output else output_dir / "metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_checkpoint_load and args.model_path is None:
        raise ValueError("--model-path is required unless --skip-checkpoint-load is set.")
    checkpoint_path = None if args.skip_checkpoint_load else resolve_checkpoint_path(Path(args.model_path))
    tokenizer = build_tokenizer(config)
    model = build_model(config)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = not args.no_generation_cache
    disable_training_router_outputs_for_generation(model)

    trainer = MoESFTTrainer(
        model=model,
        args=build_eval_args(config, output_dir),
        data_collator=SupervisedDataCollator(pad_token_id=tokenizer.pad_token_id),
    )
    if checkpoint_path is not None:
        load_trained_weights(trainer, checkpoint_path)
    model = trainer.model

    examples = select_eval_examples(config)
    rank, world_size = distributed_info()
    future_router_summary = validate_future_router_loaded(model, config, rank)
    generation_device = resolve_generation_device()
    model, model_input_device = move_model_for_generation(model, generation_device, rank)
    disable_training_router_outputs_for_generation(model)
    trainer.model = model

    expert_usage_output = Path(args.expert_usage_output) if args.expert_usage_output else output_dir / "expert_usage.pt"
    num_moe_layers, num_experts = infer_moe_shape(model)
    usage_counts = torch.zeros((num_moe_layers, num_experts), device=generation_device, dtype=torch.float64)
    token_counts = torch.zeros((num_moe_layers,), device=generation_device, dtype=torch.float64)
    routed_slot_counts = torch.zeros((num_moe_layers,), device=generation_device, dtype=torch.float64)
    cache_size = resolve_cache_size(config)
    expert_size_bytes = infer_routed_expert_size_bytes(model, override_mb=args.expert_size_mb)
    promoe_predictor = load_promoe_predictor(args.promoe_predictor_path)
    finemoe_store = load_finemoe_store(args.finemoe_store_path)
    all_requested_cache_policies = [args.cache_policy] + extra_cache_policy_names
    if "promoe" in all_requested_cache_policies and promoe_predictor is None:
        raise ValueError("--cache-policy promoe requires --promoe-predictor-path.")
    if "finemoe" in all_requested_cache_policies and finemoe_store is None:
        raise ValueError("--cache-policy finemoe requires --finemoe-store-path.")
    cache_policy_config = build_cache_policy_config(
        args=args,
        config=config,
        policy=args.cache_policy,
        promoe_predictor=promoe_predictor,
        finemoe_store=finemoe_store,
    )
    extra_cache_policy_configs = {
        policy: build_cache_policy_config(
            args=args,
            config=config,
            policy=policy,
            promoe_predictor=promoe_predictor,
            finemoe_store=finemoe_store,
        )
        for policy in extra_cache_policy_names
    }
    refine_budget_metric_labels: Dict[int, str] = {}
    if requested_refine_budgets:
        cache_model = _unwrap_cache_model(model)
        cache_moe_config = getattr(cache_model, "_cache_moe_config", None)
        spatial_router_enabled = cache_moe_config is not None and (
            getattr(cache_moe_config, "spatio_temporal_enabled", False)
            or getattr(cache_moe_config, "spatio_only_enabled", False)
        )
        if not spatial_router_enabled:
            raise ValueError(
                "--spatio-temporal-refine-budgets only applies to spatio-temporal or spatio-only routers; "
                "the regular Future Router has no spatial replace_lowest stage."
            )
        if cache_policy_config.policy != "auto":
            raise ValueError("--spatio-temporal-refine-budgets requires --cache-policy auto.")
        if cache_policy_config.spatio_temporal_refine_mode != "replace_lowest":
            raise ValueError(
                "--spatio-temporal-refine-budgets requires --spatio-temporal-refine-mode replace_lowest."
            )
        primary_budget = requested_refine_budgets[0]
        if args.spatio_temporal_refine_budget not in {-1, primary_budget}:
            raise ValueError(
                "When both refine-budget options are provided, --spatio-temporal-refine-budget must "
                "match the first value in --spatio-temporal-refine-budgets."
            )
        cache_policy_config.spatio_temporal_refine_budget = primary_budget
        refine_budget_metric_labels[primary_budget] = cache_policy_config.policy
        for budget in requested_refine_budgets[1:]:
            label = f"auto_replace_lowest_budget_{budget}"
            if label in extra_cache_policy_configs:
                raise ValueError(f"Duplicate cache metric variant label: {label}")
            extra_cache_policy_configs[label] = replace(
                cache_policy_config,
                spatio_temporal_refine_budget=budget,
            )
            refine_budget_metric_labels[budget] = label
    validate_future_only_cache_policy_binding(cache_policy_config)
    for extra_policy_config in extra_cache_policy_configs.values():
        validate_future_only_cache_policy_binding(extra_policy_config)
    generation_metric_sums = torch.zeros(24, device=generation_device, dtype=torch.float64)
    extra_cache_policy_metric_sums = {
        policy: torch.zeros(24, device=generation_device, dtype=torch.float64)
        for policy in extra_cache_policy_configs
    }

    local_examples = [(idx, example) for idx, example in enumerate(examples) if idx % world_size == rank]
    local_predictions: List[Dict[str, Any]] = []
    local_promoe_traces: List[Dict[str, Any]] = []
    local_predictions_output = output_dir / f"predictions.rank{rank}.jsonl.tmp"
    local_predictions_output.parent.mkdir(parents=True, exist_ok=True)
    if local_predictions_output.exists():
        local_predictions_output.unlink()
    local_metrics_output = output_dir / f"metrics.rank{rank}.pt"
    if local_metrics_output.exists():
        local_metrics_output.unlink()
    for stale_tmp in output_dir.glob(f"{local_metrics_output.name}.tmp.*"):
        stale_tmp.unlink()
    print(
        f"[rank{rank}] Starting GSM8K generation: local_samples={len(local_examples)}, "
        f"world_size={world_size}, max_new_tokens={args.max_new_tokens}, "
        f"device={model_input_device}.",
        flush=True,
    )

    do_sample = args.temperature > 0
    start_time = time.time()
    for local_pos, (idx, example) in enumerate(local_examples, start=1):
        if args.skip_reference_loss:
            reference_loss_sum = torch.zeros((), device=generation_device, dtype=torch.float64)
            reference_token_count = torch.zeros((), device=generation_device, dtype=torch.float64)
            sample_reference_moe_loss = 0.0
        else:
            reference_loss_sum, reference_token_count, sample_reference_moe_loss = reference_moe_loss_sum(
                model=model,
                config=config,
                tokenizer=tokenizer,
                example=example,
                device=model_input_device,
            )
        generation_metric_sums[5] += reference_loss_sum.to(device=generation_device)
        generation_metric_sums[6] += reference_token_count.to(device=generation_device)

        reset_recorded_expert_stats(model)
        prompt = render_prompt(config, tokenizer, example)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(model_input_device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=not args.no_generation_cache,
                output_router_logits=False,
                synced_gpus=False,
                return_dict_in_generate=args.compute_generation_nll,
                output_scores=args.compute_generation_nll,
            )
        accumulate_expert_usage(model, usage_counts, token_counts, routed_slot_counts)
        if args.promoe_trace_output is not None:
            local_promoe_traces.append(build_promoe_trace_sample(get_recorded_layer_stats(model), idx))
        sequences = generated.sequences if args.compute_generation_nll else generated
        new_tokens = sequences[0, encoded["input_ids"].shape[-1] :]
        if args.compute_generation_nll:
            generation_nll_sum, generation_token_count = generation_nll_from_scores(generated.scores, new_tokens)
        else:
            generation_nll_sum = torch.zeros((), device=generation_device, dtype=torch.float64)
            generation_token_count = torch.zeros((), device=generation_device, dtype=torch.float64)
        prompt_token_count = int(encoded["input_ids"].shape[-1])
        hard_metrics, decode_hard_metrics = evaluate_hard_cache_metrics_pair(
            model,
            cache_size=cache_size,
            device=generation_device,
            cache_policy_config=cache_policy_config,
            decode_count_from_token_idx=prompt_token_count,
        )
        generation_metric_sums[0] += generation_nll_sum.to(device=generation_device)
        generation_metric_sums[1] += generation_token_count.to(device=generation_device)
        accumulate_cache_metric_pair(generation_metric_sums, hard_metrics, decode_hard_metrics)
        generation_metric_sums[7] += torch.tensor(float(new_tokens.numel()), device=generation_device, dtype=torch.float64)
        for extra_policy, extra_cache_policy_config in extra_cache_policy_configs.items():
            extra_hard_metrics, extra_decode_hard_metrics = evaluate_hard_cache_metrics_pair(
                model,
                cache_size=cache_size,
                device=generation_device,
                cache_policy_config=extra_cache_policy_config,
                decode_count_from_token_idx=prompt_token_count,
            )
            accumulate_cache_metric_pair(
                extra_cache_policy_metric_sums[extra_policy],
                extra_hard_metrics,
                extra_decode_hard_metrics,
            )
        temporal_option_summary = temporal_option_generation_summary(model, generation_device)
        generation_metric_sums[19:24] += temporal_option_summary
        sample_hard_steps = float(hard_metrics["hard_step_count"].cpu().item())
        sample_hard_summary = summarize_hard_cache_metrics(hard_metrics)
        sample_decode_hard_summary = summarize_hard_cache_metrics(decode_hard_metrics)
        temporal_option_count = float(temporal_option_summary[4].detach().cpu().item())
        sample_hard_hit_rate = sample_hard_summary["hit_rate"]
        sample_hard_overlap = sample_hard_summary["overlap_count"]
        sample_generation_tokens = float(generation_token_count.cpu().item())
        if not args.compute_generation_nll:
            sample_generation_tokens = float(new_tokens.numel())
        sample_generation_nll = (
            float((generation_nll_sum / generation_token_count).cpu().item())
            if args.compute_generation_nll and sample_generation_tokens > 0
            else 0.0
        )
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred_answer = extract_gsm8k_answer(generated_text)
        ref_answer = extract_gsm8k_answer(str(example["answer"]))
        prediction = {
            "index": idx,
            "question": example["question"],
            "reference_answer": ref_answer,
            "predicted_answer": pred_answer,
            "correct": exact_match(pred_answer, ref_answer),
            "reference_moe_loss": sample_reference_moe_loss,
            "reference_loss_token_count": int(reference_token_count.cpu().item()),
            "generated_token_count": int(sample_generation_tokens),
            "generation_hard_cache_hit_rate": sample_hard_hit_rate,
            "generation_hard_cache_miss_rate": 1.0 - sample_hard_hit_rate if sample_hard_steps > 0 else 0.0,
            "generation_hard_cache_overlap_count": sample_hard_overlap,
            "generation_hard_cache_counted_steps": int(sample_hard_steps),
            "decode_hard_cache_hit_rate": sample_decode_hard_summary["hit_rate"],
            "decode_hard_cache_miss_rate": sample_decode_hard_summary["miss_rate"],
            "decode_hard_cache_overlap_count": sample_decode_hard_summary["overlap_count"],
            "decode_hard_cache_counted_steps": int(sample_decode_hard_summary["counted_steps"]),
            "decode_expert_miss_count": sample_decode_hard_summary["miss_count"],
            "decode_expert_demand_miss_count": float(
                decode_hard_metrics.get("hard_demand_miss_count", decode_hard_metrics["hard_miss_count"])
                .detach()
                .cpu()
                .item()
            ),
            "decode_expert_prefetch_load_count": float(
                decode_hard_metrics.get(
                    "hard_prefetch_load_count",
                    torch.zeros((), device=generation_device, dtype=torch.float64),
                )
                .detach()
                .cpu()
                .item()
            ),
            "generated_text": generated_text,
        }
        if temporal_option_count > 0.0:
            prediction["temporal_option_switch_rate"] = float(
                (temporal_option_summary[0] / temporal_option_summary[4].clamp_min(1.0)).detach().cpu().item()
            )
            prediction["temporal_option_teacher_coverage"] = float(
                (temporal_option_summary[2] / temporal_option_summary[4].clamp_min(1.0)).detach().cpu().item()
            )
            prediction["temporal_option_router_overlap"] = float(
                (temporal_option_summary[3] / temporal_option_summary[4].clamp_min(1.0)).detach().cpu().item()
            )
        if args.compute_generation_nll:
            prediction["generation_nll"] = sample_generation_nll
        local_predictions.append(prediction)
        with local_predictions_output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            handle.flush()
        if args.progress_every > 0 and (
            local_pos == 1 or local_pos == len(local_examples) or local_pos % args.progress_every == 0
        ):
            elapsed = max(time.time() - start_time, 1e-6)
            samples_per_minute = local_pos / elapsed * 60.0
            print(
                f"[rank{rank}] progress {local_pos}/{len(local_examples)} "
                f"(global_index={idx}, {samples_per_minute:.2f} samples/min).",
                flush=True,
            )

    save_promoe_trace_output(args.promoe_trace_output, local_promoe_traces, rank=rank, world_size=world_size)
    (
        usage_counts,
        token_counts,
        routed_slot_counts,
        generation_metric_sums,
        extra_cache_policy_metric_sums,
    ) = gather_metric_tensors_from_rank_files(
        output_dir=output_dir,
        world_size=world_size,
        rank=rank,
        usage_counts=usage_counts,
        token_counts=token_counts,
        routed_slot_counts=routed_slot_counts,
        generation_metric_sums=generation_metric_sums,
        extra_cache_policy_metric_sums=extra_cache_policy_metric_sums,
        return_extra_cache_policy_metric_sums=True,
    )
    predictions = gather_predictions_from_rank_files(
        local_predictions=local_predictions,
        output_dir=output_dir,
        world_size=world_size,
        rank=rank,
    )

    if rank == 0:
        correct = sum(1 for item in predictions if item["correct"])
        total = len(predictions)
        generation_token_count = generation_metric_sums[1].clamp_min(1.0)
        generation_hard_step_count = generation_metric_sums[4].clamp_min(1.0)
        decode_hard_step_count = generation_metric_sums[12].clamp_min(1.0)
        reference_token_count = generation_metric_sums[6].clamp_min(1.0)
        generated_token_count = generation_metric_sums[7].clamp_min(1.0)
        generation_nll = generation_metric_sums[0] / generation_token_count
        metric_mode = hard_cache_metric_mode(model, cache_policy_config)
        generation_hard_hit_rate = generation_metric_sums[2] / generation_hard_step_count
        decode_hard_hit_rate = generation_metric_sums[10] / decode_hard_step_count
        reference_moe_loss = generation_metric_sums[5] / reference_token_count
        full_hard_summary = {
            "hit_rate": float(generation_hard_hit_rate.cpu().item()),
            "miss_rate": float((1.0 - generation_hard_hit_rate).cpu().item()),
            "overlap_count": float((generation_metric_sums[3] / generation_hard_step_count).cpu().item()),
            "counted_steps": float(generation_metric_sums[4].cpu().item()),
            "access_count": float(generation_metric_sums[8].cpu().item()),
            "miss_count": float(generation_metric_sums[9].cpu().item()),
            "prefetch_load_count": float(generation_metric_sums[16].cpu().item()),
        }
        full_hard_summary["load_adjusted_access_count"] = (
            full_hard_summary["access_count"] + full_hard_summary["prefetch_load_count"]
        )
        full_hard_summary["load_adjusted_miss_rate"] = (
            full_hard_summary["miss_count"] / full_hard_summary["load_adjusted_access_count"]
            if full_hard_summary["load_adjusted_access_count"] > 0.0
            else 0.0
        )
        full_hard_summary["load_adjusted_hit_rate"] = 1.0 - full_hard_summary["load_adjusted_miss_rate"]
        decode_hard_summary = {
            "hit_rate": float(decode_hard_hit_rate.cpu().item()),
            "miss_rate": float((1.0 - decode_hard_hit_rate).cpu().item()),
            "overlap_count": float((generation_metric_sums[11] / decode_hard_step_count).cpu().item()),
            "counted_steps": float(generation_metric_sums[12].cpu().item()),
            "access_count": float(generation_metric_sums[13].cpu().item()),
            "miss_count": float(generation_metric_sums[14].cpu().item()),
            "prefetch_load_count": float(generation_metric_sums[18].cpu().item()),
        }
        decode_hard_summary["load_adjusted_access_count"] = (
            decode_hard_summary["access_count"] + decode_hard_summary["prefetch_load_count"]
        )
        decode_hard_summary["load_adjusted_miss_rate"] = (
            decode_hard_summary["miss_count"] / decode_hard_summary["load_adjusted_access_count"]
            if decode_hard_summary["load_adjusted_access_count"] > 0.0
            else 0.0
        )
        decode_hard_summary["load_adjusted_hit_rate"] = 1.0 - decode_hard_summary["load_adjusted_miss_rate"]
        full_load_summary = expert_load_summary(
            full_hard_summary,
            float(generated_token_count.cpu().item()),
            expert_size_bytes,
            bandwidth_gbps=args.expert_load_bandwidth_gbps,
        )
        decode_output_token_count = decode_hard_summary["counted_steps"] / float(max(num_moe_layers, 1))
        decode_load_summary = expert_load_summary(
            decode_hard_summary,
            decode_output_token_count,
            expert_size_bytes,
            bandwidth_gbps=args.expert_load_bandwidth_gbps,
        )
        internal_predictor_parameter_count = int(
            future_router_summary.get("future_router_parameter_count", 0)
        ) + int(future_router_summary.get("temporal_option_parameter_count", 0))
        predictor_summary = predictor_overhead_summary(
            internal_predictor_parameter_count,
            effective_tflops=args.predictor_effective_tflops,
            dtype_bytes=_dtype_num_bytes(getattr(getattr(_unwrap_cache_model(model), "config", None), "torch_dtype", None)),
            external_predictor=promoe_predictor,
            num_moe_layers=num_moe_layers,
            prefetch_lookahead=cache_policy_config.prefetch_lookahead,
        )
        finemoe_summary = finemoe_store_summary(finemoe_store)
        cache_policy_configs_by_policy = {
            cache_policy_config.policy: cache_policy_config,
            **extra_cache_policy_configs,
        }
        cache_policy_metric_sums_by_policy = {
            cache_policy_config.policy: generation_metric_sums,
            **extra_cache_policy_metric_sums,
        }
        cache_policy_metrics = {
            metric_label: summarize_cache_policy_metric_sums(
                policy=cache_policy_configs_by_policy[metric_label].policy,
                metric_sums=policy_metric_sums,
                metric_mode=hard_cache_metric_mode(model, cache_policy_configs_by_policy[metric_label]),
                cache_policy_config=cache_policy_configs_by_policy[metric_label],
                cache_size=cache_size,
                total=total,
                correct=correct,
                generated_token_count=float(generated_token_count.cpu().item()),
                num_moe_layers=num_moe_layers,
                expert_size_bytes=expert_size_bytes,
                expert_load_bandwidth_gbps=args.expert_load_bandwidth_gbps,
            )
            for metric_label, policy_metric_sums in cache_policy_metric_sums_by_policy.items()
        }
        metrics = {
            "model_path": str(Path(args.model_path).resolve()) if args.model_path is not None else config["model"]["name"],
            "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path is not None else None,
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "moe_loss": float(reference_moe_loss.cpu().item()),
            "reference_moe_loss": float(reference_moe_loss.cpu().item()),
            "reference_loss_token_count": int(generation_metric_sums[6].cpu().item()),
            "generated_token_count": int(generation_metric_sums[7].cpu().item()),
            "hard_cache_hit_rate": float(generation_hard_hit_rate.cpu().item()),
            "hard_cache_miss_rate": float((1.0 - generation_hard_hit_rate).cpu().item()),
            "load_adjusted_hard_cache_hit_rate": full_hard_summary["load_adjusted_hit_rate"],
            "load_adjusted_hard_cache_miss_rate": full_hard_summary["load_adjusted_miss_rate"],
            "hard_cache_overlap_count": float((generation_metric_sums[3] / generation_hard_step_count).cpu().item()),
            "hard_cache_access_count": int(generation_metric_sums[8].cpu().item()),
            "hard_cache_miss_count": float(generation_metric_sums[9].cpu().item()),
            "generation_hard_cache_hit_rate": float(generation_hard_hit_rate.cpu().item()),
            "generation_hard_cache_miss_rate": float((1.0 - generation_hard_hit_rate).cpu().item()),
            "generation_load_adjusted_hard_cache_hit_rate": full_hard_summary["load_adjusted_hit_rate"],
            "generation_load_adjusted_hard_cache_miss_rate": full_hard_summary["load_adjusted_miss_rate"],
            "generation_hard_cache_overlap_count": float(
                (generation_metric_sums[3] / generation_hard_step_count).cpu().item()
            ),
            "generation_hard_cache_counted_steps": int(generation_metric_sums[4].cpu().item()),
            "generation_hard_cache_access_count": int(generation_metric_sums[8].cpu().item()),
            "generation_hard_cache_miss_count": float(generation_metric_sums[9].cpu().item()),
            "generation_hard_cache_demand_miss_count": float(generation_metric_sums[15].cpu().item()),
            "generation_hard_cache_prefetch_load_count": float(generation_metric_sums[16].cpu().item()),
            "decode_hard_cache_hit_rate": decode_hard_summary["hit_rate"],
            "decode_hard_cache_miss_rate": decode_hard_summary["miss_rate"],
            "decode_load_adjusted_hard_cache_hit_rate": decode_hard_summary["load_adjusted_hit_rate"],
            "decode_load_adjusted_hard_cache_miss_rate": decode_hard_summary["load_adjusted_miss_rate"],
            "decode_hard_cache_overlap_count": decode_hard_summary["overlap_count"],
            "decode_hard_cache_counted_steps": int(decode_hard_summary["counted_steps"]),
            "decode_output_token_count": int(round(decode_output_token_count)),
            "decode_hard_cache_access_count": int(decode_hard_summary["access_count"]),
            "decode_hard_cache_miss_count": decode_hard_summary["miss_count"],
            "decode_hard_cache_demand_miss_count": float(generation_metric_sums[17].cpu().item()),
            "decode_hard_cache_prefetch_load_count": float(generation_metric_sums[18].cpu().item()),
            "temporal_option_switch_rate": (
                float((generation_metric_sums[19] / generation_metric_sums[23].clamp_min(1.0)).cpu().item())
                if float(generation_metric_sums[23].cpu().item()) > 0.0
                else 0.0
            ),
            "temporal_option_target_switch_rate": (
                float((generation_metric_sums[20] / generation_metric_sums[23].clamp_min(1.0)).cpu().item())
                if float(generation_metric_sums[23].cpu().item()) > 0.0
                else 0.0
            ),
            "temporal_option_teacher_coverage": (
                float((generation_metric_sums[21] / generation_metric_sums[23].clamp_min(1.0)).cpu().item())
                if float(generation_metric_sums[23].cpu().item()) > 0.0
                else 0.0
            ),
            "temporal_option_router_overlap": (
                float((generation_metric_sums[22] / generation_metric_sums[23].clamp_min(1.0)).cpu().item())
                if float(generation_metric_sums[23].cpu().item()) > 0.0
                else 0.0
            ),
            "temporal_option_counted_steps": float(generation_metric_sums[23].cpu().item()),
            "expert_size_MB": decode_load_summary["expert_size_MB"],
            "load_MB_per_generated_token": decode_load_summary["load_MB_per_generated_token"],
            "decode_load_MB_per_generated_token": decode_load_summary["load_MB_per_generated_token"],
            "decode_load_MB_per_decode_token": decode_load_summary["load_MB_per_generated_token"],
            "decode_expert_load_MB": decode_load_summary["expert_load_MB"],
            "decode_expert_misses_per_generated_token": decode_load_summary["expert_misses_per_generated_token"],
            "decode_expert_misses_per_decode_token": decode_load_summary["expert_misses_per_generated_token"],
            "full_request_load_MB_per_generated_token": full_load_summary["load_MB_per_generated_token"],
            "full_request_expert_load_MB": full_load_summary["expert_load_MB"],
            "full_request_expert_misses_per_generated_token": full_load_summary["expert_misses_per_generated_token"],
            "cache_size": cache_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "reference_loss_skipped": bool(args.skip_reference_loss),
            "expert_usage_output": str(expert_usage_output),
            "cache_policy": cache_policy_config.policy,
            "hard_cache_metric_mode": metric_mode,
            "cache_policy_recency_weight": cache_policy_config.recency_weight,
            "cache_policy_frequency_weight": cache_policy_config.frequency_weight,
            "cache_policy_recency_decay": cache_policy_config.recency_decay,
            "cache_policy_lrfu_lambda": cache_policy_config.lrfu_lambda,
            "cache_policy_prefetch_budget": cache_policy_config.prefetch_budget,
            "future_cache_priority_mode": cache_policy_config.future_cache_priority_mode,
            "spatio_only_enabled": bool(config.get("cache_moe", {}).get("spatio_only_enabled", False)),
            "cache_policy_effective_prefetch_budget": effective_cache_policy_prefetch_budget(
                cache_policy_config,
                cache_size=cache_size,
            ),
            "cache_policy_prefetch_lookahead": cache_policy_config.prefetch_lookahead,
            "spatio_temporal_refine_mode": cache_policy_config.spatio_temporal_refine_mode,
            "spatio_temporal_refine_budget": cache_policy_config.spatio_temporal_refine_budget,
            "spatio_temporal_refine_budgets": requested_refine_budgets,
            "spatio_temporal_refine_budget_metrics": {
                str(budget): cache_policy_metrics[metric_label]
                for budget, metric_label in refine_budget_metric_labels.items()
            },
            "promoe_predictor_path": args.promoe_predictor_path,
            "promoe_trace_output": args.promoe_trace_output,
            "finemoe_store_path": args.finemoe_store_path,
            "finemoe_top_k": cache_policy_config.finemoe_top_k,
            "finemoe_candidate_pool": cache_policy_config.finemoe_candidate_pool,
            "finemoe_semantic_weight": cache_policy_config.finemoe_semantic_weight,
            "finemoe_trajectory_weight": cache_policy_config.finemoe_trajectory_weight,
            "finemoe_prefetch_threshold": cache_policy_config.finemoe_prefetch_threshold,
            "finemoe_min_prefetch_threshold": cache_policy_config.finemoe_min_prefetch_threshold,
            "finemoe_max_prefetch_threshold": cache_policy_config.finemoe_max_prefetch_threshold,
            "finemoe_eviction_probability_weight": cache_policy_config.finemoe_eviction_probability_weight,
            "extra_cache_policies": extra_cache_policy_names,
            "cache_policy_metrics": cache_policy_metrics,
            **future_router_summary,
            **predictor_summary,
            **finemoe_summary,
        }
        if args.expert_load_bandwidth_gbps > 0.0:
            metrics.update(
                {
                    "expert_load_bandwidth_GBps": args.expert_load_bandwidth_gbps,
                    "estimated_load_TPOT_ms_per_generated_token": decode_load_summary[
                        "estimated_load_TPOT_ms_per_generated_token"
                    ],
                    "decode_estimated_load_TPOT_ms_per_generated_token": decode_load_summary[
                        "estimated_load_TPOT_ms_per_generated_token"
                    ],
                    "full_request_estimated_load_TPOT_ms_per_generated_token": full_load_summary[
                        "estimated_load_TPOT_ms_per_generated_token"
                    ],
                }
            )
        if args.compute_generation_nll:
            metrics.update(
                {
                    "generation_nll": float(generation_nll.cpu().item()),
                    "generation_ppl": float(torch.exp(generation_nll).cpu().item()),
                    "generation_nll_token_count": int(generation_metric_sums[1].cpu().item()),
                }
            )
        predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with predictions_output.open("w", encoding="utf-8") as handle:
            for item in predictions:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        metrics = emphasize_paper_metrics(metrics)
        metrics_output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        save_expert_usage(
            expert_usage_output,
            usage_counts,
            token_counts,
            routed_slot_counts,
            sample_count=total,
            model_name=config["model"]["name"],
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
