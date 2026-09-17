import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "trust_remote_code": False,
        "attn_implementation": "eager",
        "torch_dtype": "bfloat16",
        "use_cache": False,
        "output_router_logits": True,
        "config_name": None,
        "tokenizer_name": None,
        "init_from_checkpoint": None,
        "tokenizer_use_fast": True,
    },
    "dataset": {
        "name": "openai/gsm8k",
        "subset": "main",
        "train_split": "train",
        "eval_split": "test",
        "split_train_for_eval": False,
        "eval_holdout_size": None,
        "eval_holdout_ratio": 0.01,
        "split_seed": 42,
        "split_cache_path": None,
        "format": "auto",
        "messages_field": "messages",
        "prepend_system_prompt": False,
        "source_field": "source",
        "source_include_patterns": [],
        "source_exclude_patterns": [],
        "max_train_samples": None,
        "max_eval_samples": None,
        "max_length": 1024,
        "num_proc": 1,
        "label_mask_strategy": "token_length",
        "system_prompt": (
            "You are a careful math tutor. Solve the problem step by step and end "
            "with a final answer using the format `#### answer`."
        ),
    },
    "quantization": {
        "enabled": False,
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
    },
    "lora": {
        "enabled": False,
        "implementation": "peft",
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "bias": "none",
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    },
    "cache_moe": {
        "enabled": False,
        "cache_size": 20,
        "temperature": 0.25,
        "balance_weight": 0.01,
        "swap_weight": 0.10,
        "copy_current_router_init": True,
        "future_cache_priority_mode": "current_plus_future",
        "spatio_temporal_enabled": False,
        "spatio_only_enabled": False,
        "swap_loss_future_gate_only": False,
    },
    "window_cache_loss": {
        "enabled": False,
        "window_size": 16,
        "capacity": 20,
        "target_avg_swap": 1.0,
        "max_swap_thresh": 4.0,
        "mtp_swap_ratio": 1.5,
        "avg_swap_coeff": 1.0,
        "peak_swap_coeff": 2.0,
        "mtp_swap_coeff": 0.0,
        "sharpness_alpha": 2.0,
        "peak_penalty_gamma": 2.0,
        "training_mode": "cache_aware",
    },
    "lru_cache_metrics": {
        "enabled": False,
        "cache_size": 20,
    },
    "training": {
        "output_dir": "./outputs/qwen3_30b_a3b_instruct_2507_gsm8k_full_ft",
        "overwrite_output_dir": True,
        "seed": 42,
        "num_train_epochs": 2,
        "learning_rate": 1e-5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": True,
        "fp16": False,
        "tf32": True,
        "optim": "adamw_torch_fused",
        "logging_steps": 5,
        "eval_strategy": "steps",
        "eval_steps": 50,
        "save_strategy": "steps",
        "save_steps": 50,
        "save_total_limit": 2,
        "save_safetensors": True,
        "report_to": "none",
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
        "load_best_model_at_end": False,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "do_final_eval": True,
        "resume_from_checkpoint": None,
    },
}


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config(path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(paths: Union[str, Path, Iterable[Union[str, Path]]]) -> Dict[str, Any]:
    if isinstance(paths, (str, Path)):
        ordered_paths: List[Path] = [Path(paths)]
    else:
        ordered_paths = [Path(path) for path in paths]

    merged = copy.deepcopy(DEFAULT_CONFIG)
    for config_path in ordered_paths:
        merged = _merge_dicts(merged, _load_raw_config(config_path))
    return merged


def resolve_cache_size(config: Dict[str, Any], default: int = 20) -> int:
    """Resolve the active hard-cache capacity from the enabled cache method."""
    cache_moe_cfg = config.get("cache_moe", {})
    if cache_moe_cfg.get("enabled", False):
        return int(cache_moe_cfg.get("cache_size", default))

    temporal_option_cfg = config.get("temporal_option_moe", {})
    if temporal_option_cfg.get("enabled", False):
        return int(temporal_option_cfg.get("option_size", default))

    window_cache_cfg = config.get("window_cache_loss", {})
    if window_cache_cfg.get("enabled", False):
        return int(window_cache_cfg.get("capacity", default))

    lru_cache_cfg = config.get("lru_cache_metrics", {})
    if lru_cache_cfg.get("enabled", False):
        return int(lru_cache_cfg.get("cache_size", default))

    return int(default)
