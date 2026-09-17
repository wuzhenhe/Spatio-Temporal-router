import argparse
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

# Disable all cuDNN SDPA backends to avoid mha_graph recompute failure
# under FSDP + gradient checkpointing (cuDNN 9.x known issue).
# Falls back to the math kernel which handles recompute correctly.
if hasattr(torch.backends.cuda, "enable_flash_sdp"):
    torch.backends.cuda.enable_flash_sdp(False)
if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
    torch.backends.cuda.enable_mem_efficient_sdp(False)
if hasattr(torch.backends.cuda, "enable_math_sdp"):
    torch.backends.cuda.enable_math_sdp(True)
import yaml
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from collator import SupervisedDataCollator
from cache_moe import CacheMoEConfig, install_cache_moe
from config import load_config, resolve_cache_size
from data import prepare_datasets
from lru_cache_metrics import LRUCacheMetricConfig, install_lru_cache_metrics, unwrap_lru_cache_metrics
from temporal_option_moe import (
    TemporalOptionMoEConfig,
    install_temporal_option_moe,
    set_temporal_option_trainable_parameters,
)
from trainer import MoESFTTrainer
from window_cache_loss import WindowCacheLossConfig, install_window_cache_loss


def patch_transformers_remote_code_compat() -> None:
    """Keep older trusted remote modeling files working with newer Transformers."""
    try:
        from transformers.utils import import_utils
    except ImportError:
        return

    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: hasattr(torch, "fx")

    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        DynamicCache = None

    if DynamicCache is not None:
        def _dynamic_cache_seq_length(self, layer_idx: int = 0) -> int:
            key_cache = getattr(self, "key_cache", None)
            if key_cache is not None:
                if len(key_cache) == 0:
                    return 0
                idx = min(max(int(layer_idx), 0), len(key_cache) - 1)
                cache = key_cache[idx]
                return int(cache.shape[-2]) if cache is not None else 0
            if hasattr(self, "get_seq_length"):
                try:
                    return int(self.get_seq_length(layer_idx))
                except TypeError:
                    return int(self.get_seq_length())
            return 0

        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(lambda self: _dynamic_cache_seq_length(self))

        if not hasattr(DynamicCache, "get_max_length"):
            def _get_max_length(self):
                if hasattr(self, "get_max_cache_shape"):
                    return self.get_max_cache_shape()
                return None

            DynamicCache.get_max_length = _get_max_length

        if not hasattr(DynamicCache, "get_usable_length"):
            def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
                previous_seq_length = _dynamic_cache_seq_length(self, layer_idx=layer_idx)
                max_length = self.get_max_length() if hasattr(self, "get_max_length") else None
                if max_length is not None and previous_seq_length + int(new_seq_length) > int(max_length):
                    return max(int(max_length) - int(new_seq_length), 0)
                return previous_seq_length

            DynamicCache.get_usable_length = _get_usable_length


class TrainableParamsCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None and hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
        elif model is not None:
            total_params = sum(param.numel() for param in model.parameters())
            trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
            print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")


class NativeLoRALinear(nn.Module):
    """A plain-module LoRA wrapper that FSDP saves as normal model weights."""

    def __init__(self, base_layer: nn.Linear, r: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"NativeLoRALinear expects nn.Linear, got {type(base_layer)!r}.")
        if int(r) <= 0:
            raise ValueError(f"LoRA rank must be positive, got r={r}.")
        self.base_layer = base_layer
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.r)
        self.lora_dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()
        try:
            self.lora_A = nn.Linear(
                base_layer.in_features,
                self.r,
                bias=False,
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype,
            )
            self.lora_B = nn.Linear(
                self.r,
                base_layer.out_features,
                bias=False,
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype,
            )
        except TypeError:
            self.lora_A = nn.Linear(base_layer.in_features, self.r, bias=False).to(
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype,
            )
            self.lora_B = nn.Linear(self.r, base_layer.out_features, bias=False).to(
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype,
            )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(input)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(input))) * self.scaling
        return result + lora_out.to(dtype=result.dtype)


def _matches_lora_target(module_name: str, target_modules: Any) -> bool:
    target_set = {str(target) for target in target_modules}
    leaf_name = module_name.rsplit(".", 1)[-1]
    return leaf_name in target_set or module_name in target_set or any(
        module_name.endswith(f".{target}") for target in target_set
    )


def install_native_lora(model: nn.Module, lora_cfg: Dict[str, Any]) -> nn.Module:
    target_modules = lora_cfg.get("target_modules") or []
    replaced = []
    for parent_name, parent in list(model.named_modules()):
        if isinstance(parent, NativeLoRALinear):
            continue
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, NativeLoRALinear):
                continue
            if not isinstance(child, nn.Linear):
                continue
            if not _matches_lora_target(full_name, target_modules):
                continue
            setattr(
                parent,
                child_name,
                NativeLoRALinear(
                    base_layer=child,
                    r=int(lora_cfg["r"]),
                    alpha=float(lora_cfg["alpha"]),
                    dropout=float(lora_cfg.get("dropout", 0.0)),
                ),
            )
            replaced.append(full_name)

    if not replaced:
        raise ValueError(
            "lora.enabled=true with implementation=native, but no target nn.Linear modules were replaced. "
            f"target_modules={target_modules}"
        )
    model._native_lora_enabled = True
    model._native_lora_target_modules = list(target_modules)
    print(
        "Installed native LoRA: "
        f"targets={list(target_modules)}, replaced_modules={len(replaced)}, "
        f"first_modules={replaced[:8]}",
        flush=True,
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT MoE models with Transformers.")
    parser.add_argument(
        "--config",
        type=str,
        action="append",
        required=True,
        help="Path to a YAML config/profile file. Pass multiple times to merge in order.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Build the model, validate Cache-MoE predictor parameters, then exit before training.",
    )
    return parser.parse_args()


def resolve_dtype(name: str):
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": "auto",
    }
    key = str(name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def build_quantization_config(config: Dict[str, Any]) -> BitsAndBytesConfig:
    quant_cfg = config["quantization"]
    compute_dtype = resolve_dtype(quant_cfg["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_model(config: Dict[str, Any], install_methods: bool = True):
    patch_transformers_remote_code_compat()

    model_cfg = config["model"]
    quant_cfg = config["quantization"]
    lora_cfg = config["lora"]
    training_cfg = config["training"]

    dtype = resolve_dtype(model_cfg["torch_dtype"])

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": model_cfg.get("trust_remote_code", False),
        "attn_implementation": model_cfg.get("attn_implementation", "eager"),
    }
    if model_cfg.get("config_name"):
        model_kwargs["config"] = AutoConfig.from_pretrained(
            model_cfg["config_name"],
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
    dtype_key = "dtype" if "dtype" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters else "torch_dtype"
    model_kwargs[dtype_key] = dtype

    if model_cfg.get("dequantize_mxfp4", False):
        from transformers import Mxfp4Config

        if quant_cfg.get("enabled", False):
            raise ValueError("MXFP4 dequantization and bitsandbytes cannot both be enabled.")
        model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)

    if quant_cfg.get("enabled", False):
        model_kwargs["quantization_config"] = build_quantization_config(config)
        model_kwargs["device_map"] = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], **model_kwargs)
    except Exception as exc:
        requested_attn = model_kwargs.get("attn_implementation")
        if requested_attn and str(requested_attn).lower() != "eager":
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs["attn_implementation"] = "eager"
            print(
                "Model load failed with "
                f"attn_implementation={requested_attn!r}; retrying with 'eager'. "
                f"Original error: {type(exc).__name__}: {exc}",
                flush=True,
            )
            model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], **fallback_kwargs)
        else:
            raise
    model.config.use_cache = model_cfg.get("use_cache", False)
    model.config.output_router_logits = model_cfg.get("output_router_logits", True)

    if training_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=training_cfg.get("gradient_checkpointing_kwargs", {})
        )

    if quant_cfg.get("enabled", False):
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
        )

    if not install_methods:
        return model

    return install_training_methods(model, config)


def install_training_methods(model, config: Dict[str, Any]):
    lora_cfg = config["lora"]
    cache_cfg_raw = config.get("cache_moe", {})
    temporal_option_cfg_raw = config.get("temporal_option_moe", {})
    window_cache_cfg_raw = config.get("window_cache_loss", {})
    lru_cfg_raw = config.get("lru_cache_metrics", {})
    window_cache_enabled = window_cache_cfg_raw.get("enabled", window_cache_cfg_raw.get("enable", False))
    window_cache_max_swap_thresh = window_cache_cfg_raw.get(
        "max_swap_thresh",
        window_cache_cfg_raw.get("max_swap_threshold", 4.0),
    )
    enabled_methods = {
        "cache_moe": cache_cfg_raw.get("enabled", False),
        "temporal_option_moe": temporal_option_cfg_raw.get("enabled", False),
        "window_cache_loss": (
            window_cache_enabled
            and window_cache_cfg_raw.get("training_mode", "cache_aware") != "none"
        ),
        "lru_cache_metrics": lru_cfg_raw.get("enabled", False),
    }
    if sum(bool(enabled) for enabled in enabled_methods.values()) > 1:
        raise ValueError(
            "Enable at most one of cache_moe, temporal_option_moe, window_cache_loss, or lru_cache_metrics. "
            f"Received: {enabled_methods}"
        )

    temporal_option_config = TemporalOptionMoEConfig(
        enabled=temporal_option_cfg_raw.get("enabled", False),
        option_size=temporal_option_cfg_raw.get("option_size", 16),
        switch_threshold=temporal_option_cfg_raw.get("switch_threshold", 0.5),
        switch_init_bias=temporal_option_cfg_raw.get("switch_init_bias", -2.0),
        controller_hidden_dim=temporal_option_cfg_raw.get("controller_hidden_dim", 0),
        set_embed_dim=temporal_option_cfg_raw.get("set_embed_dim", 0),
        copy_router_init=temporal_option_cfg_raw.get("copy_router_init", True),
        selection_loss_weight=temporal_option_cfg_raw.get("selection_loss_weight", 0.10),
        termination_loss_weight=temporal_option_cfg_raw.get("termination_loss_weight", 0.05),
        deliberation_cost=temporal_option_cfg_raw.get("deliberation_cost", 0.02),
        entropy_weight=temporal_option_cfg_raw.get("entropy_weight", 0.0),
        force_switch_on_teacher_miss=temporal_option_cfg_raw.get("force_switch_on_teacher_miss", False),
        train_base_model=temporal_option_cfg_raw.get("train_base_model", True),
        train_router=temporal_option_cfg_raw.get("train_router", False),
        train_lora=temporal_option_cfg_raw.get("train_lora", False),
        fast_training=temporal_option_cfg_raw.get("fast_training", False),
    )
    model = install_temporal_option_moe(model, temporal_option_config)

    if lora_cfg.get("enabled", False):
        lora_implementation = str(lora_cfg.get("implementation", "peft")).lower()
        if lora_implementation == "native":
            model = install_native_lora(model, lora_cfg)
        elif lora_implementation == "peft":
            from peft import LoraConfig, get_peft_model

            lora_kwargs: Dict[str, Any] = {}
            if lora_cfg.get("modules_to_save") is not None:
                lora_kwargs["modules_to_save"] = lora_cfg["modules_to_save"]
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=lora_cfg["r"],
                lora_alpha=lora_cfg["alpha"],
                lora_dropout=lora_cfg["dropout"],
                bias=lora_cfg.get("bias", "none"),
                target_modules=lora_cfg["target_modules"],
                **lora_kwargs,
            )
            model = get_peft_model(model, peft_config)
        else:
            raise ValueError(f"Unsupported lora.implementation: {lora_cfg.get('implementation')}")

    cache_moe_config = CacheMoEConfig(
        enabled=cache_cfg_raw.get("enabled", False),
        cache_size=cache_cfg_raw.get("cache_size", 20),
        temperature=cache_cfg_raw.get("temperature", 0.25),
        balance_weight=cache_cfg_raw.get("balance_weight", 0.01),
        swap_weight=cache_cfg_raw.get("swap_weight", 0.10),
        future_gate_type=cache_cfg_raw.get("future_gate_type", "linear"),
        future_gate_hidden_dim=cache_cfg_raw.get("future_gate_hidden_dim", 0),
        future_gate_target_params_per_layer=cache_cfg_raw.get("future_gate_target_params_per_layer", 2_000_000),
        copy_current_router_init=cache_cfg_raw.get("copy_current_router_init", True),
        future_cache_priority_mode=cache_cfg_raw.get("future_cache_priority_mode", "current_plus_future"),
        spatio_temporal_enabled=cache_cfg_raw.get("spatio_temporal_enabled", False),
        spatio_only_enabled=cache_cfg_raw.get("spatio_only_enabled", False),
        swap_loss_future_gate_only=cache_cfg_raw.get("swap_loss_future_gate_only", False),
        split_grad_clip=cache_cfg_raw.get("split_grad_clip", False),
        layer_swap_weight_schedule=cache_cfg_raw.get("layer_swap_weight_schedule", "uniform"),
        layer_swap_weight_min=cache_cfg_raw.get("layer_swap_weight_min", 1.0),
        layer_swap_weight_max=cache_cfg_raw.get("layer_swap_weight_max", 1.0),
        layer_swap_weights=cache_cfg_raw.get("layer_swap_weights"),
        normalize_layer_swap_weights=cache_cfg_raw.get("normalize_layer_swap_weights", True),
        layer_adaptive_swap_weight=cache_cfg_raw.get("layer_adaptive_swap_weight", False),
        layer_adaptive_weight_min=cache_cfg_raw.get("layer_adaptive_weight_min", 0.5),
        layer_adaptive_weight_max=cache_cfg_raw.get("layer_adaptive_weight_max", 1.5),
        layer_adaptive_warmup_steps=cache_cfg_raw.get("layer_adaptive_warmup_steps", 0),
        layer_adaptive_update_interval=cache_cfg_raw.get("layer_adaptive_update_interval", 20),
        layer_adaptive_lr=cache_cfg_raw.get("layer_adaptive_lr", 0.02),
        layer_adaptive_cosine_ema=cache_cfg_raw.get("layer_adaptive_cosine_ema", 0.95),
        layer_adaptive_cosine_scale=cache_cfg_raw.get("layer_adaptive_cosine_scale", 1.0),
        layer_adaptive_cosine_deadband=cache_cfg_raw.get("layer_adaptive_cosine_deadband", 0.0),
        layer_adaptive_history=cache_cfg_raw.get("layer_adaptive_history", True),
    )
    model = install_cache_moe(model, cache_moe_config)
    if temporal_option_config.enabled and not temporal_option_config.train_base_model:
        freeze_summary = set_temporal_option_trainable_parameters(
            model,
            train_router=temporal_option_config.train_router,
            train_lora=temporal_option_config.train_lora,
        )
        print(
            "Temporal Option trainable-only mode: "
            f"trainable_params={freeze_summary['temporal_option_trainable_only_trainable_parameter_count']:,}, "
            f"controller_params={freeze_summary['temporal_option_trainable_controller_parameter_count']:,}, "
            f"router_params={freeze_summary['temporal_option_trainable_router_parameter_count']:,}, "
            f"lora_params={freeze_summary['temporal_option_trainable_lora_parameter_count']:,}, "
            f"frozen_base_params={freeze_summary['temporal_option_trainable_only_frozen_parameter_count']:,}",
            flush=True,
        )
    window_cache_config = WindowCacheLossConfig(
        enabled=window_cache_enabled,
        window_size=window_cache_cfg_raw.get("window_size", 16),
        capacity=window_cache_cfg_raw.get("capacity", 20),
        target_avg_swap=window_cache_cfg_raw.get("target_avg_swap", 1.0),
        max_swap_thresh=window_cache_max_swap_thresh,
        mtp_swap_ratio=window_cache_cfg_raw.get("mtp_swap_ratio", 1.5),
        mtp_num_draft_tokens=window_cache_cfg_raw.get("mtp_num_draft_tokens", 4),
        avg_swap_coeff=window_cache_cfg_raw.get("avg_swap_coeff", 1.0),
        peak_swap_coeff=window_cache_cfg_raw.get("peak_swap_coeff", 2.0),
        mtp_swap_coeff=window_cache_cfg_raw.get("mtp_swap_coeff", 1.0),
        sharpness_alpha=window_cache_cfg_raw.get("sharpness_alpha", 2.0),
        peak_penalty_gamma=window_cache_cfg_raw.get("peak_penalty_gamma", 2.0),
        training_mode=window_cache_cfg_raw.get("training_mode", "cache_aware"),
    )
    model = install_window_cache_loss(model, window_cache_config)
    lru_cache_metric_config = LRUCacheMetricConfig(
        enabled=lru_cfg_raw.get("enabled", False),
        cache_size=lru_cfg_raw.get("cache_size", 20),
    )
    model = install_lru_cache_metrics(model, lru_cache_metric_config)

    return model


def build_tokenizer(config: Dict[str, Any]):
    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.get("tokenizer_name") or model_cfg["name"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        use_fast=model_cfg.get("tokenizer_use_fast", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def build_training_args(config: Dict[str, Any]) -> TrainingArguments:
    train_cfg = config["training"]
    signature = inspect.signature(TrainingArguments.__init__).parameters

    args_dict: Dict[str, Any] = {
        "output_dir": train_cfg["output_dir"],
        "overwrite_output_dir": train_cfg["overwrite_output_dir"],
        "seed": train_cfg["seed"],
        "num_train_epochs": train_cfg["num_train_epochs"],
        "learning_rate": train_cfg["learning_rate"],
        "lr_scheduler_type": train_cfg["lr_scheduler_type"],
        "warmup_ratio": train_cfg["warmup_ratio"],
        "weight_decay": train_cfg["weight_decay"],
        "max_grad_norm": train_cfg["max_grad_norm"],
        "per_device_train_batch_size": train_cfg["per_device_train_batch_size"],
        "per_device_eval_batch_size": train_cfg["per_device_eval_batch_size"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "gradient_checkpointing": train_cfg["gradient_checkpointing"],
        "gradient_checkpointing_kwargs": train_cfg.get("gradient_checkpointing_kwargs"),
        "bf16": train_cfg["bf16"],
        "fp16": train_cfg["fp16"],
        "tf32": train_cfg["tf32"],
        "optim": train_cfg["optim"],
        "logging_steps": train_cfg["logging_steps"],
        "eval_steps": train_cfg["eval_steps"],
        "save_strategy": train_cfg["save_strategy"],
        "save_steps": train_cfg["save_steps"],
        "save_total_limit": train_cfg["save_total_limit"],
        "save_safetensors": train_cfg["save_safetensors"],
        "report_to": train_cfg["report_to"],
        "dataloader_num_workers": train_cfg["dataloader_num_workers"],
        "remove_unused_columns": train_cfg["remove_unused_columns"],
        "ddp_find_unused_parameters": train_cfg.get("ddp_find_unused_parameters"),
        "load_best_model_at_end": train_cfg["load_best_model_at_end"],
        "metric_for_best_model": train_cfg["metric_for_best_model"],
        "greater_is_better": train_cfg["greater_is_better"],
        "label_names": ["labels"],
    }

    if "eval_strategy" in signature:
        args_dict["eval_strategy"] = train_cfg["eval_strategy"]
    elif "evaluation_strategy" in signature:
        args_dict["evaluation_strategy"] = train_cfg["eval_strategy"]

    # Transformers 5 uses fractional warmup_steps in place of warmup_ratio.
    # Do not silently drop the recipe's warmup when filtering removed arguments.
    if "warmup_ratio" not in signature and "warmup_steps" in signature:
        args_dict["warmup_steps"] = train_cfg.get("warmup_steps", train_cfg["warmup_ratio"])
    args_dict["max_steps"] = train_cfg.get("max_steps", -1)
    filtered_args = {key: value for key, value in args_dict.items() if key in signature}
    return TrainingArguments(**filtered_args)


def save_run_artifacts(config: Dict[str, Any], config_paths: Any, output_dir: Path) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    effective_config_path = output_dir / "effective_config.yaml"
    with effective_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=False)

    run_metadata = {
        "config_paths": [str(Path(path).resolve()) for path in config_paths],
        "output_dir": str(output_dir.resolve()),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2)


def cache_moe_parameter_summary(model) -> Dict[str, Any]:
    future_names = []
    spatio_names = []
    temporal_option_names = []
    lora_names = []
    router_names = []
    future_params = 0
    spatio_params = 0
    temporal_option_params = 0
    temporal_option_trainable_params = 0
    lora_params = 0
    lora_trainable_params = 0
    router_params = 0
    router_trainable_params = 0
    trainable_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable_params += int(parameter.numel())
        if "future_gate" in name:
            future_names.append(name)
            future_params += int(parameter.numel())
        if "spatio_gate" in name:
            spatio_names.append(name)
            spatio_params += int(parameter.numel())
        if ".controller." in name:
            temporal_option_names.append(name)
            temporal_option_params += int(parameter.numel())
            if parameter.requires_grad:
                temporal_option_trainable_params += int(parameter.numel())
        if "lora_" in name or ".lora_embedding_" in name:
            lora_names.append(name)
            lora_params += int(parameter.numel())
            if parameter.requires_grad:
                lora_trainable_params += int(parameter.numel())
        if (
            ".mlp.gate." in name
            or ".mlp.original_block.gate." in name
            or ".original_block.gate." in name
            or ".mlp.router." in name
            or ".mlp.original_block.router." in name
            or ".original_block.router." in name
        ):
            router_names.append(name)
            router_params += int(parameter.numel())
            if parameter.requires_grad:
                router_trainable_params += int(parameter.numel())
    return {
        "future_gate_tensor_count": len(future_names),
        "future_gate_parameter_count": future_params,
        "first_future_gate_tensors": future_names[:8],
        "spatio_gate_tensor_count": len(spatio_names),
        "spatio_gate_parameter_count": spatio_params,
        "first_spatio_gate_tensors": spatio_names[:8],
        "temporal_option_tensor_count": len(temporal_option_names),
        "temporal_option_parameter_count": temporal_option_params,
        "temporal_option_trainable_parameter_count": temporal_option_trainable_params,
        "first_temporal_option_tensors": temporal_option_names[:8],
        "lora_tensor_count": len(lora_names),
        "lora_parameter_count": lora_params,
        "lora_trainable_parameter_count": lora_trainable_params,
        "first_lora_tensors": lora_names[:8],
        "router_tensor_count": len(router_names),
        "router_parameter_count": router_params,
        "router_trainable_parameter_count": router_trainable_params,
        "first_router_tensors": router_names[:8],
        "total_cache_moe_predictor_parameter_count": future_params + spatio_params,
        "total_temporal_option_parameter_count": temporal_option_params,
        "total_trainable_parameter_count": trainable_params,
    }


def validate_and_save_cache_moe_parameter_summary(
    model,
    config: Dict[str, Any],
    output_dir: Path,
    label: str,
    require_trainable: Optional[bool] = None,
) -> Dict[str, Any]:
    summary = cache_moe_parameter_summary(model)
    if require_trainable is None:
        require_trainable = label == "before_trainer"
    cache_cfg = config.get("cache_moe", {})
    expects_temporal = bool(cache_cfg.get("enabled", False) and not cache_cfg.get("spatio_only_enabled", False))
    if expects_temporal and summary["future_gate_tensor_count"] == 0:
        raise RuntimeError(
            "cache_moe.enabled=true, but no future_gate parameters are registered. "
            "Check that install_cache_moe() ran before Trainer/FSDP wrapping."
        )
    expects_spatio = bool(
        cache_cfg.get("spatio_temporal_enabled", False) or cache_cfg.get("spatio_only_enabled", False)
    )
    if expects_spatio and summary["spatio_gate_tensor_count"] == 0:
        raise RuntimeError(
            "A spatial cache-router mode is enabled, but no spatio_gate parameters are registered. "
            "The method profile was not installed correctly."
        )
    temporal_option_cfg = config.get("temporal_option_moe", {})
    if temporal_option_cfg.get("enabled", False) and summary["temporal_option_tensor_count"] == 0:
        raise RuntimeError(
            "temporal_option_moe.enabled=true, but no controller parameters are registered. "
            "Check that install_temporal_option_moe() ran before Trainer/FSDP wrapping."
        )
    if temporal_option_cfg.get("enabled", False) and temporal_option_cfg.get("train_lora", False):
        if not config.get("lora", {}).get("enabled", False):
            raise RuntimeError("temporal_option_moe.train_lora=true, but lora.enabled=false.")
        if summary["lora_tensor_count"] == 0:
            raise RuntimeError(
                "temporal_option_moe.train_lora=true, but no lora_ parameters are registered. "
                "Check lora.implementation and lora.target_modules before launching a full run."
            )
        if require_trainable and summary["lora_trainable_parameter_count"] == 0:
            raise RuntimeError(
                "temporal_option_moe.train_lora=true, but LoRA parameters are not trainable. "
                "Check set_temporal_option_trainable_parameters() before Trainer/FSDP wrapping."
            )
    if temporal_option_cfg.get("enabled", False) and temporal_option_cfg.get("train_router", False):
        if summary["router_tensor_count"] == 0:
            raise RuntimeError(
                "temporal_option_moe.train_router=true, but no MoE router/gate parameters are registered."
            )
        if require_trainable and summary["router_trainable_parameter_count"] == 0:
            raise RuntimeError(
                "temporal_option_moe.train_router=true, but MoE router/gate parameters are not trainable."
            )
    payload = {"label": label, **summary}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"cache_moe_parameter_summary_{label}.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(
        "Cache-MoE parameter check "
        f"({label}): future_gate_tensors={summary['future_gate_tensor_count']}, "
        f"future_gate_params={summary['future_gate_parameter_count']:,}, "
        f"spatio_gate_tensors={summary['spatio_gate_tensor_count']}, "
        f"spatio_gate_params={summary['spatio_gate_parameter_count']:,}, "
        f"temporal_option_tensors={summary['temporal_option_tensor_count']}, "
        f"temporal_option_params={summary['temporal_option_parameter_count']:,}, "
        f"temporal_option_trainable_params={summary['temporal_option_trainable_parameter_count']:,}, "
        f"lora_tensors={summary['lora_tensor_count']}, "
        f"lora_params={summary['lora_parameter_count']:,}, "
        f"lora_trainable_params={summary['lora_trainable_parameter_count']:,}, "
        f"router_tensors={summary['router_tensor_count']}, "
        f"router_params={summary['router_parameter_count']:,}, "
        f"router_trainable_params={summary['router_trainable_parameter_count']:,}, "
        f"total_trainable_params={summary['total_trainable_parameter_count']:,}",
        flush=True,
    )
    return summary


def print_run_config(config: Dict[str, Any], config_paths: Any, output_dir: Path) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=False).rstrip()
    print("=" * 80)
    print("Resolved training config")
    print("Config paths:")
    for path in config_paths:
        print(f"  - {Path(path).resolve()}")
    print(f"Output dir: {output_dir.resolve()}")
    print(rendered)
    print("=" * 80)


def resolve_resume_from_checkpoint(config: Dict[str, Any], output_dir: Path):
    resume_from_checkpoint = config["training"].get("resume_from_checkpoint")
    if resume_from_checkpoint is None or resume_from_checkpoint is False:
        return None
    if isinstance(resume_from_checkpoint, str) and resume_from_checkpoint.lower() == "auto":
        return get_last_checkpoint(str(output_dir))
    return resume_from_checkpoint


def checkpoint_step(path: Path) -> int:
    if not path.name.startswith("checkpoint-"):
        return -1
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def resolve_model_init_checkpoint(path_like: str) -> Path:
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"model.init_from_checkpoint does not exist: {path}")
    if path.is_file() or path.name.startswith("checkpoint-"):
        return path
    checkpoints = sorted(
        [child for child in path.iterdir() if child.is_dir() and child.name.startswith("checkpoint-")],
        key=checkpoint_step,
    )
    if checkpoints:
        return checkpoints[-1]
    return path


def ignore_fsdp_state_restore_error(exc: KeyError) -> bool:
    if exc.args != (None,):
        return False
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return False
    rank = torch.distributed.get_rank()
    if rank == 0:
        print(
            "Ignored FSDP state_dict_type restore KeyError(None) after init checkpoint load. "
            "This is a torch/accelerate compatibility issue during context cleanup.",
            flush=True,
        )
    return True


def load_model_init_checkpoint(trainer: MoESFTTrainer, checkpoint_path: Path) -> None:
    load_from_checkpoint = getattr(trainer, "_load_from_checkpoint", None)
    if load_from_checkpoint is None:
        raise RuntimeError("This Transformers Trainer version does not expose _load_from_checkpoint.")
    print(f"Initializing model weights from checkpoint: {checkpoint_path}", flush=True)

    fsdp_plugin = getattr(getattr(trainer, "accelerator", None), "state", None)
    fsdp_plugin = getattr(fsdp_plugin, "fsdp_plugin", None)
    original_set_state_dict_type = None
    StateDictType = None
    FullyShardedDataParallel = None
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

    if fsdp_plugin is not None and getattr(fsdp_plugin, "state_dict_type", None) is None:
        if StateDictType is None:
            raise RuntimeError("FSDP plugin is active, but torch.distributed is not initialized.")
        fsdp_plugin.state_dict_type = StateDictType.SHARDED_STATE_DICT

    if (
        FullyShardedDataParallel is not None
        and StateDictType is not None
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        FullyShardedDataParallel.set_state_dict_type(
            trainer.model,
            StateDictType.SHARDED_STATE_DICT,
        )

    try:
        try:
            load_from_checkpoint(str(checkpoint_path), model=trainer.model)
        except TypeError:
            try:
                load_from_checkpoint(str(checkpoint_path))
            except KeyError as exc:
                if not ignore_fsdp_state_restore_error(exc):
                    raise
        except KeyError as exc:
            if not ignore_fsdp_state_restore_error(exc):
                raise
        print(f"Finished initializing model weights from checkpoint: {checkpoint_path}", flush=True)
    finally:
        if original_set_state_dict_type is not None and FullyShardedDataParallel is not None:
            FullyShardedDataParallel.set_state_dict_type = original_set_state_dict_type


def unwrap_trainer_model(trainer: MoESFTTrainer):
    model = trainer.model
    unwrap_model = getattr(getattr(trainer, "accelerator", None), "unwrap_model", None)
    if callable(unwrap_model):
        try:
            model = unwrap_model(model)
        except TypeError:
            model = unwrap_model(model, keep_fp32_wrapper=False)
    while hasattr(model, "module"):
        nested = getattr(model, "module")
        if nested is model:
            break
        model = nested
    return model


def latest_eval_metrics_for_current_step(trainer: MoESFTTrainer) -> Optional[Dict[str, Any]]:
    current_step = int(getattr(trainer.state, "global_step", 0))
    for record in reversed(getattr(trainer.state, "log_history", [])):
        if not any(str(key).startswith("eval_") for key in record):
            continue
        record_step = record.get("step")
        if record_step is not None and int(record_step) != current_step:
            return None
        return {key: value for key, value in record.items() if key != "step"}
    return None


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_artifacts(config, args.config, output_dir)
    print_run_config(config, args.config, output_dir)

    set_seed(config["training"]["seed"])
    tokenizer = build_tokenizer(config)
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer)
    print(f"Prepared datasets: train={len(train_dataset):,}, eval={len(eval_dataset):,}")
    collator = SupervisedDataCollator(pad_token_id=tokenizer.pad_token_id)
    training_args = build_training_args(config)

    init_from_checkpoint = config.get("model", {}).get("init_from_checkpoint")
    model = build_model(config, install_methods=not bool(init_from_checkpoint))
    if init_from_checkpoint:
        init_checkpoint_lru_metrics = bool(
            config.get("model", {}).get("init_checkpoint_lru_cache_metrics", False)
        )
        if init_checkpoint_lru_metrics:
            model = install_lru_cache_metrics(
                model,
                LRUCacheMetricConfig(enabled=True, cache_size=resolve_cache_size(config)),
            )
        model.config.pad_token_id = tokenizer.pad_token_id
        init_checkpoint_path = resolve_model_init_checkpoint(str(init_from_checkpoint))
        init_trainer = MoESFTTrainer(
            model=model,
            args=training_args,
            data_collator=collator,
        )
        load_model_init_checkpoint(init_trainer, init_checkpoint_path)
        model = unwrap_trainer_model(init_trainer)
        del init_trainer
        if init_checkpoint_lru_metrics:
            model = unwrap_lru_cache_metrics(model)
        model = install_training_methods(model, config)

    validate_and_save_cache_moe_parameter_summary(model, config, output_dir, label="before_trainer")
    if args.preflight_only:
        print("Preflight check completed; exiting before Trainer construction/training.", flush=True)
        return

    model.config.pad_token_id = tokenizer.pad_token_id

    trainer = MoESFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[TrainableParamsCallback()],
    )

    trainer.train(resume_from_checkpoint=resolve_resume_from_checkpoint(config, output_dir))
    trainer.save_state()
    validate_and_save_cache_moe_parameter_summary(trainer.model, config, output_dir, label="before_save_model")
    trainer.save_model()
    validate_and_save_cache_moe_parameter_summary(trainer.model, config, output_dir, label="after_save_model")
    trainer.save_cache_moe_layer_adaptive_state(str(output_dir))
    tokenizer.save_pretrained(output_dir)

    if config["training"].get("do_final_eval", True):
        metrics = latest_eval_metrics_for_current_step(trainer)
        if metrics is None:
            metrics = trainer.evaluate()
            trainer.log_metrics("eval", metrics)
        else:
            print(
                f"Reusing eval metrics already computed at final global_step={trainer.state.global_step}; "
                "skipping duplicate final evaluate()."
            )
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
