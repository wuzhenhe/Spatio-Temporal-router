import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import yaml


WEIGHT_FILE_NAMES = {
    "pytorch_model.bin",
    "model.safetensors",
}


def has_weight_payload(path: Path) -> bool:
    if not path.is_dir():
        return False
    names = {child.name for child in path.iterdir()}
    if names.intersection(WEIGHT_FILE_NAMES):
        return True
    return any(child.is_dir() and child.name.startswith("pytorch_model_fsdp_") for child in path.iterdir())


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[-1])
    except ValueError:
        return -1


def resolve_checkpoint_path(model_path: Path) -> Path:
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if model_path.is_file():
        raise ValueError(f"Expected a directory, got file: {model_path}")
    if has_weight_payload(model_path):
        return model_path
    checkpoints = sorted(
        [child for child in model_path.iterdir() if child.is_dir() and child.name.startswith("checkpoint-")],
        key=checkpoint_step,
    )
    if checkpoints:
        return checkpoints[-1]
    raise FileNotFoundError(f"Could not find final model weights or checkpoint-* directories under {model_path}")


def dcp_metadata_keys(path: Path) -> Optional[Set[str]]:
    try:
        from torch.distributed.checkpoint import FileSystemReader
    except Exception:
        return None

    candidates = [path]
    if path.is_dir():
        candidates.extend(child for child in path.iterdir() if child.is_dir() and child.name.startswith("pytorch_model_fsdp_"))

    keys: Set[str] = set()
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        try:
            metadata = FileSystemReader(str(candidate)).read_metadata()
        except Exception:
            continue
        state_metadata = getattr(metadata, "state_dict_metadata", None)
        if isinstance(state_metadata, dict):
            keys.update(str(key) for key in state_metadata.keys())
    return keys or None


def safetensors_keys(path: Path) -> Optional[Set[str]]:
    try:
        from safetensors import safe_open
    except Exception:
        return None

    keys: Set[str] = set()
    files: List[Path] = []
    if path.is_dir():
        files.extend(path.glob("*.safetensors"))
    elif path.suffix == ".safetensors":
        files.append(path)
    for file_path in files:
        try:
            with safe_open(str(file_path), framework="pt", device="cpu") as handle:
                keys.update(str(key) for key in handle.keys())
        except Exception:
            continue
    return keys or None


def collect_checkpoint_keys(path: Path) -> Set[str]:
    keys = dcp_metadata_keys(path)
    if keys is not None:
        return keys
    keys = safetensors_keys(path)
    if keys is not None:
        return keys
    raise RuntimeError(
        "Could not inspect checkpoint keys. For FSDP checkpoints this script expects "
        "torch.distributed.checkpoint metadata; for regular checkpoints it expects safetensors."
    )


def load_effective_config(path: Path) -> Dict:
    candidates = []
    if path.is_dir():
        candidates.append(path / "effective_config.yaml")
        candidates.append(path.parent / "effective_config.yaml")
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
    return {}


def summarize_keys(keys: Iterable[str]) -> Dict[str, object]:
    key_list = sorted(keys)
    future = [key for key in key_list if "future_gate" in key]
    spatio = [key for key in key_list if "spatio_gate" in key]
    temporal_option = [key for key in key_list if ".controller." in key]
    lora = [key for key in key_list if "lora_" in key or ".lora_embedding_" in key]
    router = [
        key for key in key_list
        if ".mlp.gate." in key or ".mlp.original_block.gate." in key or ".original_block.gate." in key
    ]
    return {
        "total_key_count": len(key_list),
        "future_gate_key_count": len(future),
        "spatio_gate_key_count": len(spatio),
        "temporal_option_key_count": len(temporal_option),
        "lora_key_count": len(lora),
        "router_key_count": len(router),
        "first_future_gate_keys": future[:8],
        "first_spatio_gate_keys": spatio[:8],
        "first_temporal_option_keys": temporal_option[:8],
        "first_lora_keys": lora[:8],
        "first_router_keys": router[:8],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a Cache-MoE / temporal-option checkpoint contains method-specific weights."
    )
    parser.add_argument("model_path", help="Output directory or checkpoint directory to inspect.")
    parser.add_argument("--expect-spatio", action="store_true", help="Fail if no spatio_gate keys are found.")
    parser.add_argument(
        "--expect-temporal-option",
        action="store_true",
        help="Fail if no temporal option controller keys are found.",
    )
    parser.add_argument("--expect-lora", action="store_true", help="Fail if no LoRA adapter keys are found.")
    parser.add_argument("--expect-router", action="store_true", help="Fail if no MoE router/gate keys are found.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_path = Path(args.model_path)
    checkpoint_path = resolve_checkpoint_path(requested_path)
    config = load_effective_config(checkpoint_path)
    cache_cfg = config.get("cache_moe", {}) if isinstance(config, dict) else {}
    temporal_option_cfg = config.get("temporal_option_moe", {}) if isinstance(config, dict) else {}
    expect_spatio = bool(
        args.expect_spatio
        or cache_cfg.get("spatio_temporal_enabled", False)
        or cache_cfg.get("spatio_only_enabled", False)
    )
    expect_temporal_option = bool(args.expect_temporal_option or temporal_option_cfg.get("enabled", False))
    expect_lora = bool(args.expect_lora or (temporal_option_cfg.get("enabled", False) and temporal_option_cfg.get("train_lora", False)))
    expect_router = bool(args.expect_router or (temporal_option_cfg.get("enabled", False) and temporal_option_cfg.get("train_router", False)))

    keys = collect_checkpoint_keys(checkpoint_path)
    summary = summarize_keys(keys)
    payload = {
        "requested_path": str(requested_path.resolve()),
        "resolved_checkpoint_path": str(checkpoint_path.resolve()),
        "effective_config_spatio_temporal_enabled": cache_cfg.get("spatio_temporal_enabled"),
        "effective_config_spatio_only_enabled": cache_cfg.get("spatio_only_enabled"),
        "effective_config_temporal_option_enabled": temporal_option_cfg.get("enabled"),
        **summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if expect_spatio and int(summary["spatio_gate_key_count"]) == 0:
        raise SystemExit(
            "Expected spatio_gate keys, but none were found. This is not a valid "
            "spatio-temporal checkpoint. Use the final output directory from the correct run, "
            "or retrain into a clean/new output directory."
        )
    if expect_temporal_option and int(summary["temporal_option_key_count"]) == 0:
        raise SystemExit(
            "Expected temporal option controller keys, but none were found. This is not a valid "
            "temporal-option checkpoint. Use the final output directory from the correct run, "
            "or retrain into a clean/new output directory."
        )
    if expect_lora and int(summary["lora_key_count"]) == 0:
        raise SystemExit(
            "Expected LoRA adapter keys, but none were found. This is not a valid "
            "paper-aligned temporal-option checkpoint."
        )
    if expect_router and int(summary["router_key_count"]) == 0:
        raise SystemExit(
            "Expected MoE router/gate keys, but none were found. This is not a valid "
            "paper-aligned temporal-option checkpoint."
        )


if __name__ == "__main__":
    main()
