"""Resolve final or numbered Trainer checkpoints without importing training code."""
import re
from pathlib import Path

WEIGHT_FILE_NAMES = {
    "pytorch_model.bin", "pytorch_model.bin.index.json",
    "model.safetensors", "model.safetensors.index.json",
    "adapter_model.bin", "adapter_model.safetensors",
}


def has_weight_payload(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(p.name in WEIGHT_FILE_NAMES or
               (p.is_dir() and p.name.startswith("pytorch_model_fsdp_"))
               for p in path.iterdir())


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def resolve_checkpoint_path(model_path: Path) -> Path:
    if not model_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {model_path}")
    if has_weight_payload(model_path):
        return model_path
    candidates = [p for p in model_path.iterdir()
                  if checkpoint_step(p) >= 0 and has_weight_payload(p)]
    if candidates:
        return max(candidates, key=checkpoint_step)
    raise FileNotFoundError(f"No weight payload found under {model_path}")
