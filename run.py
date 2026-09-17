"""Portable launcher. All relative paths are relative to this project, not cwd."""
import argparse
import json
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from config import load_config
import yaml


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    train = sub.add_parser("train")
    train.add_argument("--model", choices=["qwen3", "gpt-oss"], default="qwen3")
    train.add_argument("--dataset", choices=["gsm8k", "math", "commonsenseqa"], default="gsm8k")
    train.add_argument("--method", choices=["temporal", "spatio-temporal"], required=True)
    train.add_argument("--run-dir", required=True)
    train.add_argument("--resume", help="Explicit checkpoint directory, with its optimizer state.")
    evaluate = sub.add_parser("eval", help="Generate benchmark answers and replay cache policies.")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--checkpoint", help="Optional explicit checkpoint; default: latest/final.")
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--max-eval-samples", type=int)
    evaluate.add_argument("--max-new-tokens", type=int, default=512)
    evaluate.add_argument("--refine-mode", choices=["topk", "replace_lowest"], default="replace_lowest")
    evaluate.add_argument("--refine-budget", type=int, help="Default: 15 for Qwen3, 6 for GPT-OSS.")
    for command in (train, evaluate):
        command.add_argument("--gpus", type=int, default=8)
        command.add_argument("--port", type=int, default=29500)
        command.add_argument("--dry-run", action="store_true")
    train.add_argument("--config", action="append", default=[], help="Extra YAML overrides, applied last.")
    return p


def make_plan(args):
    if args.gpus < 1:
        raise ValueError("--gpus must be positive")
    run_dir = project_path(args.run_dir)
    if args.action == "train":
        paths = [ROOT / "configs/base.yaml",
                 ROOT / f"configs/datasets/{args.dataset}.yaml",
                 ROOT / f"configs/models/{args.model}.yaml",
                 ROOT / f"configs/methods/{args.method}.yaml"]
        paths += [project_path(path) for path in args.config]
        config = load_config(paths)
        config["training"]["output_dir"] = str(run_dir)
        config["training"]["resume_from_checkpoint"] = (
            str(project_path(args.resume)) if args.resume else None)
        if args.resume and not project_path(args.resume).is_dir():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")
        script_args = [str(ROOT / "src/train.py")]
    else:
        saved = run_dir / "effective_config.yaml"
        if not saved.is_file():
            raise FileNotFoundError(f"Missing saved training configuration: {saved}")
        config = load_config(saved)
        dataset = config["dataset"]["format"]
        if dataset not in {"gsm8k", "math", "commonsenseqa"}:
            raise ValueError(f"Unsupported evaluation dataset format: {dataset}")
        from checkpoints import resolve_checkpoint_path
        checkpoint = resolve_checkpoint_path(project_path(args.checkpoint) if args.checkpoint else run_dir)
        budget = args.refine_budget
        if budget is None:
            budget = 6 if "gpt-oss" in config["model"]["name"] else 15
        capacity = int(config["cache_moe"]["cache_size"])
        if args.refine_mode == "replace_lowest" and not 0 <= budget <= capacity:
            raise ValueError("--refine-budget must be between 0 and cache_size")
        script_args = [str(ROOT / f"src/eval_{dataset}_accuracy.py"),
                       "--model-path", str(checkpoint),
                       "--output-dir", str(project_path(args.output_dir)),
                       "--skip-reference-loss", "--max-new-tokens", str(args.max_new_tokens),
                       "--cache-policy", "auto",
                       "--spatio-temporal-refine-mode", args.refine_mode,
                       "--spatio-temporal-refine-budget", str(budget)]
        if args.max_eval_samples is not None:
            if args.max_eval_samples < 1:
                raise ValueError("--max-eval-samples must be positive")
            script_args += ["--max-eval-samples", str(args.max_eval_samples)]
        if args.max_new_tokens < 1:
            raise ValueError("--max-new-tokens must be positive")
        config["training"]["output_dir"] = str(project_path(args.output_dir))
    # One process still needs an FSDP distributed context when reading FSDP checkpoints.
    command = [sys.executable, "-m", "accelerate.commands.launch",
               "--config_file", str(ROOT / "accelerate/fsdp.yaml"),
               "--num_processes", str(args.gpus), "--main_process_port", str(args.port)]
    return config, command, script_args


def main():
    args = parser().parse_args()
    config, command, script_args = make_plan(args)
    if args.dry_run:
        print(json.dumps({"project_root": str(ROOT), "config": config,
                          "command": command + script_args + ["--config", "<generated-config.yaml>"]},
                         indent=2))
        return
    output = project_path(args.run_dir if args.action == "train" else args.output_dir)
    if output.exists() and any(output.iterdir()) and not (args.action == "train" and args.resume):
        raise FileExistsError(f"Output is not empty: {output}. Choose a new directory or use --resume.")
    output.mkdir(parents=True, exist_ok=True)
    # Keep the snapshot next to the outputs, not in the original development tree.
    snapshot = output / ("launch_config.yaml" if args.action == "train" else "evaluation_config.yaml")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output, delete=False) as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        temporary = Path(handle.name)
    os.replace(temporary, snapshot)
    metadata = {
        "python": sys.version,
        "packages": {name: version(name) for name in
                     ("torch", "transformers", "accelerate", "datasets", "PyYAML")},
        "command": command + script_args + ["--config", str(snapshot)],
    }
    (output / ("launch_environment.json" if args.action == "train" else "evaluation_environment.json")).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    environment = os.environ.copy()
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run(command + script_args + ["--config", str(snapshot)], cwd=ROOT,
                   env=environment, check=True)


if __name__ == "__main__":
    main()
