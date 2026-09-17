# Reproduction guidance for coding agents

This repository is already the standalone release root. Do not search for or depend on its former parent development project.

For a reproduction request:

1. Read `README.md`, then `configs/README.md` and `VALIDATION.md`.
2. Install the pinned environment exactly as described in `README.md`; use Linux and CUDA-capable NVIDIA GPUs for real training.
3. Run `python -m unittest discover -s tests -v` before editing source code.
4. Start from one documented `python run.py train ... --dry-run` command and inspect the merged configuration.
5. Remove `--dry-run` only after confirming the requested model, dataset, method, GPU count and output directory.
6. Evaluate with `python run.py eval --run-dir ...`; this reuses `effective_config.yaml` and discovers the final/latest valid checkpoint.

The public name **Temporal Router** corresponds to historical internal names containing `future_`. Do not rename those state-dict keys: old checkpoints depend on them. `spatio_temporal_enabled: false` selects Temporal Router; `true` selects Spatio-Temporal Router.

Treat `outputs/`, `eval_outputs/`, model weights, datasets and credentials as local artifacts. Never commit them. Full experiments are expensive full-parameter BF16 runs; do not launch one merely to validate packaging. Cache results are routing-trace replay metrics, not measurements from a physical expert-offloading runtime.
