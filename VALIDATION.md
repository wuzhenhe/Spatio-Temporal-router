# Packaging validation

This is a source-code release, not a rerun of the paper experiments.

## Completed

- Nine offline unittest cases passed on Windows / Python 3.14.4 / PyTorch 2.11.0 CPU / Transformers 5.17.0 / Accelerate 1.15.0 / Datasets 4.8.4.
- All 12 combinations of two models, three datasets and two methods compose correctly.
- Launcher dry-run works outside the project working directory.
- Training, benchmark evaluators and retained comparison preparation modules import successfully.
- Tiny Qwen3 models: unchanged forward logits after router installation, finite auxiliary loss, nonzero auxiliary-head gradients, state-dict round trips, generation and cache-policy replay.
- Tiny GPT-OSS models: both router variants support forward/backward execution.
- Both Qwen3 methods: an actual one-step Trainer run, checkpoint save, evaluation-loader restore and equality of all saved parameters.
- Checkpoint selection ignores newer empty checkpoint directories.

## Packaging fixes

- Removed dependencies on the legacy Tulu evaluation module for checkpoint discovery.
- Added root-relative launch paths, saved training-config reuse, nonempty-output protection and environment metadata.
- Preserved warmup when Transformers 5 replaces warmup_ratio with fractional warmup_steps.
- Preserved fused expert parameter names during non-FSDP saving: Transformers 5's default original-format conversion otherwise breaks Trainer checkpoint restoration.
- Made training metadata writing rank-zero-only and enabled explicit GPT-OSS MXFP4 dequantization for full fine-tuning.
- Kept method mathematics and historical auxiliary-head checkpoint names intact.

## Not verified here

- Full pretrained model downloads, full-data training, Linux CUDA/NCCL/FSDP execution, distributed checkpoint restoration and full benchmark accuracy.
- Exact reproduction of historical paper tables; original immutable model/data revisions and the original complete environment lock were not supplied.
- All optional comparison variants and optional LoRA/quantized training paths.

IDE analysis was unavailable because this session did not expose PyCharm's required `lint_files` interface. No IDE-clean result is claimed. Scope: 21 Python source files (`run.py`, 19 `src/*.py` modules and `tests/test_project.py`). The fixes above were validated by the available offline checks rather than IDE inspections.
