# Temporal and Spatio-Temporal Routers

Standalone research code for training Temporal Router and Spatio-Temporal Router on sparse Mixture-of-Experts language models, generating benchmark answers, and evaluating expert-cache policies on the resulting routing traces.

**Terminology:** Temporal Router is the method previously called Future Router. Historical internal names such as `future_gate`, `future_probs`, and `future_cache_priority_mode` are retained for checkpoint compatibility. Temporal Option MoE is a separate comparison method, not Temporal Router.

**Scope:** generation runs the actual language model. Expert-cache policies are replayed over its recorded routing traces; this repository is not a physical CPU/GPU expert-offloading engine. Cache capacity does not reduce the resident model memory. Reported bandwidth/compute-based latency estimates are proxies, not measured end-to-end acceleration.

## 1. Installation

Download this project from GitHub and open a terminal in the directory containing this README and `run.py`. The folder is self-contained: its parent development repository is not required. The commands below target **Linux, Python 3.11, NVIDIA GPUs and a CUDA-compatible driver**. Windows CPU smoke checks are supported; distributed training should run on Linux.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print('GPUs:', torch.cuda.device_count())"
```

The CUDA 12.8 wheel command follows the [official PyTorch version-specific instructions](https://pytorch.org/get-started/previous-versions/). Use a driver compatible with that runtime. Do not install a CPU-only PyTorch wheel for training. FlashAttention, bitsandbytes and PEFT are not needed for the supplied full-fine-tuning recipes.

Models and datasets download automatically from Hugging Face on first use. Allow network access and enough cache/disk space. If authentication is required, run `hf auth login`; never put credentials in tracked files. You can set `HF_HOME` to a large writable cache directory before launching.

The original experimental setup used 8 GPUs with approximately 140 GB per GPU. These are full-parameter BF16 fine-tuning recipes, not lightweight adapter training. Budget host RAM and several hundred GB of storage for weights, optimizer checkpoints and downloads. Actual requirements depend on sequence length, model and hardware. In evaluation, each process holds a complete model replica; increasing the GPU count does not shard generation memory. Start with the same GPU count used for training when loading FSDP checkpoints.

## 2. Configuration and defaults

Only a small set of composable profiles is included:

```text
run.py                     # portable training / benchmark inference launcher
configs/base.yaml          # shared full-fine-tuning settings
configs/models/            # qwen3, gpt-oss
configs/datasets/          # gsm8k, math, commonsenseqa
configs/methods/            # temporal, spatio-temporal
accelerate/fsdp.yaml        # single-node FSDP1, sharded checkpoints
src/                       # model patches, trainer, evaluators, comparisons
tests/                     # small offline CPU tests
```

| Setting | Qwen3 | GPT-OSS |
| --- | --- | --- |
| Base model | `Qwen/Qwen3-30B-A3B-Instruct-2507` | `openai/gpt-oss-20b` |
| Cache capacity per layer | 20 | 8 |
| Relaxation temperature | 0.03 | 0.05 |
| Swap-loss weight | 0.1 | 0.01 |
| Learning rate | 1e-5 | 5e-6 |
| Epochs | 4 | 4 |

Both methods use linear auxiliary heads, initialized from the corresponding MoE routers. Spatio-Temporal Router additionally enables `spatio_gate`. GPT-OSS is explicitly dequantized from MXFP4 for full-parameter training.

Shared defaults: seed 42, batch size 1 per GPU, gradient accumulation 8, cosine schedule, warmup ratio 0.03, weight decay 0.1, gradient clipping 1.0, gradient checkpointing and eager attention. With 8 GPUs the effective batch size is 64. Checkpoints are saved each epoch and the latest two are retained. This clean packaging uses epoch-based saving rather than dataset-specific historical step numbers.

| Dataset option | Hugging Face dataset | Training / evaluation | Training length |
| --- | --- | --- | --- |
| `gsm8k` | `openai/gsm8k`, subset `main` | train / test | 512 |
| `math` | `DigitalLearningGmbH/MATH-lighteval` | train / test | 2048 |
| `commonsenseqa` | `tau/commonsense_qa` | train / validation | 512 |

Prompts, answer formats and label masking are in the dataset profiles and `src/data.py`. Training uses supervised answer-token loss. Evaluation generates answers and applies the dataset-specific answer extractor/scorer.

## 3. Train Temporal Router

First inspect the command and merged configuration without downloading a model or starting a run:

```bash
python run.py train --model qwen3 --dataset gsm8k --method temporal --run-dir outputs/qwen3-gsm8k-temporal --gpus 8 --dry-run
```

Then train:

```bash
python run.py train --model qwen3 --dataset gsm8k --method temporal --run-dir outputs/qwen3-gsm8k-temporal --gpus 8
```

## 4. Train Spatio-Temporal Router

```bash
python run.py train --model qwen3 --dataset gsm8k --method spatio-temporal --run-dir outputs/qwen3-gsm8k-spatio-temporal --gpus 8
```

The two runs have separate output directories. Existing nonempty directories are rejected unless explicitly resuming. Nothing is uploaded to Hugging Face or GitHub by these commands.

The launcher invokes Accelerate using the active Python interpreter. All relative run/config/output/checkpoint paths are resolved against this project directory, even if `run.py` is invoked from another working directory. Absolute paths are also accepted. Use `--port` to select a different distributed port when running concurrent jobs.

## 5. Inference and evaluation

These commands perform autoregressive generation, answer scoring, and cache-trace evaluation. They load the run's `effective_config.yaml` automatically, so the router architecture, capacity, prompts and base model match training. A final weight payload is preferred; otherwise the latest numbered checkpoint with a weight payload is selected. No hard-coded checkpoint step is needed.

Temporal Router:

```bash
python run.py eval --run-dir outputs/qwen3-gsm8k-temporal --output-dir eval_outputs/qwen3-gsm8k-temporal --gpus 8 --max-new-tokens 512
```

Spatio-Temporal Router:

```bash
python run.py eval --run-dir outputs/qwen3-gsm8k-spatio-temporal --output-dir eval_outputs/qwen3-gsm8k-spatio-temporal --gpus 8 --max-new-tokens 512 --refine-mode replace_lowest --refine-budget 15
```

For an initial 16-example run, use a separate output directory:

```bash
python run.py eval --run-dir outputs/qwen3-gsm8k-spatio-temporal --output-dir eval_outputs/qwen3-gsm8k-spatio-temporal-smoke --gpus 8 --max-new-tokens 512 --max-eval-samples 16
```

Generation is greedy by default. Reference-answer loss is skipped by this launcher; generated-answer accuracy is still computed. `--refine-mode replace_lowest` controls the spatial refinement stage, with default budgets 15 for Qwen3 and 6 for GPT-OSS. This setting has no spatial effect on Temporal Router. `--refine-mode topk` selects the alternative full-priority top-k policy. Keep the policy and budget fixed when comparing results; these are explicit release presets, not a claim that every historical paper table used the same inference variant.

Each evaluation writes:

- `predictions.jsonl`: generated text, extracted answers, correctness and per-example cache statistics.
- `metrics.json`: aggregated accuracy, cache and load statistics, including the `paper_*` summary fields.
- `expert_usage.pt`: recorded aggregate expert usage.
- `evaluation_config.yaml` and `evaluation_environment.json`: effective configuration, command and package versions.

Use decode-only metrics for decoding comparisons. Prefetch traffic must be included when interpreting load-adjusted metrics; raw hit rate alone does not measure total traffic savings. Rank-local intermediate files may also remain in the evaluation directory. Choose a fresh directory for each evaluation to avoid mixing runs.

## 6. Other model and dataset examples

GPT-OSS Temporal Router on GSM8K:

```bash
python run.py train --model gpt-oss --dataset gsm8k --method temporal --run-dir outputs/gpt-oss-gsm8k-temporal --gpus 8
python run.py eval --run-dir outputs/gpt-oss-gsm8k-temporal --output-dir eval_outputs/gpt-oss-gsm8k-temporal --gpus 8 --max-new-tokens 512
```

GPT-OSS Spatio-Temporal Router on MATH:

```bash
python run.py train --model gpt-oss --dataset math --method spatio-temporal --run-dir outputs/gpt-oss-math-spatio-temporal --gpus 8
python run.py eval --run-dir outputs/gpt-oss-math-spatio-temporal --output-dir eval_outputs/gpt-oss-math-spatio-temporal --gpus 8 --max-new-tokens 1024 --refine-budget 6
```

Qwen3 Spatio-Temporal Router on CommonsenseQA:

```bash
python run.py train --model qwen3 --dataset commonsenseqa --method spatio-temporal --run-dir outputs/qwen3-commonsenseqa-spatio-temporal --gpus 8
python run.py eval --run-dir outputs/qwen3-commonsenseqa-spatio-temporal --output-dir eval_outputs/qwen3-commonsenseqa-spatio-temporal --gpus 8 --max-new-tokens 64
```

Both methods support all model/dataset combinations listed above. Use a distinct run directory for each combination. Generation budget is independent of the training token-length limit.

## 7. Overrides, resume and checkpoints

For custom settings, create a YAML file, for example `configs/local.yaml`:

```yaml
training:
  num_train_epochs: 4
  gradient_accumulation_steps: 8
dataset:
  max_train_samples: 64
  max_eval_samples: 16
```

Append `--config configs/local.yaml` to a training command. Overrides apply after base, dataset, model and method profiles. Remove sample limits for full experiments. Changing GPU count, accumulation, truncation or sample limits changes the experiment.

To resume, use the same training command and overrides, adding `--resume` with an existing `checkpoint-N` directory from that run. Select the actual directory on disk, not an assumed historical step. Resume requires optimizer/scheduler state; a final weights-only directory is not enough. Do not change the model, dataset or method when resuming. To evaluate a particular checkpoint, append `--checkpoint` with its directory to the evaluation command.

Keep the complete run directory when transferring checkpoints: `effective_config.yaml`, model/tokenizer files and the entire FSDP checkpoint directory. FSDP shards must not be copied individually. Loading auxiliary-router checkpoints directly with an unpatched `AutoModelForCausalLM.from_pretrained` is not supported; use this project's evaluation entrypoints. Old checkpoints from different Transformers versions or router layouts may need conversion and are not automatically guaranteed compatible.

For local base-model mirrors, override `model.name`, `model.config_name` and `model.tokenizer_name` together. Local paths in saved configurations must be updated if moved to a different machine. Remote model IDs used by the default recipes do not require such edits.

## 8. Comparisons and tests

The source retains LRU/cache-policy comparisons, Window Cache Loss, Temporal Option MoE, ProMoE predictor support and FineMoE store support. Their implementation and optional preparation entrypoints remain in `src/`; they are not the main recipes documented here. Optional LoRA paths require installing PEFT separately and are outside the supplied full-fine-tuning dependency/test scope.

Run the small offline tests without downloading pretrained models:

```bash
python -m unittest discover -s tests -v
```

The offline tests cover recipe composition, paths, tiny Qwen3/GPT-OSS router execution, gradient flow, generation/cache replay and Qwen3 Trainer checkpoint round trips. Full pretrained-model multi-GPU training and complete benchmark scores are outside the test suite.
