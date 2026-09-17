# Configuration reference

`run.py` composes one training configuration in this fixed order:

1. Built-in compatibility defaults from `src/config.py`.
2. `base.yaml`.
3. One dataset profile from `datasets/`.
4. One model profile from `models/`.
5. One method profile from `methods/`.
6. Any YAML files supplied with `--config`, in command-line order.

Later files override earlier values recursively. The launcher saves the resolved configuration as `launch_config.yaml`; training also writes `effective_config.yaml`, which evaluation loads automatically. Do not edit a completed run's snapshot by hand.

## Included profiles

| File | Purpose |
| --- | --- |
| `base.yaml` | Shared four-epoch, full-parameter BF16 SFT recipe with gradient checkpointing, cosine scheduling and epoch checkpoints. |
| `models/qwen3.yaml` | Qwen3-30B-A3B model ID and Qwen3 router defaults: capacity 20, temperature 0.03 and swap weight 0.1. |
| `models/gpt-oss.yaml` | GPT-OSS-20B model ID, MXFP4 dequantization, offset label masking and GPT-OSS router defaults: capacity 8, temperature 0.05 and swap weight 0.01. |
| `datasets/gsm8k.yaml` | GSM8K `main`, train/test splits, 512-token training examples and `#### answer` formatting. |
| `datasets/math.yaml` | MATH-lighteval train/test splits, 2048-token training examples and boxed-answer formatting. |
| `datasets/commonsenseqa.yaml` | CommonsenseQA train/validation splits, 512-token multiple-choice examples. |
| `methods/temporal.yaml` | Temporal Router, called Future Router in historical internal parameter names. Spatial refinement is disabled. |
| `methods/spatio-temporal.yaml` | Spatio-Temporal Router with the additional spatial gate enabled. |

All 2 model x 3 dataset x 2 method combinations are accepted by the launcher. The paper-facing main methods are selected only through `--method temporal` and `--method spatio-temporal`; retained comparison implementations under `src/` do not have release recipe profiles.

## Safe overrides

Create a new local YAML instead of modifying a release profile. For example:

```yaml
training:
  num_train_epochs: 1
dataset:
  max_train_samples: 64
  max_eval_samples: 16
```

Pass it last with `--config configs/local.yaml`. Keep model, dataset, method, GPU count and effective batch size fixed when reproducing a reported run. Run the resulting command once with `--dry-run` and inspect the merged JSON before starting GPU work.
