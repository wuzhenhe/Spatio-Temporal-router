# Temporal Router 与 Spatio-Temporal Router

这是可独立上传到 GitHub 的项目目录，不依赖外层 `qwen3_moe_mmlu_sft`。完整参数、数据集说明、恢复训练和结果解释见 [英文 README](README.md)。

本文统一使用 **Temporal Router**，即以前的 Future Router。源码中的 `future_gate` 等历史字段保留，用于兼容参数命名；Temporal Option MoE 是另一种对比方法。

## 安装

在包含 `run.py` 的项目根目录执行。训练使用 Linux、Python 3.11 和 NVIDIA GPU；下面安装 CUDA 12.8 对应的 PyTorch，请确保驱动兼容。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.device_count())"
```

模型和数据集首次运行时自动从 Hugging Face 下载，需要网络、足够磁盘和主机内存。默认是 BF16 全参数训练，不是 LoRA；原实验环境为 8 张约 140 GB 显存的 GPU。不要把“小缓存容量”理解为仅需少量 GPU 显存。

## 训练

Temporal Router：

```bash
python run.py train --model qwen3 --dataset gsm8k --method temporal --run-dir outputs/qwen3-gsm8k-temporal --gpus 8
```

Spatio-Temporal Router：

```bash
python run.py train --model qwen3 --dataset gsm8k --method spatio-temporal --run-dir outputs/qwen3-gsm8k-spatio-temporal --gpus 8
```

任一训练命令末尾加 `--dry-run`，即可只查看合并配置和启动命令，不加载模型、不启动训练。默认 4 个 epoch，每卡 batch size 为 1，梯度累积 8，8 卡有效 batch size 为 64。默认按 epoch 保存，保留最近两个 checkpoint。

`--model` 可选 `qwen3`、`gpt-oss`；`--dataset` 可选 `gsm8k`、`math`、`commonsenseqa`。每种组合使用独立输出目录。GPT-OSS 自动启用 MXFP4 反量化后全参数训练。

## 推理与评估

Temporal Router：

```bash
python run.py eval --run-dir outputs/qwen3-gsm8k-temporal --output-dir eval_outputs/qwen3-gsm8k-temporal --gpus 8 --max-new-tokens 512
```

Spatio-Temporal Router：

```bash
python run.py eval --run-dir outputs/qwen3-gsm8k-spatio-temporal --output-dir eval_outputs/qwen3-gsm8k-spatio-temporal --gpus 8 --max-new-tokens 512 --refine-mode replace_lowest --refine-budget 15
```

这一步实际生成答案、计算正确率，并回放专家路由轨迹统计缓存指标。自动读取训练保存的配置和最终权重/最新 checkpoint，无需填写固定步数。先试跑时可加 `--max-eval-samples 16`，并使用新的评估输出目录。

MATH 示例生成长度用 1024，CommonsenseQA 用 64。GPT-OSS 的空间更新预算默认 6，Qwen3 默认 15；可显式修改 `--refine-budget`，或使用 `--refine-mode topk`。这些推理预设应与最终论文实验设置核对，不能混用后直接比较。

结果在指定评估目录：`predictions.jsonl` 是逐条生成答案，`metrics.json` 是汇总指标，`expert_usage.pt` 是专家使用统计。训练和评估同时保存配置、启动命令及主要包版本。

注意：这里不是实际的专家换入换出引擎，缓存指标来自真实生成轨迹的策略回放。带宽/算力推算的延迟不是实测加速；推理时每个进程持有完整模型副本。

## 路径与上传

所有启动器相对路径均相对于本项目根目录，支持绝对路径。现有非空输出目录默认拒绝覆盖。恢复训练使用同一配置并添加 `--resume` 指向实际的 checkpoint 目录；完整说明见英文 README。

直接将本目录内容作为全新匿名 GitHub 仓库的根目录上传，不要复用原始项目的 Git 历史，也不要上传环境、模型和实验输出。匿名审稿版本有意不包含论文标题、作者、机构、个人仓库链接、引用和许可证署名；审稿结束后再按会议政策补充。提交前还需检查匿名账号、commit 作者邮箱和远程仓库地址，详见 [匿名检查清单](ANONYMITY.md)。对比方法源码已保留，不在 README 展开其运行命令。

已经完成必要的离线小模型检查，未在本机重跑完整多 GPU 实验；具体边界见 [验证说明](VALIDATION.md)。
