"""Offline CPU tests: no pretrained weights or benchmark downloads."""
import ast
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import run
from checkpoints import resolve_checkpoint_path
from config import load_config


class ProjectTests(unittest.TestCase):
    def test_sources_parse_and_local_imports_exist(self):
        paths = list((ROOT / "src").glob("*.py")) + [ROOT / "run.py"]
        for path in paths:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_all_recipes(self):
        for model in ("qwen3", "gpt-oss"):
            for dataset in ("gsm8k", "math", "commonsenseqa"):
                for method in ("temporal", "spatio-temporal"):
                    args = run.parser().parse_args([
                        "train", "--method", method, "--model", model,
                        "--dataset", dataset, "--run-dir", "outputs/test", "--dry-run"])
                    cfg, command, script = run.make_plan(args)
                    self.assertTrue(Path(cfg["training"]["output_dir"]).is_absolute())
                    self.assertTrue(Path(script[0]).is_file())
                    self.assertEqual(cfg["dataset"]["format"], dataset)
                    self.assertEqual(cfg["cache_moe"]["spatio_temporal_enabled"],
                                     method == "spatio-temporal")
                    self.assertEqual(cfg["cache_moe"]["cache_size"], 20 if model == "qwen3" else 8)
                    self.assertIn("--num_processes", command)

    def test_outside_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "run.py"), "train", "--method", "temporal",
                 "--run-dir", "outputs/test", "--dry-run"], cwd=directory,
                check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout)["project_root"], str(ROOT))
            self.assertFalse((Path(directory) / "outputs").exists())

    def test_checkpoint_selection_ignores_empty_newer_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = root / "checkpoint-2"
            complete.mkdir()
            (complete / "model.safetensors").touch()
            (root / "checkpoint-10").mkdir()
            self.assertEqual(resolve_checkpoint_path(root), complete)
            (root / "model.safetensors").touch()
            self.assertEqual(resolve_checkpoint_path(root), root)

    def test_entrypoint_imports(self):
        for name in ("train", "eval_gsm8k_accuracy", "eval_math_accuracy",
                     "eval_commonsenseqa_accuracy", "train_promoe_predictor",
                     "train_finemoe_store", "check_cache_moe_checkpoint"):
            importlib.import_module(name)

    def test_warmup_is_not_silently_dropped(self):
        from train import build_training_args
        with tempfile.TemporaryDirectory() as directory:
            cfg = load_config(ROOT / "configs/base.yaml")
            cfg["training"].update(output_dir=directory, bf16=False, tf32=False)
            args = build_training_args(cfg)
            self.assertEqual(args.get_warmup_steps(1000), 30)


class RouterTests(unittest.TestCase):
    def test_forward_backward_checkpoint_and_generation(self):
        import torch
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
        from cache_moe import CacheMoEConfig, install_cache_moe

        torch.set_num_threads(1)
        torch.manual_seed(42)
        cfg = Qwen3MoeConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            moe_intermediate_size=16, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            num_experts=4, num_experts_per_tok=2, max_position_embeddings=64,
            attention_dropout=0.0, pad_token_id=0, eos_token_id=2)
        cfg._attn_implementation = "eager"
        original = Qwen3MoeForCausalLM(cfg).eval()
        tokens = torch.tensor([[3, 4, 5, 6, 7, 8]])
        mask = torch.ones_like(tokens)
        with torch.no_grad():
            expected = original(tokens, attention_mask=mask).logits
        for spatial in (False, True):
            with self.subTest(spatio_temporal=spatial):
                model = copy.deepcopy(original)
                method = CacheMoEConfig(enabled=True, cache_size=3, temperature=0.2,
                                        balance_weight=0, spatio_temporal_enabled=spatial)
                install_cache_moe(model, method)
                output = model(tokens, attention_mask=mask, labels=tokens)
                torch.testing.assert_close(output.logits, expected)
                aux = model.compute_cache_moe_aux(mask, compute_hard_metrics=True)
                self.assertTrue(torch.isfinite(aux["swap_loss"]).item())
                (output.loss + 0.1 * aux["swap_loss"]).backward()
                temporal = [p for n, p in model.named_parameters() if "future_gate" in n]
                self.assertTrue(temporal)
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in temporal))
                if spatial:
                    heads = [p for n, p in model.named_parameters() if "spatio_gate" in n]
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in heads))
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / "weights.pt"
                    torch.save(model.state_dict(), checkpoint)
                    restored = copy.deepcopy(original)
                    install_cache_moe(restored, method)
                    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(value, restored.state_dict()[key])
                model.reset_cache_moe_state()
                generated = model.generate(tokens, attention_mask=mask, max_new_tokens=2,
                                           do_sample=False, use_cache=True)
                self.assertGreater(generated.shape[1], tokens.shape[1])
                from cache_policy_baselines import CachePolicyConfig
                from eval_gsm8k_accuracy import evaluate_hard_cache_metrics_pair
                full, decode = evaluate_hard_cache_metrics_pair(
                    model, 3, torch.device("cpu"),
                    CachePolicyConfig(spatio_temporal_refine_mode="replace_lowest",
                                      spatio_temporal_refine_budget=2), tokens.shape[1])
                for metrics in (full, decode):
                    self.assertTrue(all(torch.isfinite(v).all() for v in metrics.values()
                                        if isinstance(v, torch.Tensor)))

    def test_real_trainer_one_step_and_checkpoint(self):
        import torch
        from datasets import Dataset
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, TrainingArguments
        from cache_moe import CacheMoEConfig, install_cache_moe
        from collator import SupervisedDataCollator
        from trainer import MoESFTTrainer

        torch.set_num_threads(1)
        for spatial in (False, True):
            with self.subTest(spatio_temporal=spatial), tempfile.TemporaryDirectory() as directory:
                cfg = Qwen3MoeConfig(
                    vocab_size=64, hidden_size=32, intermediate_size=64,
                    moe_intermediate_size=16, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                    num_experts=4, num_experts_per_tok=2, pad_token_id=0)
                cfg._attn_implementation = "eager"
                model = Qwen3MoeForCausalLM(cfg)
                install_cache_moe(model, CacheMoEConfig(
                    enabled=True, cache_size=3, temperature=0.2,
                    balance_weight=0, spatio_temporal_enabled=spatial))
                data = Dataset.from_dict({
                    "input_ids": [[3, 4, 5, 6], [7, 8, 9, 10]],
                    "attention_mask": [[1, 1, 1, 1]] * 2,
                    "labels": [[-100, -100, 5, 6], [-100, -100, 9, 10]],
                })
                args = TrainingArguments(
                    output_dir=directory, use_cpu=True, max_steps=1,
                    per_device_train_batch_size=2, report_to="none",
                    save_strategy="no", optim="adamw_torch",
                    remove_unused_columns=False, dataloader_pin_memory=False)
                trainer = MoESFTTrainer(
                    model=model, args=args, train_dataset=data,
                    data_collator=SupervisedDataCollator(pad_token_id=0))
                trainer.train()
                self.assertEqual(trainer.state.global_step, 1)
                trainer.save_model(directory)
                self.assertEqual(resolve_checkpoint_path(Path(directory)), Path(directory))
                from eval_gsm8k_accuracy import load_trained_weights
                restored = Qwen3MoeForCausalLM(cfg)
                install_cache_moe(restored, model._cache_moe_config)
                reader = MoESFTTrainer(model=restored, args=args)
                load_trained_weights(reader, Path(directory))
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, restored.state_dict()[key])

    def test_gpt_oss_router_backward(self):
        import torch
        from transformers import GptOssConfig, GptOssForCausalLM
        from cache_moe import CacheMoEConfig, install_cache_moe
        torch.set_num_threads(1)
        for spatial in (False, True):
            with self.subTest(spatio_temporal=spatial):
                cfg = GptOssConfig(
                    vocab_size=64, hidden_size=32, intermediate_size=16,
                    num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, head_dim=8, num_local_experts=4,
                    num_experts_per_tok=2, max_position_embeddings=131072,
                    sliding_window=8, pad_token_id=0,
                    layer_types=["full_attention", "sliding_attention"])
                cfg._attn_implementation = "eager"
                model = GptOssForCausalLM(cfg)
                install_cache_moe(model, CacheMoEConfig(
                    enabled=True, cache_size=3, temperature=0.2,
                    spatio_temporal_enabled=spatial))
                tokens = torch.tensor([[3, 4, 5, 6]])
                output = model(tokens, labels=tokens)
                aux = model.compute_cache_moe_aux(torch.ones_like(tokens))
                loss = output.loss + 0.1 * aux["swap_loss"]
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                heads = [p for name, p in model.named_parameters() if "future_gate" in name]
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in heads))


if __name__ == "__main__":
    unittest.main()
