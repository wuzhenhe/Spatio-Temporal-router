from __future__ import annotations

import argparse
import contextlib
import gc
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from promoe_predictor import (
    PROMOE_PREDICTOR_FORMAT,
    ProMoELayerLogitsMLP,
    build_promoe_layer_mapping,
    infer_promoe_hidden_dim,
    make_promoe_model_key,
    promoe_mlp_parameter_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ProMoE's learned layer-input two-layer MLP predictors from saved traces."
    )
    parser.add_argument(
        "--trace-input",
        action="append",
        help="Torch .pt trace file from eval --promoe-trace-output. Can be passed multiple times.",
    )
    parser.add_argument("--output", required=True, help="Output torch .pt predictor path.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap for smoke tests.")
    parser.add_argument(
        "--max-events-per-predictor",
        type=int,
        default=None,
        help="Optional event cap per source/target predictor for smoke tests.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--eval-batch-size", type=int, default=0, help="0 reuses --batch-size.")
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--amp-dtype",
        choices=["auto", "bf16", "fp16", "none"],
        default="auto",
        help="Mixed precision for predictor matmuls on CUDA. auto uses bf16 when supported, else fp16.",
    )
    parser.add_argument(
        "--random-split",
        action="store_true",
        help="Use a random 9:1 split. Default uses a contiguous 9:1 split to avoid large index-copy overhead.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Paper uses a 9:1 train/eval split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden width for each two-layer MLP. Default is inferred for about 2M params per predictor.",
    )
    parser.add_argument(
        "--target-params-per-predictor",
        type=int,
        default=2_000_000,
        help="Used only when --hidden-dim is omitted; ProMoE reports about 2M params per layer predictor.",
    )
    parser.add_argument("--layer-predict-interval", type=int, default=1)
    parser.add_argument("--layer-predict-max-window", type=int, default=3)
    parser.add_argument("--layer-predict-use-last-output", dest="layer_predict_use_last_output", action="store_true")
    parser.add_argument("--no-layer-predict-use-last-output", dest="layer_predict_use_last_output", action="store_false")
    parser.set_defaults(layer_predict_use_last_output=True)
    parser.add_argument(
        "--num-predict-expert-per-layer",
        type=int,
        default=0,
        help="Prefetch top-k stored in metadata. 0 means infer from the model's selected-expert width.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print collect/start/epoch details for each predictor.")
    return parser.parse_args()


def _load_trace_samples(path: Path) -> List[Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "samples" in payload:
        samples = payload["samples"]
    elif isinstance(payload, list):
        samples = payload
    else:
        raise ValueError(f"Unsupported trace payload in {path}: expected dict with 'samples' or a list.")
    if not isinstance(samples, list):
        raise ValueError(f"Trace samples in {path} are not a list.")
    return samples


def _valid_layers(sample: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
    layers = []
    for layer in sample.get("layers", []):
        selected = layer.get("selected_experts")
        promoe_inputs = layer.get("promoe_inputs")
        if selected is None or promoe_inputs is None:
            continue
        if selected.dim() != 2 or promoe_inputs.dim() != 2:
            continue
        if selected.size(0) == 0 or promoe_inputs.size(0) == 0:
            continue
        layers.append(
            {
                "layer_idx": int(layer["layer_idx"]),
                "num_experts": int(layer.get("num_experts", 0) or (int(selected.max().item()) + 1)),
                "selected_experts": selected.detach().to(dtype=torch.long, device="cpu"),
                "promoe_inputs": promoe_inputs.detach().to(device="cpu")
                if torch.is_floating_point(promoe_inputs)
                else promoe_inputs.detach().to(dtype=torch.float32, device="cpu"),
            }
        )
    layers.sort(key=lambda item: int(item["layer_idx"]))
    return layers


def _preprocess_trace_samples(
    samples: List[Dict[str, Any]],
    max_samples: Optional[int],
) -> List[List[Dict[str, torch.Tensor]]]:
    processed: List[List[Dict[str, torch.Tensor]]] = []
    for sample in samples:
        if max_samples is not None and len(processed) >= int(max_samples):
            break
        layers = _valid_layers(sample)
        if not layers:
            continue
        processed.append(layers)
    return processed


def _infer_trace_shape(processed_samples: List[List[Dict[str, torch.Tensor]]]) -> Tuple[int, int, int, int]:
    for layers in processed_samples:
        num_layers = len(layers)
        input_dim = int(layers[0]["promoe_inputs"].size(-1))
        num_experts = int(layers[0]["num_experts"])
        selected_width = int(layers[0]["selected_experts"].size(-1))
        return num_layers, input_dim, num_experts, selected_width
    raise ValueError("No valid ProMoE trace samples with selected_experts and promoe_inputs were found.")


def _multi_hot(selected: torch.Tensor, num_experts: int) -> torch.Tensor:
    labels = torch.zeros((selected.size(0), num_experts), dtype=torch.bool, device="cpu")
    labels.scatter_(1, selected.to(dtype=torch.long).clamp(0, num_experts - 1), True)
    return labels


def _collect_pair_events(
    processed_samples: List[List[Dict[str, torch.Tensor]]],
    source_layer_pos: int,
    target_layer_pos: int,
    num_layers: int,
    input_dim: int,
    num_experts: int,
    max_events: Optional[int],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    inputs: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    event_count = 0

    for layers in processed_samples:
        if len(layers) <= target_layer_pos or len(layers) != num_layers:
            continue

        target_selected = layers[target_layer_pos]["selected_experts"]
        if source_layer_pos == num_layers:
            source_inputs = layers[-1]["promoe_inputs"]
            if source_inputs.size(0) <= 0 or target_selected.size(0) <= 1:
                continue
            length = min(int(source_inputs.size(0)), int(target_selected.size(0) - 1))
            pair_inputs = source_inputs[:length]
            pair_selected = target_selected[1 : length + 1]
        else:
            if len(layers) <= source_layer_pos:
                continue
            source_inputs = layers[source_layer_pos]["promoe_inputs"]
            length = min(int(source_inputs.size(0)), int(target_selected.size(0)))
            if length <= 0:
                continue
            pair_inputs = source_inputs[:length]
            pair_selected = target_selected[:length]

        if pair_inputs.size(-1) != input_dim:
            continue
        if max_events is not None:
            remaining = max(0, int(max_events) - event_count)
            if remaining <= 0:
                break
            pair_inputs = pair_inputs[:remaining]
            pair_selected = pair_selected[:remaining]

        inputs.append(pair_inputs.detach())
        labels.append(_multi_hot(pair_selected, num_experts))
        event_count += int(pair_inputs.size(0))

    if not inputs:
        return None, None, event_count
    return torch.cat(inputs, dim=0), torch.cat(labels, dim=0), event_count


def _topk_hit_rate(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    if scores.numel() == 0 or labels.numel() == 0:
        return 0.0
    k = max(1, min(int(k), int(scores.size(-1))))
    predicted = torch.topk(scores, k=k, dim=-1, largest=True).indices
    label_values = labels.float()
    gathered = label_values.gather(1, predicted)
    denom = label_values.sum(dim=-1).clamp_min(1.0)
    return float((gathered.sum(dim=-1) / denom).mean().item())


def _resolve_amp_dtype(device: torch.device, amp_dtype: str) -> Optional[torch.dtype]:
    if device.type != "cuda" or amp_dtype == "none":
        return None
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "fp16":
        return torch.float16
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
    if torch.cuda.is_available() and is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype]):
    if device.type == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return contextlib.nullcontext()


def _configure_runtime(args: argparse.Namespace) -> torch.device:
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device requested CUDA, but CUDA is not available.")
        torch.cuda.set_device(device.index or 0)
        torch.backends.cuda.matmul.allow_tf32 = True
    return device


def _prepare_batch_inputs(
    batch_inputs: torch.Tensor,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    batch_inputs = batch_inputs.to(device=device)
    if amp_dtype is None:
        batch_inputs = batch_inputs.to(dtype=torch.float32)
    return batch_inputs


def _iter_tensor_batches(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
):
    row_count = int(inputs.size(0))
    step = max(1, int(batch_size))
    for start in range(0, row_count, step):
        end = min(start + step, row_count)
        yield inputs[start:end], labels[start:end]


def _evaluate_model(
    model: ProMoELayerLogitsMLP,
    eval_inputs: Optional[torch.Tensor],
    eval_labels: Optional[torch.Tensor],
    loss_fn: nn.Module,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    batch_size: int,
    topk: int,
) -> Tuple[float, float, int]:
    if eval_inputs is None or eval_labels is None or eval_inputs.size(0) <= 0:
        return 0.0, 0.0, 0

    model.eval()
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_hit = torch.zeros((), device=device, dtype=torch.float32)
    total_rows = 0
    k = max(1, min(int(topk), int(eval_labels.size(-1))))

    with torch.no_grad():
        for batch_inputs, batch_labels in _iter_tensor_batches(eval_inputs, eval_labels, batch_size=batch_size):
            batch_inputs = _prepare_batch_inputs(batch_inputs, device=device, amp_dtype=amp_dtype)
            batch_labels = batch_labels.to(device=device, dtype=torch.float32)
            with _autocast_context(device, amp_dtype):
                scores = model(batch_inputs)
            scores = scores.float()
            loss = loss_fn(scores, batch_labels)
            predicted = torch.topk(scores, k=k, dim=-1, largest=True).indices
            gathered = batch_labels.gather(1, predicted)
            denom = batch_labels.sum(dim=-1).clamp_min(1.0)
            batch_size_actual = int(batch_inputs.size(0))
            total_loss = total_loss + loss.detach() * batch_size_actual
            total_hit = total_hit + (gathered.sum(dim=-1) / denom).sum()
            total_rows += batch_size_actual

    if total_rows <= 0:
        return 0.0, 0.0, 0
    return (
        float((total_loss / float(total_rows)).detach().cpu().item()),
        float((total_hit / float(total_rows)).detach().cpu().item()),
        total_rows,
    )


def _train_one_predictor(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    input_dim: int,
    num_experts: int,
    hidden_dim: int,
    args: argparse.Namespace,
    topk: int,
    log_prefix: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    device = torch.device(args.device)
    event_count = int(inputs.size(0))
    eval_count = int(round(event_count * float(args.eval_ratio)))
    eval_count = min(max(eval_count, 1 if event_count > 1 else 0), event_count - 1 if event_count > 1 else 0)
    if args.random_split and eval_count > 0:
        indices = torch.randperm(event_count)
        eval_indices = indices[:eval_count]
        train_indices = indices[eval_count:]
        train_inputs = inputs[train_indices]
        train_labels = labels[train_indices]
        eval_inputs = inputs[eval_indices]
        eval_labels = labels[eval_indices]
        split_mode = "random"
    elif eval_count > 0:
        split_at = event_count - eval_count
        train_inputs = inputs[:split_at]
        train_labels = labels[:split_at]
        eval_inputs = inputs[split_at:]
        eval_labels = labels[split_at:]
        split_mode = "contiguous"
    else:
        train_inputs = inputs
        train_labels = labels
        eval_inputs = None
        eval_labels = None
        split_mode = "none"

    amp_dtype = _resolve_amp_dtype(device, str(args.amp_dtype))
    eval_batch_size = int(args.eval_batch_size) if int(args.eval_batch_size) > 0 else int(args.batch_size)

    model = ProMoELayerLogitsMLP(input_dim=input_dim, num_experts=num_experts, hidden_dim=hidden_dim).to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = nn.BCEWithLogitsLoss()

    last_loss = 0.0
    if bool(args.verbose):
        print(
            f"{log_prefix} start: events={event_count}, train_events={train_inputs.size(0)}, "
            f"eval_events={eval_count}, batch_size={int(args.batch_size)}, amp={amp_dtype}, split={split_mode}",
            flush=True,
        )
    for epoch_idx in range(max(1, int(args.epochs))):
        epoch_start = time.time()
        model.train()
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        total_seen = 0
        for batch_inputs, batch_labels in _iter_tensor_batches(
            train_inputs,
            train_labels,
            batch_size=int(args.batch_size),
        ):
            batch_inputs = _prepare_batch_inputs(batch_inputs, device=device, amp_dtype=amp_dtype)
            batch_labels = batch_labels.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, amp_dtype):
                logits = model(batch_inputs)
            loss = loss_fn(logits.float(), batch_labels)
            loss.backward()
            optimizer.step()
            batch_size = int(batch_inputs.size(0))
            total_loss = total_loss + loss.detach() * batch_size
            total_seen += batch_size
        if total_seen > 0:
            last_loss = float((total_loss / float(total_seen)).detach().cpu().item())
        if bool(args.verbose):
            print(
                f"{log_prefix} epoch {epoch_idx + 1}/{max(1, int(args.epochs))}: "
                f"train_loss={last_loss:.6f}, seconds={time.time() - epoch_start:.1f}",
                flush=True,
            )

    val_loss, val_hit, actual_eval_count = _evaluate_model(
        model=model,
        eval_inputs=eval_inputs,
        eval_labels=eval_labels,
        loss_fn=loss_fn,
        device=device,
        amp_dtype=amp_dtype,
        batch_size=eval_batch_size,
        topk=topk,
    )

    state_dict = {key: value.detach().to(device="cpu") for key, value in model.state_dict().items()}
    stats = {
        "events": float(event_count),
        "train_events": float(train_inputs.size(0)),
        "eval_events": float(actual_eval_count),
        "train_loss": float(last_loss),
        "eval_loss": float(val_loss),
        "eval_topk_hit_rate": float(val_hit),
        "split_mode": split_mode,
    }
    return state_dict, stats


def main() -> None:
    args = parse_args()
    if not args.trace_input:
        raise ValueError("--trace-input is required.")
    device = _configure_runtime(args)

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    samples: List[Dict[str, Any]] = []
    for trace_input in args.trace_input:
        samples.extend(_load_trace_samples(Path(trace_input)))
    samples.sort(key=lambda item: int(item.get("index", 0)))
    raw_sample_count = len(samples)
    preprocess_start = time.time()
    processed_samples = _preprocess_trace_samples(samples, max_samples=args.max_samples)
    del samples
    gc.collect()

    num_layers, input_dim, num_experts, selected_width = _infer_trace_shape(processed_samples)
    hidden_dim = int(args.hidden_dim) if args.hidden_dim is not None else infer_promoe_hidden_dim(
        input_dim=input_dim,
        num_experts=num_experts,
        target_params=int(args.target_params_per_predictor),
    )
    num_predict = int(args.num_predict_expert_per_layer) or int(selected_width)
    mapping = build_promoe_layer_mapping(
        num_layers=num_layers,
        interval=int(args.layer_predict_interval),
        max_window=int(args.layer_predict_max_window),
        replace_first_input_with_last_output=bool(args.layer_predict_use_last_output),
    )

    models: Dict[str, Dict[str, torch.Tensor]] = {}
    pair_stats: Dict[str, Dict[str, Any]] = {}
    total_events = 0
    topk_hits = []
    pairs = [
        (global_pair_idx, source_pos, target_pos)
        for global_pair_idx, (source_pos, target_pos) in enumerate(
            [(source_pos, target_pos) for source_pos, targets in mapping.items() for target_pos in targets],
            start=1,
        )
    ]
    trained_pair_indices: List[int] = []
    print(
        "Prepared ProMoE trace: "
        f"samples={len(processed_samples)}, num_layers={num_layers}, input_dim={input_dim}, "
        f"num_experts={num_experts}, selected_width={selected_width}, hidden_dim={hidden_dim}, "
        f"predictor_pairs={len(pairs)}, batch_size={int(args.batch_size)}, amp={args.amp_dtype}, "
        f"device={device}, seconds={time.time() - preprocess_start:.1f}",
        flush=True,
    )
    for pair_idx, (global_pair_idx, source_pos, target_pos) in enumerate(pairs, start=1):
        collect_start = time.time()
        inputs, labels, event_count = _collect_pair_events(
            processed_samples=processed_samples,
            source_layer_pos=source_pos,
            target_layer_pos=target_pos,
            num_layers=num_layers,
            input_dim=input_dim,
            num_experts=num_experts,
            max_events=args.max_events_per_predictor,
        )
        if inputs is None or labels is None:
            continue
        model_key = make_promoe_model_key(source_pos, target_pos)
        if bool(args.verbose):
            print(
                f"[{pair_idx}/{len(pairs)}] "
                f"collected ProMoE predictor {model_key}: "
                f"events={event_count}, input_dtype={inputs.dtype}, collect_seconds={time.time() - collect_start:.1f}",
                flush=True,
            )
        state_dict, stats = _train_one_predictor(
            inputs=inputs,
            labels=labels,
            input_dim=input_dim,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            args=args,
            topk=num_predict,
            log_prefix=f"[{pair_idx}/{len(pairs)}] {model_key}",
        )
        models[model_key] = state_dict
        pair_stats[model_key] = stats
        total_events += int(event_count)
        topk_hits.append(float(stats["eval_topk_hit_rate"]))
        trained_pair_indices.append(global_pair_idx)
        print(
            f"[{pair_idx}/{len(pairs)}] "
            f"trained ProMoE predictor {model_key}: "
            f"events={event_count}, eval_topk_hit={stats['eval_topk_hit_rate']:.4f}",
            flush=True,
        )
        del inputs, labels
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    per_model_params = promoe_mlp_parameter_count(input_dim=input_dim, num_experts=num_experts, hidden_dim=hidden_dim)
    predictor = {
        "format": PROMOE_PREDICTOR_FORMAT,
        "config": {
            "predictor_type": "sep",
            "input_mode": "moe_layer_logits",
            "num_layers": num_layers,
            "input_dim": input_dim,
            "num_experts": num_experts,
            "hidden_dim": hidden_dim,
            "target_params_per_predictor": int(args.target_params_per_predictor),
            "params_per_predictor": per_model_params,
            "layer_predict_interval": int(args.layer_predict_interval),
            "layer_predict_max_window": int(args.layer_predict_max_window),
            "layer_predict_replace_first_input_with_last_output": bool(args.layer_predict_use_last_output),
            "num_predict_expert_per_layer": num_predict,
            "eval_ratio": float(args.eval_ratio),
        },
        "models": models,
        "metadata": {
            "trace_inputs": [str(Path(path)) for path in args.trace_input],
            "raw_sample_count": raw_sample_count,
            "sample_count": len(processed_samples),
            "event_count": total_events,
            "model_count": len(models),
            "parameter_count": int(per_model_params * len(models)),
            "mean_eval_topk_hit_rate": float(sum(topk_hits) / len(topk_hits)) if topk_hits else 0.0,
            "global_pair_count": len(pairs),
            "trained_pair_indices": trained_pair_indices,
            "pair_stats": pair_stats,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(predictor, output)
    print(
        "Saved ProMoE predictor: "
        f"path={output}, format={PROMOE_PREDICTOR_FORMAT}, "
        f"models={len(models)}, input_dim={input_dim}, num_experts={num_experts}, hidden_dim={hidden_dim}, "
        f"params={predictor['metadata']['parameter_count']}, events={total_events}",
        flush=True,
    )


if __name__ == "__main__":
    main()
