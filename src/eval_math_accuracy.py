import argparse
import contextlib
import json
import re
import signal
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import torch

from cache_policy_baselines import (
    build_promoe_trace_sample,
)
from config import load_config, resolve_cache_size
from data import _build_messages, _load_raw_dataset, _render_messages
from eval_gsm8k_accuracy import (
    CACHE_POLICY_CHOICES,
    accumulate_cache_metric_pair,
    accumulate_expert_usage,
    build_cache_policy_config,
    build_eval_args,
    disable_training_router_outputs_for_generation,
    effective_cache_policy_prefetch_budget,
    gather_predictions_from_rank_files,
    gather_metric_tensors_from_rank_files,
    distributed_info,
    evaluate_hard_cache_metrics,
    evaluate_hard_cache_metrics_pair,
    generation_nll_from_scores,
    generation_hard_cache_metrics,
    hard_cache_metric_mode,
    expert_load_summary,
    emphasize_paper_metrics,
    infer_moe_shape,
    infer_routed_expert_size_bytes,
    get_recorded_layer_stats,
    load_promoe_predictor,
    load_trained_weights,
    move_model_for_generation,
    reference_moe_loss_sum,
    reset_recorded_expert_stats,
    resolve_generation_device,
    save_promoe_trace_output,
    save_expert_usage,
    summarize_hard_cache_metrics,
    temporal_option_generation_summary,
    predictor_overhead_summary,
    normalize_cache_policy_name,
    parse_cache_policy_list,
    validate_future_router_loaded,
    validate_future_only_cache_policy_binding,
    summarize_cache_policy_metric_sums,
    _dtype_num_bytes,
    _unwrap_cache_model,
)
from checkpoints import resolve_checkpoint_path
from finemoe_store import finemoe_store_summary, load_finemoe_store
from trainer import MoESFTTrainer
from train import build_model, build_tokenizer
from collator import SupervisedDataCollator

T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MATH answers and compute boxed-answer accuracy.")
    parser.add_argument("--config", action="append", required=True, help="YAML config/profile path.")
    parser.add_argument("--model-path", default=None, help="Run output dir or checkpoint-* dir.")
    parser.add_argument(
        "--skip-checkpoint-load",
        action="store_true",
        help="Evaluate the base model loaded from config.model without loading a fine-tuned checkpoint.",
    )
    parser.add_argument("--output-dir", default="./eval_outputs/math_accuracy", help="Output directory.")
    parser.add_argument("--predictions-output", default=None, help="Optional JSONL prediction path.")
    parser.add_argument("--metrics-output", default=None, help="Optional JSON metrics path.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Limit examples for smoke tests.")
    parser.add_argument("--eval-split", default=None, help="Override dataset.eval_split, e.g. train for trace collection.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Generation budget.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Use 0 for greedy decoding.")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--compute-generation-nll",
        action="store_true",
        help="Also compute NLL of model-generated tokens. Disabled by default to avoid storing vocab scores.",
    )
    parser.add_argument(
        "--skip-reference-loss",
        action="store_true",
        help="Skip the extra supervised reference-loss forward pass; accuracy/cache/load metrics are still computed.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Print progress every N local samples per rank. Use 0 to disable progress logs.",
    )
    parser.add_argument(
        "--expert-usage-output",
        default=None,
        help="Optional torch .pt path for averaged per-layer expert usage counts.",
    )
    parser.add_argument(
        "--no-generation-cache",
        action="store_true",
        help="Disable KV cache during generation. Expert usage recording is cleaner with cache enabled.",
    )
    parser.add_argument(
        "--no-symbolic-match",
        action="store_true",
        help="Disable rank-0 symbolic equivalence matching and report exact boxed-answer accuracy only.",
    )
    parser.add_argument(
        "--symbolic-timeout",
        type=float,
        default=2.0,
        help="Maximum seconds for symbolic matching one sample on rank0. Use 0 to disable. Linux/Unix only.",
    )
    parser.add_argument(
        "--expert-size-mb",
        type=float,
        default=None,
        help="Override routed expert weight size in decimal MB for expert-load estimates.",
    )
    parser.add_argument(
        "--expert-load-bandwidth-gbps",
        type=float,
        default=0.0,
        help=(
            "Optional effective expert-load bandwidth in decimal GB/s. "
            "When positive, report load-only TPOT proxy in ms per generated token."
        ),
    )
    parser.add_argument(
        "--predictor-effective-tflops",
        type=float,
        default=0.0,
        help=(
            "Optional effective compute throughput for predictor overhead estimates. "
            "When positive, report predictor_overhead_ms_per_token."
        ),
    )
    parser.add_argument(
        "--cache-policy",
        choices=CACHE_POLICY_CHOICES,
        default="auto",
        help=(
            "Hard-cache simulation policy. 'auto' keeps the method's native behavior "
            "(future-router cache for cache_moe, LRU otherwise). "
            "lfu/lrfu/least_stale/specmd_prefetch/promoe/finemoe/future_prefetch "
            "are isolated replay baselines and do not affect model generation. "
            "future_cache_priority_mode=future_only must use future_prefetch."
        ),
    )
    parser.add_argument(
        "--extra-cache-policies",
        default=None,
        help=(
            "Comma-separated additional hard-cache replay policies to evaluate on the same generation trace, "
            "for example lru,lfu,lrfu,specmd_prefetch. These policies do not trigger extra generation."
        ),
    )
    parser.add_argument("--cache-policy-recency-weight", type=float, default=1.0)
    parser.add_argument("--cache-policy-frequency-weight", type=float, default=1.0)
    parser.add_argument("--cache-policy-recency-decay", type=float, default=128.0)
    parser.add_argument(
        "--cache-policy-lrfu-lambda",
        type=float,
        default=0.5,
        help="For lrfu, original LRFU CRF decay parameter lambda. 0 approximates LFU; larger values favor recency.",
    )
    parser.add_argument(
        "--cache-policy-prefetch-budget",
        type=int,
        default=-1,
        help=(
            "For promoe/specmd_prefetch, number of experts to prefetch per predicted layer. "
            "For promoe, -1 uses the predictor's trained num_predict_expert_per_layer. "
            "For specmd_prefetch, -1 uses cache_size. "
            "0 disables prefetch; positive values are capped at cache_size."
        ),
    )
    parser.add_argument(
        "--cache-policy-prefetch-lookahead",
        type=int,
        default=3,
        help="For promoe/specmd_prefetch, layer lookahead window; the paper/official ProMoE best config uses 3.",
    )
    parser.add_argument(
        "--spatio-temporal-refine-mode",
        choices=["topk", "replace_lowest"],
        default="topk",
        help=(
            "For spatio_temporal_enabled + --cache-policy auto only. "
            "'topk' replaces the layer cache with top cache_size experts by full priority. "
            "'replace_lowest' takes the top --spatio-temporal-refine-budget experts by full priority "
            "and only inserts the missing ones, evicting the lowest-priority resident experts."
        ),
    )
    parser.add_argument(
        "--spatio-temporal-refine-budget",
        type=int,
        default=-1,
        help=(
            "For --spatio-temporal-refine-mode replace_lowest, number of full-priority top experts "
            "to consider during the second-stage spatio-temporal refinement. "
            "-1 uses cache_size; pass 8 to only conditionally insert missing top-8 experts."
        ),
    )
    parser.add_argument(
        "--promoe-predictor-path",
        default=None,
        help="Torch .pt predictor from src/train_promoe_predictor.py. Required for --cache-policy promoe.",
    )
    parser.add_argument(
        "--promoe-trace-output",
        default=None,
        help="Optional torch .pt path to save selected_experts traces for training a ProMoE-style predictor.",
    )
    parser.add_argument(
        "--finemoe-store-path",
        default=None,
        help="Torch .pt Expert Map Store from src/train_finemoe_store.py. Required for --cache-policy finemoe.",
    )
    parser.add_argument("--finemoe-top-k", type=int, default=8)
    parser.add_argument("--finemoe-candidate-pool", type=int, default=64)
    parser.add_argument("--finemoe-semantic-weight", type=float, default=1.0)
    parser.add_argument("--finemoe-trajectory-weight", type=float, default=1.0)
    parser.add_argument(
        "--finemoe-prefetch-threshold",
        type=float,
        default=-1.0,
        help="Fixed cumulative probability threshold. Negative enables FineMoE-style dynamic thresholding.",
    )
    parser.add_argument("--finemoe-min-prefetch-threshold", type=float, default=0.05)
    parser.add_argument("--finemoe-max-prefetch-threshold", type=float, default=0.35)
    parser.add_argument("--finemoe-eviction-probability-weight", type=float, default=1.0)
    return parser.parse_args()


@contextlib.contextmanager
def time_limit(seconds: float):
    if seconds <= 0 or not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return

    def handle_timeout(signum, frame):
        raise TimeoutError(f"symbolic match exceeded {seconds} seconds")

    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def call_with_timeout(func: Callable[[], T], timeout_seconds: float) -> T:
    with time_limit(timeout_seconds):
        return func()


def extract_braced_content(text: str, command_start: int) -> Optional[str]:
    brace_start = text.find("{", command_start)
    if brace_start < 0:
        return None

    depth = 0
    for pos in range(brace_start, len(text)):
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : pos]
    return None


def extract_boxed_answer(text: str) -> Optional[str]:
    if text is None:
        return None
    candidates = []
    for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        content = extract_braced_content(text, match.start())
        if content is not None and content.strip():
            candidates.append(content.strip())
    if candidates:
        return candidates[-1]

    answer_markers = [
        "final answer is",
        "answer is",
        "final answer:",
        "answer:",
    ]
    lowered = text.lower()
    for marker in answer_markers:
        marker_pos = lowered.rfind(marker)
        if marker_pos >= 0:
            tail = text[marker_pos + len(marker) :].strip()
            if tail:
                return tail.splitlines()[0].strip()

    nonempty_lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if nonempty_lines:
        return nonempty_lines[-1].strip()
    return None


def strip_outer_braces(text: str) -> str:
    candidate = text.strip()
    while candidate.startswith("{") and candidate.endswith("}"):
        depth = 0
        balanced_outer = True
        for idx, char in enumerate(candidate):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and idx != len(candidate) - 1:
                    balanced_outer = False
                    break
        if not balanced_outer:
            break
        candidate = candidate[1:-1].strip()
    return candidate


def normalize_math_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    normalized = answer.strip()
    if not normalized:
        return None

    boxed = extract_boxed_answer(normalized)
    if boxed is not None and boxed != normalized:
        normalized = boxed

    normalized = normalized.strip().strip("$").strip()
    normalized = normalized.rstrip(".")
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\:", "")
    normalized = normalized.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    normalized = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    normalized = strip_outer_braces(normalized)
    return normalized or None


def _read_balanced_group(text: str, open_pos: int, open_char: str = "{", close_char: str = "}") -> Tuple[Optional[str], int]:
    if open_pos < 0 or open_pos >= len(text) or text[open_pos] != open_char:
        return None, open_pos
    depth = 0
    for pos in range(open_pos, len(text)):
        char = text[pos]
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[open_pos + 1 : pos], pos + 1
    return None, open_pos


def _replace_latex_command_with_groups(text: str, command: str, replacement) -> str:
    while command in text:
        start = text.find(command)
        cursor = start + len(command)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        first, cursor_after_first = _read_balanced_group(text, cursor)
        if first is None:
            break
        cursor = cursor_after_first
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        second, cursor_after_second = _read_balanced_group(text, cursor)
        if second is None:
            break
        text = text[:start] + replacement(first, second) + text[cursor_after_second:]
    return text


def _replace_latex_sqrt(text: str) -> str:
    command = "\\sqrt"
    while command in text:
        start = text.find(command)
        cursor = start + len(command)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        degree = None
        if cursor < len(text) and text[cursor] == "[":
            degree, cursor = _read_balanced_group(text, cursor, "[", "]")
            if degree is None:
                break
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
        radicand, cursor_after = _read_balanced_group(text, cursor)
        if radicand is None:
            break
        if degree:
            replacement = f"(({_latex_to_sympy_text(radicand)})**(1/({_latex_to_sympy_text(degree)})))"
        else:
            replacement = f"sqrt({_latex_to_sympy_text(radicand)})"
        text = text[:start] + replacement + text[cursor_after:]
    return text


def _latex_to_sympy_text(answer: str) -> str:
    text = answer.strip().strip("$").strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\:", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)
    text = _replace_latex_command_with_groups(
        text,
        "\\frac",
        lambda numerator, denominator: f"(({_latex_to_sympy_text(numerator)})/({_latex_to_sympy_text(denominator)}))",
    )
    text = _replace_latex_sqrt(text)
    replacements = {
        "\\cdot": "*",
        "\\times": "*",
        "\\div": "/",
        "\\pi": "pi",
        "\\infty": "oo",
        "\\leq": "<=",
        "\\geq": ">=",
        "\\neq": "!=",
        "\\sin": "sin",
        "\\cos": "cos",
        "\\tan": "tan",
        "\\log": "log",
        "\\ln": "log",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("^", "**")
    text = text.replace("{", "(").replace("}", ")")
    text = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", text)
    text = re.sub(r"(?<=[A-Za-z)])(?=\d)", "*", text)
    text = re.sub(r"(?<=\))(?=[A-Za-z0-9(])", "*", text)
    text = text.replace(" ", "")
    return text


def _strip_answer_assignment(answer: str) -> str:
    text = answer.strip()
    if "=" not in text:
        return text
    left, right = text.split("=", 1)
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", left.strip()):
        return right.strip()
    return text


def _split_top_level_items(text: str) -> Optional[List[str]]:
    text = strip_outer_braces(text.strip())
    if len(text) >= 2 and text[0] in "([{" and text[-1] in ")]}":
        text = text[1:-1]
    items = []
    depth = 0
    start = 0
    for idx, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            items.append(text[start:idx].strip())
            start = idx + 1
    if not items:
        return None
    items.append(text[start:].strip())
    return items if all(item for item in items) else None


def _try_parse_sympy(answer: str):
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
    except Exception:
        return None

    answer = _strip_answer_assignment(answer)
    parse_candidates = [answer, _latex_to_sympy_text(answer)]

    try:
        from sympy.parsing.latex import parse_latex

        parse_candidates.insert(0, answer)
        for candidate in parse_candidates[:2]:
            try:
                return parse_latex(candidate)
            except Exception:
                pass
    except Exception:
        pass

    transformations = standard_transformations + (implicit_multiplication_application,)
    local_dict = {
        "pi": sp.pi,
        "e": sp.E,
        "oo": sp.oo,
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "log": sp.log,
    }
    for candidate in parse_candidates:
        try:
            return parse_expr(candidate, local_dict=local_dict, transformations=transformations, evaluate=True)
        except Exception:
            continue
    return None


def math_symbolic_match(prediction: Optional[str], reference: Optional[str]) -> bool:
    pred_norm = normalize_math_answer(prediction)
    ref_norm = normalize_math_answer(reference)
    if pred_norm is None or ref_norm is None:
        return False
    if pred_norm == ref_norm:
        return True
    if max(len(pred_norm), len(ref_norm)) > 256:
        return False

    pred_items = _split_top_level_items(pred_norm)
    ref_items = _split_top_level_items(ref_norm)
    if pred_items is not None or ref_items is not None:
        pred_wrapped = len(pred_norm) >= 2 and pred_norm[0] in "([{" and pred_norm[-1] in ")]}"
        ref_wrapped = len(ref_norm) >= 2 and ref_norm[0] in "([{" and ref_norm[-1] in ")]}"
        if pred_wrapped != ref_wrapped:
            return False
        if pred_wrapped and ref_wrapped and (pred_norm[0], pred_norm[-1]) != (ref_norm[0], ref_norm[-1]):
            return False
        if pred_items is None or ref_items is None or len(pred_items) != len(ref_items):
            return False
        return all(math_symbolic_match(pred_item, ref_item) for pred_item, ref_item in zip(pred_items, ref_items))

    pred_expr = _try_parse_sympy(pred_norm)
    ref_expr = _try_parse_sympy(ref_norm)
    if pred_expr is None or ref_expr is None:
        return False

    try:
        import sympy as sp

        if bool(sp.simplify(pred_expr - ref_expr) == 0):
            return True
        pred_numeric = sp.N(pred_expr, 30)
        ref_numeric = sp.N(ref_expr, 30)
        return bool(abs(pred_numeric - ref_numeric) < sp.Float("1e-12"))
    except Exception:
        return False


def math_exact_match(prediction: Optional[str], reference: Optional[str]) -> bool:
    return math_symbolic_match(prediction, reference)


def math_match_type(prediction: Optional[str], reference: Optional[str]) -> str:
    pred_norm = normalize_math_answer(prediction)
    ref_norm = normalize_math_answer(reference)
    if pred_norm is not None and pred_norm == ref_norm:
        return "exact"
    if math_symbolic_match(prediction, reference):
        return "symbolic"
    return "incorrect"



def select_eval_examples(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    dataset_cfg = config["dataset"]
    raw = _load_raw_dataset(dataset_cfg["name"], dataset_cfg.get("subset"))
    split_name = dataset_cfg.get("eval_split") or "test"
    dataset = raw[split_name]
    max_eval_samples = dataset_cfg.get("max_eval_samples")
    if max_eval_samples:
        dataset = dataset.select(range(min(int(max_eval_samples), len(dataset))))
    return [dict(example) for example in dataset]


def render_prompt(config: Dict[str, Any], tokenizer, example: Dict[str, Any]) -> str:
    dataset_cfg = config["dataset"]
    full_messages = _build_messages(
        example,
        dataset_name=dataset_cfg["name"],
        system_prompt=dataset_cfg["system_prompt"],
        dataset_format=dataset_cfg.get("format", "auto"),
        messages_field=dataset_cfg.get("messages_field", "messages"),
        prepend_system_prompt=dataset_cfg.get("prepend_system_prompt", False),
    )
    prompt_messages = [message for message in full_messages if message["role"] != "assistant"]
    return _render_messages(tokenizer, prompt_messages, add_generation_prompt=True)


def main() -> None:
    args = parse_args()
    args.cache_policy = normalize_cache_policy_name(args.cache_policy)
    extra_cache_policy_names = [
        policy for policy in parse_cache_policy_list(args.extra_cache_policies) if policy != args.cache_policy
    ]
    config = load_config(args.config)
    if args.eval_split is not None:
        config["dataset"]["eval_split"] = args.eval_split
    if args.max_eval_samples is not None:
        config["dataset"]["max_eval_samples"] = args.max_eval_samples

    output_dir = Path(args.output_dir)
    predictions_output = Path(args.predictions_output) if args.predictions_output else output_dir / "predictions.jsonl"
    metrics_output = Path(args.metrics_output) if args.metrics_output else output_dir / "metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_checkpoint_load and args.model_path is None:
        raise ValueError("--model-path is required unless --skip-checkpoint-load is set.")
    checkpoint_path = None if args.skip_checkpoint_load else resolve_checkpoint_path(Path(args.model_path))
    tokenizer = build_tokenizer(config)
    model = build_model(config)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = not args.no_generation_cache
    disable_training_router_outputs_for_generation(model)

    trainer = MoESFTTrainer(
        model=model,
        args=build_eval_args(config, output_dir),
        data_collator=SupervisedDataCollator(pad_token_id=tokenizer.pad_token_id),
    )
    if checkpoint_path is not None:
        load_trained_weights(trainer, checkpoint_path)
    model = trainer.model

    examples = select_eval_examples(config)
    rank, world_size = distributed_info()
    future_router_summary = validate_future_router_loaded(model, config, rank)
    generation_device = resolve_generation_device()
    model, model_input_device = move_model_for_generation(model, generation_device, rank)
    disable_training_router_outputs_for_generation(model)
    trainer.model = model

    expert_usage_output = Path(args.expert_usage_output) if args.expert_usage_output else output_dir / "expert_usage.pt"
    num_moe_layers, num_experts = infer_moe_shape(model)
    usage_counts = torch.zeros((num_moe_layers, num_experts), device=generation_device, dtype=torch.float64)
    token_counts = torch.zeros((num_moe_layers,), device=generation_device, dtype=torch.float64)
    routed_slot_counts = torch.zeros((num_moe_layers,), device=generation_device, dtype=torch.float64)
    cache_size = resolve_cache_size(config)
    expert_size_bytes = infer_routed_expert_size_bytes(model, override_mb=args.expert_size_mb)
    promoe_predictor = load_promoe_predictor(args.promoe_predictor_path)
    finemoe_store = load_finemoe_store(args.finemoe_store_path)
    all_requested_cache_policies = [args.cache_policy] + extra_cache_policy_names
    if "promoe" in all_requested_cache_policies and promoe_predictor is None:
        raise ValueError("--cache-policy promoe requires --promoe-predictor-path.")
    if "finemoe" in all_requested_cache_policies and finemoe_store is None:
        raise ValueError("--cache-policy finemoe requires --finemoe-store-path.")
    cache_policy_config = build_cache_policy_config(
        args=args,
        config=config,
        policy=args.cache_policy,
        promoe_predictor=promoe_predictor,
        finemoe_store=finemoe_store,
    )
    extra_cache_policy_configs = {
        policy: build_cache_policy_config(
            args=args,
            config=config,
            policy=policy,
            promoe_predictor=promoe_predictor,
            finemoe_store=finemoe_store,
        )
        for policy in extra_cache_policy_names
    }
    validate_future_only_cache_policy_binding(cache_policy_config)
    for extra_policy_config in extra_cache_policy_configs.values():
        validate_future_only_cache_policy_binding(extra_policy_config)
    generation_metric_sums = torch.zeros(24, device=generation_device, dtype=torch.float64)
    cache_policy_metric_sums = {
        policy: torch.zeros(24, device=generation_device, dtype=torch.float64)
        for policy in all_requested_cache_policies
    }

    local_examples = [(idx, example) for idx, example in enumerate(examples) if idx % world_size == rank]
    local_predictions: List[Dict[str, Any]] = []
    local_promoe_traces: List[Dict[str, Any]] = []
    local_predictions_output = output_dir / f"predictions.rank{rank}.jsonl.tmp"
    local_predictions_output.parent.mkdir(parents=True, exist_ok=True)
    if local_predictions_output.exists():
        local_predictions_output.unlink()
    local_metrics_output = output_dir / f"metrics.rank{rank}.pt"
    if local_metrics_output.exists():
        local_metrics_output.unlink()
    for stale_tmp in output_dir.glob(f"{local_metrics_output.name}.tmp.*"):
        stale_tmp.unlink()
    print(
        f"[rank{rank}] Starting MATH generation: local_samples={len(local_examples)}, "
        f"world_size={world_size}, max_new_tokens={args.max_new_tokens}, "
        f"device={model_input_device}.",
        flush=True,
    )

    do_sample = args.temperature > 0
    start_time = time.time()
    for local_pos, (idx, example) in enumerate(local_examples, start=1):
        if args.skip_reference_loss:
            reference_loss_sum = torch.zeros((), device=generation_device, dtype=torch.float64)
            reference_token_count = torch.zeros((), device=generation_device, dtype=torch.float64)
            sample_reference_moe_loss = 0.0
        else:
            reference_loss_sum, reference_token_count, sample_reference_moe_loss = reference_moe_loss_sum(
                model=model,
                config=config,
                tokenizer=tokenizer,
                example=example,
                device=model_input_device,
            )
        generation_metric_sums[3] += reference_loss_sum.to(device=generation_device)
        generation_metric_sums[4] += reference_token_count.to(device=generation_device)

        reset_recorded_expert_stats(model)
        prompt = render_prompt(config, tokenizer, example)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(model_input_device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=not args.no_generation_cache,
                output_router_logits=False,
                synced_gpus=False,
                return_dict_in_generate=args.compute_generation_nll,
                output_scores=args.compute_generation_nll,
            )
        accumulate_expert_usage(model, usage_counts, token_counts, routed_slot_counts)
        if args.promoe_trace_output is not None:
            local_promoe_traces.append(build_promoe_trace_sample(get_recorded_layer_stats(model), idx))
        sequences = generated.sequences if args.compute_generation_nll else generated
        new_tokens = sequences[0, encoded["input_ids"].shape[-1] :]
        if args.compute_generation_nll:
            generation_nll_sum, generation_token_count = generation_nll_from_scores(generated.scores, new_tokens)
        else:
            generation_nll_sum = torch.zeros((), device=generation_device, dtype=torch.float64)
            generation_token_count = torch.zeros((), device=generation_device, dtype=torch.float64)
        prompt_token_count = int(encoded["input_ids"].shape[-1])
        hard_metrics, decode_hard_metrics = evaluate_hard_cache_metrics_pair(
            model,
            cache_size=cache_size,
            device=generation_device,
            cache_policy_config=cache_policy_config,
            decode_count_from_token_idx=prompt_token_count,
        )
        accumulate_cache_metric_pair(
            cache_policy_metric_sums[cache_policy_config.policy],
            hard_metrics,
            decode_hard_metrics,
        )
        for extra_policy, extra_cache_policy_config in extra_cache_policy_configs.items():
            extra_hard_metrics, extra_decode_hard_metrics = evaluate_hard_cache_metrics_pair(
                model,
                cache_size=cache_size,
                device=generation_device,
                cache_policy_config=extra_cache_policy_config,
                decode_count_from_token_idx=prompt_token_count,
            )
            accumulate_cache_metric_pair(
                cache_policy_metric_sums[extra_policy],
                extra_hard_metrics,
                extra_decode_hard_metrics,
            )
        generation_metric_sums[22] += generation_nll_sum.to(device=generation_device)
        generation_metric_sums[23] += generation_token_count.to(device=generation_device)
        generation_metric_sums[0] += hard_metrics["hard_hit_rate_sum"]
        generation_metric_sums[1] += hard_metrics["hard_overlap_sum"]
        generation_metric_sums[2] += hard_metrics["hard_step_count"]
        generation_metric_sums[5] += torch.tensor(float(new_tokens.numel()), device=generation_device, dtype=torch.float64)
        generation_metric_sums[6] += hard_metrics["hard_access_count"]
        generation_metric_sums[7] += hard_metrics["hard_miss_count"]
        generation_metric_sums[8] += decode_hard_metrics["hard_hit_rate_sum"]
        generation_metric_sums[9] += decode_hard_metrics["hard_overlap_sum"]
        generation_metric_sums[10] += decode_hard_metrics["hard_step_count"]
        generation_metric_sums[11] += decode_hard_metrics["hard_access_count"]
        generation_metric_sums[12] += decode_hard_metrics["hard_miss_count"]
        generation_metric_sums[13] += hard_metrics.get("hard_demand_miss_count", hard_metrics["hard_miss_count"])
        generation_metric_sums[14] += hard_metrics.get(
            "hard_prefetch_load_count",
            torch.zeros((), device=generation_device, dtype=torch.float64),
        )
        generation_metric_sums[15] += decode_hard_metrics.get(
            "hard_demand_miss_count",
            decode_hard_metrics["hard_miss_count"],
        )
        generation_metric_sums[16] += decode_hard_metrics.get(
            "hard_prefetch_load_count",
            torch.zeros((), device=generation_device, dtype=torch.float64),
        )
        temporal_option_summary = temporal_option_generation_summary(model, generation_device)
        generation_metric_sums[17:22] += temporal_option_summary

        sample_hard_steps = float(hard_metrics["hard_step_count"].cpu().item())
        sample_hard_summary = summarize_hard_cache_metrics(hard_metrics)
        sample_decode_hard_summary = summarize_hard_cache_metrics(decode_hard_metrics)
        temporal_option_count = float(temporal_option_summary[4].detach().cpu().item())
        sample_hard_hit_rate = sample_hard_summary["hit_rate"]
        sample_hard_overlap = sample_hard_summary["overlap_count"]
        sample_generation_tokens = float(generation_token_count.cpu().item())
        if not args.compute_generation_nll:
            sample_generation_tokens = float(new_tokens.numel())
        sample_generation_nll = (
            float((generation_nll_sum / generation_token_count).cpu().item())
            if args.compute_generation_nll and sample_generation_tokens > 0
            else 0.0
        )
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred_answer = extract_boxed_answer(generated_text)
        ref_answer = extract_boxed_answer(str(example["solution"]))
        normalized_ref_answer = normalize_math_answer(ref_answer)
        normalized_pred_answer = normalize_math_answer(pred_answer)
        exact_match = (
            normalized_ref_answer is not None
            and normalized_pred_answer is not None
            and normalized_ref_answer == normalized_pred_answer
        )
        prediction = {
            "index": idx,
            "problem": example["problem"],
            "level": example.get("level"),
            "type": example.get("type"),
            "reference_answer": ref_answer,
            "predicted_answer": pred_answer,
            "normalized_reference_answer": normalized_ref_answer,
            "normalized_predicted_answer": normalized_pred_answer,
            "correct": exact_match,
            "match_type": "exact" if exact_match else "pending_symbolic",
            "reference_moe_loss": sample_reference_moe_loss,
            "reference_loss_token_count": int(reference_token_count.cpu().item()),
            "generated_token_count": int(sample_generation_tokens),
            "generation_hard_cache_hit_rate": sample_hard_hit_rate,
            "generation_hard_cache_miss_rate": 1.0 - sample_hard_hit_rate if sample_hard_steps > 0 else 0.0,
            "generation_hard_cache_overlap_count": sample_hard_overlap,
            "generation_hard_cache_counted_steps": int(sample_hard_steps),
            "decode_hard_cache_hit_rate": sample_decode_hard_summary["hit_rate"],
            "decode_hard_cache_miss_rate": sample_decode_hard_summary["miss_rate"],
            "decode_hard_cache_overlap_count": sample_decode_hard_summary["overlap_count"],
            "decode_hard_cache_counted_steps": int(sample_decode_hard_summary["counted_steps"]),
            "decode_expert_miss_count": sample_decode_hard_summary["miss_count"],
            "decode_expert_demand_miss_count": float(
                decode_hard_metrics.get("hard_demand_miss_count", decode_hard_metrics["hard_miss_count"])
                .detach()
                .cpu()
                .item()
            ),
            "decode_expert_prefetch_load_count": float(
                decode_hard_metrics.get(
                    "hard_prefetch_load_count",
                    torch.zeros((), device=generation_device, dtype=torch.float64),
                )
                .detach()
                .cpu()
                .item()
            ),
            "generated_text": generated_text,
        }
        if temporal_option_count > 0.0:
            prediction["temporal_option_switch_rate"] = float(
                (temporal_option_summary[0] / temporal_option_summary[4].clamp_min(1.0)).detach().cpu().item()
            )
            prediction["temporal_option_teacher_coverage"] = float(
                (temporal_option_summary[2] / temporal_option_summary[4].clamp_min(1.0)).detach().cpu().item()
            )
            prediction["temporal_option_router_overlap"] = float(
                (temporal_option_summary[3] / temporal_option_summary[4].clamp_min(1.0)).detach().cpu().item()
            )
        if args.compute_generation_nll:
            prediction["generation_nll"] = sample_generation_nll
        local_predictions.append(prediction)
        with local_predictions_output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            handle.flush()
        if args.progress_every > 0 and (
            local_pos == 1 or local_pos == len(local_examples) or local_pos % args.progress_every == 0
        ):
            elapsed = max(time.time() - start_time, 1e-6)
            samples_per_minute = local_pos / elapsed * 60.0
            print(
                f"[rank{rank}] progress {local_pos}/{len(local_examples)} "
                f"(global_index={idx}, {samples_per_minute:.2f} samples/min).",
                flush=True,
            )

    save_promoe_trace_output(args.promoe_trace_output, local_promoe_traces, rank=rank, world_size=world_size)
    (
        usage_counts,
        token_counts,
        routed_slot_counts,
        generation_metric_sums,
        cache_policy_metric_sums,
    ) = gather_metric_tensors_from_rank_files(
        output_dir=output_dir,
        world_size=world_size,
        rank=rank,
        usage_counts=usage_counts,
        token_counts=token_counts,
        routed_slot_counts=routed_slot_counts,
        generation_metric_sums=generation_metric_sums,
        extra_cache_policy_metric_sums=cache_policy_metric_sums,
        return_extra_cache_policy_metric_sums=True,
    )
    predictions = gather_predictions_from_rank_files(
        local_predictions=local_predictions,
        output_dir=output_dir,
        world_size=world_size,
        rank=rank,
    )

    if rank == 0:
        symbolic_timeout_count = 0
        if not args.no_symbolic_match:
            print("Running symbolic equivalence matching on rank0 after distributed metric reduction.", flush=True)
            for item in predictions:
                if item.get("match_type") == "exact":
                    item["correct"] = True
                    continue
                try:
                    match_type = call_with_timeout(
                        lambda: math_match_type(item.get("predicted_answer"), item.get("reference_answer")),
                        timeout_seconds=args.symbolic_timeout,
                    )
                except TimeoutError:
                    match_type = "timeout"
                    symbolic_timeout_count += 1
                item["match_type"] = match_type
                item["correct"] = match_type in {"exact", "symbolic"}
        else:
            for item in predictions:
                if item.get("match_type") == "pending_symbolic":
                    item["match_type"] = "incorrect"
                    item["correct"] = False

        correct = sum(1 for item in predictions if item["correct"])
        exact_correct = sum(1 for item in predictions if item.get("match_type") == "exact")
        symbolic_correct = sum(1 for item in predictions if item.get("match_type") == "symbolic")
        total = len(predictions)
        generation_hard_step_count = generation_metric_sums[2].clamp_min(1.0)
        decode_hard_step_count = generation_metric_sums[10].clamp_min(1.0)
        reference_token_count = generation_metric_sums[4].clamp_min(1.0)
        generated_token_count = generation_metric_sums[5].clamp_min(1.0)
        generation_nll_token_count = generation_metric_sums[23].clamp_min(1.0)
        generation_nll = generation_metric_sums[22] / generation_nll_token_count
        metric_mode = hard_cache_metric_mode(model, cache_policy_config)
        generation_hard_hit_rate = generation_metric_sums[0] / generation_hard_step_count
        decode_hard_hit_rate = generation_metric_sums[8] / decode_hard_step_count
        reference_moe_loss = generation_metric_sums[3] / reference_token_count
        full_hard_summary = {
            "hit_rate": float(generation_hard_hit_rate.cpu().item()),
            "miss_rate": float((1.0 - generation_hard_hit_rate).cpu().item()),
            "overlap_count": float((generation_metric_sums[1] / generation_hard_step_count).cpu().item()),
            "counted_steps": float(generation_metric_sums[2].cpu().item()),
            "access_count": float(generation_metric_sums[6].cpu().item()),
            "miss_count": float(generation_metric_sums[7].cpu().item()),
            "prefetch_load_count": float(generation_metric_sums[14].cpu().item()),
        }
        full_hard_summary["load_adjusted_access_count"] = (
            full_hard_summary["access_count"] + full_hard_summary["prefetch_load_count"]
        )
        full_hard_summary["load_adjusted_miss_rate"] = (
            full_hard_summary["miss_count"] / full_hard_summary["load_adjusted_access_count"]
            if full_hard_summary["load_adjusted_access_count"] > 0.0
            else 0.0
        )
        full_hard_summary["load_adjusted_hit_rate"] = 1.0 - full_hard_summary["load_adjusted_miss_rate"]
        decode_hard_summary = {
            "hit_rate": float(decode_hard_hit_rate.cpu().item()),
            "miss_rate": float((1.0 - decode_hard_hit_rate).cpu().item()),
            "overlap_count": float((generation_metric_sums[9] / decode_hard_step_count).cpu().item()),
            "counted_steps": float(generation_metric_sums[10].cpu().item()),
            "access_count": float(generation_metric_sums[11].cpu().item()),
            "miss_count": float(generation_metric_sums[12].cpu().item()),
            "prefetch_load_count": float(generation_metric_sums[16].cpu().item()),
        }
        decode_hard_summary["load_adjusted_access_count"] = (
            decode_hard_summary["access_count"] + decode_hard_summary["prefetch_load_count"]
        )
        decode_hard_summary["load_adjusted_miss_rate"] = (
            decode_hard_summary["miss_count"] / decode_hard_summary["load_adjusted_access_count"]
            if decode_hard_summary["load_adjusted_access_count"] > 0.0
            else 0.0
        )
        decode_hard_summary["load_adjusted_hit_rate"] = 1.0 - decode_hard_summary["load_adjusted_miss_rate"]
        full_load_summary = expert_load_summary(
            full_hard_summary,
            float(generated_token_count.cpu().item()),
            expert_size_bytes,
            bandwidth_gbps=args.expert_load_bandwidth_gbps,
        )
        decode_output_token_count = decode_hard_summary["counted_steps"] / float(max(num_moe_layers, 1))
        decode_load_summary = expert_load_summary(
            decode_hard_summary,
            decode_output_token_count,
            expert_size_bytes,
            bandwidth_gbps=args.expert_load_bandwidth_gbps,
        )
        internal_predictor_parameter_count = int(
            future_router_summary.get("future_router_parameter_count", 0)
        ) + int(future_router_summary.get("temporal_option_parameter_count", 0))
        predictor_summary = predictor_overhead_summary(
            internal_predictor_parameter_count,
            effective_tflops=args.predictor_effective_tflops,
            dtype_bytes=_dtype_num_bytes(getattr(getattr(_unwrap_cache_model(model), "config", None), "torch_dtype", None)),
            external_predictor=promoe_predictor,
            num_moe_layers=num_moe_layers,
            prefetch_lookahead=cache_policy_config.prefetch_lookahead,
        )
        finemoe_summary = finemoe_store_summary(finemoe_store)
        cache_policy_configs_by_policy = {
            cache_policy_config.policy: cache_policy_config,
            **extra_cache_policy_configs,
        }
        cache_policy_metrics = {
            policy: summarize_cache_policy_metric_sums(
                policy=policy,
                metric_sums=policy_metric_sums,
                metric_mode=hard_cache_metric_mode(model, cache_policy_configs_by_policy[policy]),
                cache_policy_config=cache_policy_configs_by_policy[policy],
                cache_size=cache_size,
                total=total,
                correct=correct,
                generated_token_count=float(generated_token_count.cpu().item()),
                num_moe_layers=num_moe_layers,
                expert_size_bytes=expert_size_bytes,
                expert_load_bandwidth_gbps=args.expert_load_bandwidth_gbps,
            )
            for policy, policy_metric_sums in cache_policy_metric_sums.items()
        }
        metrics = {
            "model_path": str(Path(args.model_path).resolve()) if args.model_path is not None else config["model"]["name"],
            "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path is not None else None,
            "total": total,
            "correct": correct,
            "exact_correct": exact_correct,
            "symbolic_correct": symbolic_correct,
            "symbolic_timeout_count": symbolic_timeout_count,
            "accuracy": correct / total if total else 0.0,
            "exact_accuracy": exact_correct / total if total else 0.0,
            "symbolic_accuracy": symbolic_correct / total if total else 0.0,
            "moe_loss": float(reference_moe_loss.cpu().item()),
            "reference_moe_loss": float(reference_moe_loss.cpu().item()),
            "reference_loss_token_count": int(generation_metric_sums[4].cpu().item()),
            "generated_token_count": int(generation_metric_sums[5].cpu().item()),
            "hard_cache_hit_rate": float(generation_hard_hit_rate.cpu().item()),
            "hard_cache_miss_rate": float((1.0 - generation_hard_hit_rate).cpu().item()),
            "load_adjusted_hard_cache_hit_rate": full_hard_summary["load_adjusted_hit_rate"],
            "load_adjusted_hard_cache_miss_rate": full_hard_summary["load_adjusted_miss_rate"],
            "hard_cache_overlap_count": float((generation_metric_sums[1] / generation_hard_step_count).cpu().item()),
            "hard_cache_access_count": int(generation_metric_sums[6].cpu().item()),
            "hard_cache_miss_count": float(generation_metric_sums[7].cpu().item()),
            "generation_hard_cache_hit_rate": float(generation_hard_hit_rate.cpu().item()),
            "generation_hard_cache_miss_rate": float((1.0 - generation_hard_hit_rate).cpu().item()),
            "generation_load_adjusted_hard_cache_hit_rate": full_hard_summary["load_adjusted_hit_rate"],
            "generation_load_adjusted_hard_cache_miss_rate": full_hard_summary["load_adjusted_miss_rate"],
            "generation_hard_cache_overlap_count": float(
                (generation_metric_sums[1] / generation_hard_step_count).cpu().item()
            ),
            "generation_hard_cache_counted_steps": int(generation_metric_sums[2].cpu().item()),
            "generation_hard_cache_access_count": int(generation_metric_sums[6].cpu().item()),
            "generation_hard_cache_miss_count": float(generation_metric_sums[7].cpu().item()),
            "generation_hard_cache_demand_miss_count": float(generation_metric_sums[13].cpu().item()),
            "generation_hard_cache_prefetch_load_count": float(generation_metric_sums[14].cpu().item()),
            "decode_hard_cache_hit_rate": decode_hard_summary["hit_rate"],
            "decode_hard_cache_miss_rate": decode_hard_summary["miss_rate"],
            "decode_load_adjusted_hard_cache_hit_rate": decode_hard_summary["load_adjusted_hit_rate"],
            "decode_load_adjusted_hard_cache_miss_rate": decode_hard_summary["load_adjusted_miss_rate"],
            "decode_hard_cache_overlap_count": decode_hard_summary["overlap_count"],
            "decode_hard_cache_counted_steps": int(decode_hard_summary["counted_steps"]),
            "decode_output_token_count": int(round(decode_output_token_count)),
            "decode_hard_cache_access_count": int(decode_hard_summary["access_count"]),
            "decode_hard_cache_miss_count": decode_hard_summary["miss_count"],
            "decode_hard_cache_demand_miss_count": float(generation_metric_sums[15].cpu().item()),
            "decode_hard_cache_prefetch_load_count": float(generation_metric_sums[16].cpu().item()),
            "temporal_option_switch_rate": (
                float((generation_metric_sums[17] / generation_metric_sums[21].clamp_min(1.0)).cpu().item())
                if float(generation_metric_sums[21].cpu().item()) > 0.0
                else 0.0
            ),
            "temporal_option_target_switch_rate": (
                float((generation_metric_sums[18] / generation_metric_sums[21].clamp_min(1.0)).cpu().item())
                if float(generation_metric_sums[21].cpu().item()) > 0.0
                else 0.0
            ),
            "temporal_option_teacher_coverage": (
                float((generation_metric_sums[19] / generation_metric_sums[21].clamp_min(1.0)).cpu().item())
                if float(generation_metric_sums[21].cpu().item()) > 0.0
                else 0.0
            ),
            "temporal_option_router_overlap": (
                float((generation_metric_sums[20] / generation_metric_sums[21].clamp_min(1.0)).cpu().item())
                if float(generation_metric_sums[21].cpu().item()) > 0.0
                else 0.0
            ),
            "temporal_option_counted_steps": float(generation_metric_sums[21].cpu().item()),
            "expert_size_MB": decode_load_summary["expert_size_MB"],
            "load_MB_per_generated_token": decode_load_summary["load_MB_per_generated_token"],
            "decode_load_MB_per_generated_token": decode_load_summary["load_MB_per_generated_token"],
            "decode_load_MB_per_decode_token": decode_load_summary["load_MB_per_generated_token"],
            "decode_expert_load_MB": decode_load_summary["expert_load_MB"],
            "decode_expert_misses_per_generated_token": decode_load_summary["expert_misses_per_generated_token"],
            "decode_expert_misses_per_decode_token": decode_load_summary["expert_misses_per_generated_token"],
            "full_request_load_MB_per_generated_token": full_load_summary["load_MB_per_generated_token"],
            "full_request_expert_load_MB": full_load_summary["expert_load_MB"],
            "full_request_expert_misses_per_generated_token": full_load_summary["expert_misses_per_generated_token"],
            "cache_size": cache_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "symbolic_match_enabled": not args.no_symbolic_match,
            "reference_loss_skipped": bool(args.skip_reference_loss),
            "expert_usage_output": str(expert_usage_output),
            "cache_policy": cache_policy_config.policy,
            "hard_cache_metric_mode": metric_mode,
            "cache_policy_recency_weight": cache_policy_config.recency_weight,
            "cache_policy_frequency_weight": cache_policy_config.frequency_weight,
            "cache_policy_recency_decay": cache_policy_config.recency_decay,
            "cache_policy_lrfu_lambda": cache_policy_config.lrfu_lambda,
            "cache_policy_prefetch_budget": cache_policy_config.prefetch_budget,
            "future_cache_priority_mode": cache_policy_config.future_cache_priority_mode,
            "cache_policy_effective_prefetch_budget": effective_cache_policy_prefetch_budget(
                cache_policy_config,
                cache_size=cache_size,
            ),
            "cache_policy_prefetch_lookahead": cache_policy_config.prefetch_lookahead,
            "spatio_temporal_refine_mode": cache_policy_config.spatio_temporal_refine_mode,
            "spatio_temporal_refine_budget": cache_policy_config.spatio_temporal_refine_budget,
            "promoe_predictor_path": args.promoe_predictor_path,
            "promoe_trace_output": args.promoe_trace_output,
            "finemoe_store_path": args.finemoe_store_path,
            "finemoe_top_k": cache_policy_config.finemoe_top_k,
            "finemoe_candidate_pool": cache_policy_config.finemoe_candidate_pool,
            "finemoe_semantic_weight": cache_policy_config.finemoe_semantic_weight,
            "finemoe_trajectory_weight": cache_policy_config.finemoe_trajectory_weight,
            "finemoe_prefetch_threshold": cache_policy_config.finemoe_prefetch_threshold,
            "finemoe_min_prefetch_threshold": cache_policy_config.finemoe_min_prefetch_threshold,
            "finemoe_max_prefetch_threshold": cache_policy_config.finemoe_max_prefetch_threshold,
            "finemoe_eviction_probability_weight": cache_policy_config.finemoe_eviction_probability_weight,
            "extra_cache_policies": extra_cache_policy_names,
            "cache_policy_metrics": cache_policy_metrics,
            **future_router_summary,
            **predictor_summary,
            **finemoe_summary,
        }
        if args.expert_load_bandwidth_gbps > 0.0:
            metrics.update(
                {
                    "expert_load_bandwidth_GBps": args.expert_load_bandwidth_gbps,
                    "estimated_load_TPOT_ms_per_generated_token": decode_load_summary[
                        "estimated_load_TPOT_ms_per_generated_token"
                    ],
                    "decode_estimated_load_TPOT_ms_per_generated_token": decode_load_summary[
                        "estimated_load_TPOT_ms_per_generated_token"
                    ],
                    "full_request_estimated_load_TPOT_ms_per_generated_token": full_load_summary[
                        "estimated_load_TPOT_ms_per_generated_token"
                    ],
                }
            )
        if args.compute_generation_nll:
            metrics.update(
                {
                    "generation_nll": float(generation_nll.cpu().item()),
                    "generation_ppl": float(torch.exp(generation_nll).cpu().item()),
                    "generation_nll_token_count": int(generation_metric_sums[23].cpu().item()),
                }
            )
        predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with predictions_output.open("w", encoding="utf-8") as handle:
            for item in predictions:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        metrics = emphasize_paper_metrics(metrics)
        metrics_output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        save_expert_usage(
            expert_usage_output,
            usage_counts,
            token_counts,
            routed_slot_counts,
            sample_count=total,
            model_name=config["model"]["name"],
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
