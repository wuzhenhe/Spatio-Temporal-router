from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch

from finemoe_store import (
    FINEMOE_STORE_FORMAT,
    build_finemoe_store_from_trace_samples,
    finemoe_store_summary,
    load_trace_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the FineMoE Expert Map Store from generation traces. "
            "This is the offline/training stage for the FineMoE baseline."
        )
    )
    parser.add_argument(
        "--trace-input",
        action="append",
        required=True,
        help="Torch .pt trace from eval --promoe-trace-output. Can be passed multiple times.",
    )
    parser.add_argument("--output", required=True, help="Output .pt path for the FineMoE Expert Map Store.")
    parser.add_argument(
        "--max-entries",
        type=int,
        default=1000,
        help="Maximum token-level Expert Map Store entries. Use <=0 to keep all entries.",
    )
    parser.add_argument(
        "--max-tokens-per-sample",
        type=int,
        default=None,
        help="Optional cap on stored iterations/tokens per trace sample.",
    )
    parser.add_argument(
        "--semantic-layer-idx",
        type=int,
        default=None,
        help="MoE layer used for the semantic key. Defaults to the first layer with recorded inputs.",
    )
    parser.add_argument(
        "--dedup-similarity-threshold",
        type=float,
        default=0.0,
        help=(
            "Optional cosine-similarity threshold for skipping near-duplicate semantic keys. "
            "0 disables deduplication."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_paths = [Path(path) for path in args.trace_input]
    samples = load_trace_samples(trace_paths)
    if not samples:
        raise ValueError("No trace samples were found.")

    store = build_finemoe_store_from_trace_samples(
        samples,
        max_entries=int(args.max_entries),
        max_tokens_per_sample=args.max_tokens_per_sample,
        semantic_layer_idx=args.semantic_layer_idx,
        dedup_similarity_threshold=float(args.dedup_similarity_threshold),
    )
    if store.get("format") != FINEMOE_STORE_FORMAT or not store.get("entries"):
        raise ValueError("FineMoE store construction produced no entries.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(store, output)

    summary: Dict[str, Any] = finemoe_store_summary(store)
    print(
        "Saved FineMoE Expert Map Store: "
        f"path={output}, entries={summary['finemoe_store_entries']}, "
        f"layers={summary['finemoe_store_num_layers']}, "
        f"experts={summary['finemoe_store_num_experts']}, "
        f"storage_MB={summary['finemoe_store_storage_MB']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
