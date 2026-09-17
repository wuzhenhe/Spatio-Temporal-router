from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn


PROMOE_PREDICTOR_FORMAT = "promoe_moe_layer_logits_mlp_v1"


class ProMoELayerLogitsMLP(nn.Module):
    """ProMoE learned predictor: a two-layer MLP from layer input to expert priority."""

    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, layer_input: torch.Tensor) -> torch.Tensor:
        return self.net(layer_input)


def infer_promoe_hidden_dim(input_dim: int, num_experts: int, target_params: int = 2_000_000) -> int:
    """Choose the hidden width that makes one predictor close to the paper's 2M parameters."""

    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive.")
    target_params = max(int(target_params), num_experts + 1)
    # Params = input_dim*hidden + hidden + hidden*num_experts + num_experts.
    return max(1, int(round((target_params - num_experts) / float(input_dim + num_experts + 1))))


def promoe_mlp_parameter_count(input_dim: int, num_experts: int, hidden_dim: int) -> int:
    return int(hidden_dim) * (int(input_dim) + int(num_experts) + 1) + int(num_experts)


def build_promoe_layer_mapping(
    num_layers: int,
    interval: int = 1,
    max_window: int = 3,
    replace_first_input_with_last_output: bool = True,
    limit_layer_0_window: int = -1,
) -> Dict[int, List[int]]:
    """Port of ProMoE's build_predict_layer_mapping.

    Keys 0..num_layers-1 are current-token source layer positions. Key num_layers
    is the previous token's last-layer output when first-layer input is replaced.
    Values are target layer positions.
    """

    num_layers = int(num_layers)
    interval = max(1, int(interval))
    max_window = max(0, int(max_window))
    if num_layers <= 0:
        return {}

    layers_to_predict = list(range(0, num_layers, interval))
    predict_layers = [0 for _ in range(num_layers + 1)]
    stop_l = 0
    for layer_pos in layers_to_predict:
        window = int(max_window)
        if layer_pos == 0 and int(limit_layer_0_window) != -1:
            window = int(limit_layer_0_window)
        predict_layers[layer_pos] = stop_l
        predict_layers[layer_pos + 1] = min((layer_pos % num_layers) + window, num_layers)
        stop_l = predict_layers[layer_pos + 1]

    for layer_pos in range(1, num_layers + 1):
        if predict_layers[layer_pos] < predict_layers[layer_pos - 1]:
            predict_layers[layer_pos] = predict_layers[layer_pos - 1]

    mapping = {
        layer_pos: list(range(predict_layers[layer_pos], predict_layers[layer_pos + 1]))
        for layer_pos in range(num_layers)
    }
    if replace_first_input_with_last_output:
        mapping[num_layers] = mapping.get(0, [])
        mapping[0] = []
    return mapping


def _parse_model_key(key: Any) -> Tuple[int, int]:
    if isinstance(key, tuple) and len(key) == 2:
        return int(key[0]), int(key[1])
    if isinstance(key, str):
        if "-" in key:
            src, dst = key.split("-", 1)
            return int(src), int(dst)
        if "," in key:
            src, dst = key.split(",", 1)
            return int(src), int(dst)
    raise ValueError(f"Unsupported ProMoE predictor key: {key!r}")


def make_promoe_model_key(source_layer_idx: int, target_layer_idx: int) -> str:
    return f"{int(source_layer_idx)}-{int(target_layer_idx)}"


def load_promoe_predictor_runtime(predictor: Dict[str, Any]) -> Dict[str, Any]:
    if predictor.get("_runtime_models") is not None:
        return predictor
    if predictor.get("format") != PROMOE_PREDICTOR_FORMAT:
        raise ValueError(f"Invalid ProMoE predictor format: {predictor.get('format')!r}")

    config = predictor.get("config", {})
    input_dim = int(config.get("input_dim", config.get("num_experts", 0)))
    num_experts = int(config.get("num_experts", 0))
    hidden_dim = int(config.get("hidden_dim", 0))
    if input_dim <= 0 or num_experts <= 0 or hidden_dim <= 0:
        raise ValueError("ProMoE predictor config must include positive input_dim, num_experts, and hidden_dim.")

    runtime_models: Dict[Tuple[int, int], ProMoELayerLogitsMLP] = {}
    for raw_key, state_dict in predictor.get("models", {}).items():
        source_layer_idx, target_layer_idx = _parse_model_key(raw_key)
        model = ProMoELayerLogitsMLP(input_dim=input_dim, num_experts=num_experts, hidden_dim=hidden_dim)
        model.load_state_dict(state_dict)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        runtime_models[(source_layer_idx, target_layer_idx)] = model.cpu()
    predictor["_runtime_models"] = runtime_models
    return predictor


def promoe_predict(
    predictor: Dict[str, Any],
    source_layer_idx: int,
    target_layer_idx: int,
    layer_input: torch.Tensor,
) -> Optional[torch.Tensor]:
    runtime = load_promoe_predictor_runtime(predictor).get("_runtime_models", {})
    model = runtime.get((int(source_layer_idx), int(target_layer_idx)))
    if model is None:
        return None

    config = predictor.get("config", {})
    input_dim = int(config.get("input_dim", config.get("num_experts", 0)))
    inputs = layer_input.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    if input_dim > 0 and inputs.numel() != input_dim:
        raise ValueError(f"Expected {input_dim} ProMoE input features, got {inputs.numel()}.")
    with torch.no_grad():
        scores = model(inputs.reshape(1, -1))[0]
    return scores.detach().to(device="cpu", dtype=torch.float32)


def promoe_predictor_parameter_count(predictor: Optional[Dict[str, Any]]) -> int:
    if not predictor:
        return 0
    metadata = predictor.get("metadata", {})
    if "parameter_count" in metadata:
        return int(metadata["parameter_count"])
    total = 0
    for state_dict in predictor.get("models", {}).values():
        total += sum(int(tensor.numel()) for tensor in state_dict.values())
    return total


def promoe_predictor_model_count(predictor: Optional[Dict[str, Any]]) -> int:
    if not predictor:
        return 0
    return len(predictor.get("models", {}))


def promoe_predictor_macs_per_model(predictor: Optional[Dict[str, Any]]) -> int:
    if not predictor:
        return 0
    config = predictor.get("config", {})
    input_dim = int(config.get("input_dim", config.get("num_experts", 0)))
    num_experts = int(config.get("num_experts", 0))
    hidden_dim = int(config.get("hidden_dim", 0))
    if input_dim <= 0 or num_experts <= 0 or hidden_dim <= 0:
        return 0
    return input_dim * hidden_dim + hidden_dim * num_experts
