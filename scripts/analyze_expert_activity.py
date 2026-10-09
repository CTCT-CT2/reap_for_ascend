#!/usr/bin/env python3
"""Rank and summarize expert activity from a REAP observation file."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch


def _as_float(value: Any) -> float:
    value = float(value)
    return value if math.isfinite(value) else 0.0


def _tensor_value(value: Any) -> Any:
    """Unwrap REAP's OnlineStatsTracker while accepting plain tensors."""
    if isinstance(value, torch.Tensor):
        return value
    mean = getattr(value, "mean", None)
    return mean if isinstance(mean, torch.Tensor) else value


def _distribution_stats(values: list[float]) -> tuple[float, float]:
    """Return normalized entropy and Gini coefficient for nonnegative values."""
    total = sum(values)
    if total <= 0 or not values:
        return 0.0, 0.0
    probabilities = [value / total for value in values]
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    normalized_entropy = entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
    ordered = sorted(probabilities)
    gini = sum((2 * index - len(ordered) - 1) * value for index, value in enumerate(ordered, 1))
    gini /= len(ordered) * total
    return normalized_entropy, gini


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _layer_rows(layer: int, state: dict[str, Any]) -> list[dict[str, Any]]:
    count = _tensor_value(state["expert_frequency"]).double()
    total = float(state["total_tokens"])
    score = _tensor_value(state.get("reap", state.get("ean_mean", count))).double()
    weighted = state.get("weighted_expert_frequency_sum")
    if weighted is not None:
        weighted = _tensor_value(weighted).double()
    finite = torch.full_like(count, total)
    selection_rate = count / max(total, 1.0)
    share = count / count.sum().clamp_min(1.0)
    nan_count = torch.zeros_like(count, dtype=torch.long)

    rows = []
    for expert in range(count.numel()):
        rows.append(
            {
                "layer": layer,
                "expert": expert,
                "activity_score": _as_float(score[expert]),
                "selection_count": int(count[expert]),
                "selection_rate": _as_float(selection_rate[expert]),
                "assignment_share": _as_float(share[expert]),
                "weighted_router_sum": "" if weighted is None else _as_float(weighted[expert]),
                "finite_samples": int(finite[expert]),
                "nan_count": int(nan_count[expert]),
            }
        )
    return rows


def analyze(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifact = _load(Path(path))
    rows: list[dict[str, Any]] = []
    for layer, state in sorted(artifact.items(), key=lambda item: int(item[0])):
        rows.extend(_layer_rows(int(layer), state))

    summary = {
        "source_format": "reap",
        "layers": len({row["layer"] for row in rows}),
        "experts_per_layer": max((row["expert"] for row in rows), default=-1) + 1,
        "rows": len(rows),
        "layers_summary": [],
    }
    for layer in sorted({row["layer"] for row in rows}):
        layer_rows = [row for row in rows if row["layer"] == layer]
        scores = sorted(layer_rows, key=lambda row: row["activity_score"], reverse=True)
        shares = sorted(layer_rows, key=lambda row: row["assignment_share"], reverse=True)
        entropy, gini = _distribution_stats([row["assignment_share"] for row in layer_rows])
        summary["layers_summary"].append(
            {
                "layer": layer,
                "zero_activity_experts": sum(row["selection_count"] == 0 for row in layer_rows),
                "nan_affected_experts": sum(row["nan_count"] > 0 for row in layer_rows),
                "top_experts_by_score": [row["expert"] for row in scores[:10]],
                "top_experts_by_share": [row["expert"] for row in shares[:10]],
                "top10_assignment_share": sum(row["assignment_share"] for row in shares[:10]),
                "normalized_entropy": entropy,
                "gini": gini,
                "mean_selection_rate": sum(row["selection_rate"] for row in layer_rows)
                / max(len(layer_rows), 1),
            }
        )
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="REAP observation .pt file")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = analyze(args.input)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["layer"])
        writer.writeheader()
        writer.writerows(rows)
    args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for layer in summary["layers_summary"]:
        print(
            f"layer {layer['layer']:>3}: zero={layer['zero_activity_experts']:>3} "
            f"nan={layer['nan_affected_experts']:>3} "
            f"top10_share={layer['top10_assignment_share']:.4f} "
            f"entropy={layer['normalized_entropy']:.4f} "
            f"gini={layer['gini']:.4f} "
            f"top={layer['top_experts_by_score'][:5]}"
        )


if __name__ == "__main__":
    main()
