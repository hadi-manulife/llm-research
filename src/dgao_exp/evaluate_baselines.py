from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

from .config import ExperimentConfig
from .data import build_grouped_examples, ensure_dir
from .training import evaluate_model, load_model_and_tokenizer, set_seed


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


def _load_config_file(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() == ".json":
        return json.loads(raw)

    result: Dict[str, Any] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        result[k.strip()] = _coerce_scalar(v.strip())
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline models with no fine-tuning on grouped order-variant test data"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to JSON or simple YAML config")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--eval_groups", type=int, default=None)
    parser.add_argument("--num_order_variants", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated model ids, for example: Qwen/Qwen2.5-0.5B-Instruct,google/gemma-2-2b-it",
    )
    return parser.parse_args()


def _merge_config(args: argparse.Namespace) -> ExperimentConfig:
    file_cfg = _load_config_file(args.config)
    cfg = ExperimentConfig(**file_cfg)

    override_keys = [
        "dataset_name",
        "eval_groups",
        "num_order_variants",
        "seed",
        "device",
        "output_dir",
    ]
    for key in override_keys:
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)

    return cfg


def _write_csv(out_file: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "model_name",
        "dataset_name",
        "eval_groups",
        "num_order_variants",
        "accuracy",
        "consistency_rate",
        "overconfidence_rate",
    ]
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    cfg = _merge_config(args)
    ensure_dir(cfg.output_dir)
    set_seed(cfg.seed)

    # Build grouped train/eval sets once and only evaluate on eval grouped variants.
    _, eval_data = build_grouped_examples(cfg)

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_list:
        raise ValueError("No models were provided to --models")

    rows: List[Dict[str, Any]] = []
    for model_name in model_list:
        eval_cfg = replace(cfg, model_name=model_name)
        model, tokenizer = load_model_and_tokenizer(eval_cfg)
        metrics = evaluate_model(model, tokenizer, eval_cfg, eval_data)

        row = {
            "model_name": model_name,
            "dataset_name": eval_cfg.dataset_name,
            "eval_groups": eval_cfg.eval_groups,
            "num_order_variants": eval_cfg.num_order_variants,
            "accuracy": round(metrics.accuracy, 4),
            "consistency_rate": round(metrics.consistency_rate, 4),
            "overconfidence_rate": round(metrics.overconfidence_rate, 4),
        }
        rows.append(row)

        print(
            f"model={model_name} "
            f"acc={metrics.accuracy:.2f} "
            f"cr={metrics.consistency_rate:.2f} "
            f"or={metrics.overconfidence_rate:.2f}"
        )

    output_dir = Path(cfg.output_dir)
    json_path = output_dir / "baseline_eval_results.json"
    csv_path = output_dir / "baseline_eval_results.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_csv(csv_path, rows)

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
