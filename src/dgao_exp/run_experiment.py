from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from .config import ExperimentConfig
from .data import build_grouped_examples, ensure_dir
from .training import (
    evaluate_model,
    load_model_and_tokenizer,
    run_dgao_training,
    run_sft_training,
    save_results,
    set_seed,
)


def _load_config_file(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() == ".json":
        return json.loads(raw)

    # Minimal YAML support without extra dependencies.
    # If full YAML is needed, user can provide JSON instead.
    result: Dict[str, Any] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        result[k.strip()] = _coerce_scalar(v.strip())
    return result


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DGAO reproduction experiment")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON or simple YAML config")
    parser.add_argument("--mode", type=str, default=None, choices=["sft", "paft", "dgao"])
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_groups", type=int, default=None)
    parser.add_argument("--eval_groups", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def _merge_config(args: argparse.Namespace) -> ExperimentConfig:
    file_cfg = _load_config_file(args.config)
    base = ExperimentConfig(**file_cfg)

    override_keys = [
        "mode",
        "dataset_name",
        "model_name",
        "output_dir",
        "train_groups",
        "eval_groups",
        "epochs",
        "seed",
        "device",
    ]
    for key in override_keys:
        value = getattr(args, key)
        if value is not None:
            setattr(base, key, value)
    return base


def _select_train_view(mode: str, train_data):
    if mode == "sft":
        by_group = {}
        for ex in train_data:
            by_group.setdefault(ex.group_id, []).append(ex)
        selected = []
        for records in by_group.values():
            selected.append(sorted(records, key=lambda x: x.variant_id)[0])
        return selected
    return train_data


def main() -> None:
    args = _parse_args()
    cfg = _merge_config(args)
    ensure_dir(cfg.output_dir)
    set_seed(cfg.seed)

    train_data, eval_data = build_grouped_examples(cfg)
    model, tokenizer = load_model_and_tokenizer(cfg)

    if cfg.mode == "dgao":
        train_stats = run_dgao_training(model, tokenizer, cfg, train_data)
    else:
        train_view = _select_train_view(cfg.mode, train_data)
        train_stats = run_sft_training(model, tokenizer, cfg, train_view)

    eval_metrics = evaluate_model(model, tokenizer, cfg, eval_data)
    result_file = save_results(Path(cfg.output_dir), cfg, train_stats, eval_metrics)

    print("Run completed")
    print(f"mode={cfg.mode} dataset={cfg.dataset_name} model={cfg.model_name}")
    print(
        "metrics: "
        f"accuracy={eval_metrics.accuracy:.2f}, "
        f"consistency_rate={eval_metrics.consistency_rate:.2f}, "
        f"overconfidence_rate={eval_metrics.overconfidence_rate:.2f}"
    )
    print(f"results_file={result_file}")


if __name__ == "__main__":
    main()
