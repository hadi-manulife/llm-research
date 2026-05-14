from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from .config import ExperimentConfig
from .data import build_grouped_examples, ensure_dir
from .training import evaluate_model, load_model_and_tokenizer, set_seed


DEFAULT_BASELINE_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "google/gemma-2-2b-it",
]


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

    # Minimal YAML parser for flat key-value files.
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
        description="Evaluate baseline models without any fine-tuning"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to JSON/simple YAML config")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--eval_groups", type=int, default=None)
    parser.add_argument("--num_order_variants", type=int, default=None)
    parser.add_argument("--max_prompt_length", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated Hugging Face model names",
    )
    group.add_argument(
        "--use_default_models",
        action="store_true",
        help="Evaluate a default baseline model list",
    )

    return parser.parse_args()


def _merge_config(args: argparse.Namespace) -> ExperimentConfig:
    file_cfg = _load_config_file(args.config)
    cfg = ExperimentConfig(**file_cfg)

    override_keys = [
        "dataset_name",
        "eval_groups",
        "num_order_variants",
        "max_prompt_length",
        "max_new_tokens",
        "seed",
        "device",
        "cache_dir",
        "output_dir",
    ]
    for key in override_keys:
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)

    return cfg


def _parse_models_arg(models_arg: str | None, use_default_models: bool, fallback_model: str) -> List[str]:
    if models_arg:
        models = [m.strip() for m in models_arg.split(",") if m.strip()]
        if models:
            return models
    if use_default_models:
        return DEFAULT_BASELINE_MODELS
    return [fallback_model]


def _save_baseline_results(output_dir: Path, payload: Dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "baseline_eval_results.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_file


def _build_markdown_table(results: List[Dict[str, Any]]) -> str:
    header = [
        "| Model | Dataset | Accuracy | Consistency Rate | Overconfidence Rate |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    rows: List[str] = []
    for item in results:
        metrics = item["metrics"]
        rows.append(
            "| "
            f"{item['model_name']} | "
            f"{item['dataset_name']} | "
            f"{metrics['accuracy']:.2f} | "
            f"{metrics['consistency_rate']:.2f} | "
            f"{metrics['overconfidence_rate']:.2f} |"
        )
    return "\n".join(header + rows)


def _save_markdown_summary(output_dir: Path, payload: Dict[str, Any]) -> Path:
    md_file = output_dir / "baseline_eval_results.md"
    lines = [
        "## Baseline Evaluation Results",
        "",
        _build_markdown_table(payload.get("results", [])),
        "",
        "### Run Configuration",
        "",
        "```json",
        json.dumps(payload.get("config", {}), indent=2),
        "```",
    ]
    md_file.write_text("\n".join(lines), encoding="utf-8")
    return md_file


def main() -> None:
    args = _parse_args()
    cfg = _merge_config(args)
    cfg.mode = "baseline_eval"

    ensure_dir(cfg.output_dir)
    set_seed(cfg.seed)

    # Reuse the same grouped construction logic and evaluate only on test/eval groups.
    _, eval_data = build_grouped_examples(cfg)

    model_names = _parse_models_arg(
        models_arg=args.models,
        use_default_models=args.use_default_models,
        fallback_model=cfg.model_name,
    )

    all_results: List[Dict[str, Any]] = []
    for model_name in model_names:
        run_cfg = ExperimentConfig(**asdict(cfg))
        run_cfg.model_name = model_name

        print(f"Evaluating baseline model: {model_name}")
        model, tokenizer = load_model_and_tokenizer(run_cfg)
        metrics = evaluate_model(model, tokenizer, run_cfg, eval_data)

        all_results.append(
            {
                "model_name": model_name,
                "dataset_name": run_cfg.dataset_name,
                "eval_groups": run_cfg.eval_groups,
                "num_order_variants": run_cfg.num_order_variants,
                "metrics": {
                    "accuracy": metrics.accuracy,
                    "consistency_rate": metrics.consistency_rate,
                    "overconfidence_rate": metrics.overconfidence_rate,
                },
            }
        )

        print(
            "metrics: "
            f"accuracy={metrics.accuracy:.2f}, "
            f"consistency_rate={metrics.consistency_rate:.2f}, "
            f"overconfidence_rate={metrics.overconfidence_rate:.2f}"
        )

    payload = {
        "config": asdict(cfg),
        "results": all_results,
    }
    output_file = _save_baseline_results(Path(cfg.output_dir), payload)
    markdown_file = _save_markdown_summary(Path(cfg.output_dir), payload)
    print(f"Saved baseline evaluation results to: {output_file}")
    print(f"Saved markdown summary table to: {markdown_file}")


if __name__ == "__main__":
    main()
