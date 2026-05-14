from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import Dataset, load_dataset

from .config import ExperimentConfig


POSITIVE = "Positive"
NEGATIVE = "Negative"


@dataclass
class VariantExample:
    group_id: int
    variant_id: int
    prompt: str
    label: str
    task: str


def _label_to_text(label: int) -> str:
    return POSITIVE if label == 1 else NEGATIVE


def _sample_without_index(pool: List[dict], k: int, idx: int, rng: random.Random) -> List[dict]:
    if len(pool) <= 1:
        return []
    candidate_ids = [i for i in range(len(pool)) if i != idx]
    chosen = rng.sample(candidate_ids, k=min(k, len(candidate_ids)))
    return [pool[i] for i in chosen]


def _build_sst2_prompt(query: str, support_rows: List[dict], order: List[int]) -> str:
    lines = [
        "You are a sentiment classifier.",
        "Decide if the query sentence sentiment is Positive or Negative.",
        "Answer with exactly one word: Positive or Negative.",
        "",
        "In-context examples:",
    ]
    for pos in order:
        row = support_rows[pos]
        lines.append(f"Sentence: {row['sentence']}")
        lines.append(f"Label: {_label_to_text(int(row['label']))}")
        lines.append("")

    lines.append("Now classify the following sentence.")
    lines.append(f"Sentence: {query}")
    lines.append("Label:")
    return "\n".join(lines)


def _build_gsm8k_prompt(question: str, support_rows: List[dict], order: List[int]) -> str:
    lines = [
        "You are a math problem solver.",
        "Given examples, solve the final problem and output only the final numeric answer.",
        "",
        "In-context examples:",
    ]
    for pos in order:
        row = support_rows[pos]
        answer = str(row["answer"]).split("####")[-1].strip()
        lines.append(f"Q: {row['question']}")
        lines.append(f"A: {answer}")
        lines.append("")

    lines.append(f"Q: {question}")
    lines.append("A:")
    return "\n".join(lines)


def _extract_final_answer(raw_answer: str) -> str:
    if "####" in raw_answer:
        return raw_answer.split("####")[-1].strip()
    return raw_answer.strip()


def _build_squad_rag_prompt(question: str, docs: List[str], order: List[int]) -> str:
    lines = [
        "You are a QA assistant.",
        "Use the provided documents to answer the question with a short span.",
        "",
    ]
    for i, pos in enumerate(order):
        lines.append(f"Document {i + 1}: {docs[pos]}")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("Answer:")
    return "\n".join(lines)


def _generate_orders(num_elements: int, num_variants: int, rng: random.Random) -> List[List[int]]:
    base = list(range(num_elements))
    variants: List[List[int]] = []
    for _ in range(num_variants):
        order = base[:]
        rng.shuffle(order)
        variants.append(order)
    return variants


def _load_hf_dataset(config: ExperimentConfig):
    ds_name = config.dataset_name.lower()
    if ds_name == "sst2":
        return load_dataset("glue", "sst2", cache_dir=config.cache_dir)
    if ds_name == "gsm8k":
        return load_dataset("gsm8k", "main", cache_dir=config.cache_dir)
    if ds_name in {"squad_v2", "squadv2", "squad2"}:
        return load_dataset("squad_v2", cache_dir=config.cache_dir)
    if ds_name in {"searchqa", "search_qa", "cm17k"}:
        if not config.local_dataset_path:
            raise ValueError(
                "Dataset requires local jsonl path. Set local_dataset_path in config for "
                f"{config.dataset_name}."
            )
        return {
            config.train_split_name: Dataset.from_json(str(config.local_dataset_path)),
            config.eval_split_name: Dataset.from_json(str(config.local_dataset_path)),
        }
    raise ValueError(f"Unsupported dataset_name: {config.dataset_name}")


def _choose_eval_split(ds: Dict[str, Dataset], preferred: str) -> str:
    if preferred in ds:
        return preferred
    if "validation" in ds:
        return "validation"
    if "test" in ds:
        return "test"
    keys = list(ds.keys())
    if not keys:
        raise ValueError("No dataset splits found.")
    return keys[0]


def _build_sst2_variants(
    split_rows: List[dict],
    groups: int,
    variants: int,
    shots: int,
    seed: int,
    start_group_id: int = 0,
) -> List[VariantExample]:
    rng = random.Random(seed)
    examples: List[VariantExample] = []
    chosen_ids = rng.sample(list(range(len(split_rows))), k=min(groups, len(split_rows)))

    for offset, idx in enumerate(chosen_ids):
        row = split_rows[idx]
        support = _sample_without_index(split_rows, shots, idx, rng)
        if not support:
            continue
        orders = _generate_orders(len(support), variants, rng)
        group_id = start_group_id + offset
        label = _label_to_text(int(row["label"]))
        for variant_id, order in enumerate(orders):
            prompt = _build_sst2_prompt(row["sentence"], support, order)
            examples.append(
                VariantExample(
                    group_id=group_id,
                    variant_id=variant_id,
                    prompt=prompt,
                    label=label,
                    task="sst2",
                )
            )
    return examples


def _build_gsm8k_variants(
    split_rows: List[dict],
    groups: int,
    variants: int,
    shots: int,
    seed: int,
    start_group_id: int = 0,
) -> List[VariantExample]:
    rng = random.Random(seed)
    examples: List[VariantExample] = []
    chosen_ids = rng.sample(list(range(len(split_rows))), k=min(groups, len(split_rows)))

    for offset, idx in enumerate(chosen_ids):
        row = split_rows[idx]
        support = _sample_without_index(split_rows, shots, idx, rng)
        if not support:
            continue
        orders = _generate_orders(len(support), variants, rng)
        group_id = start_group_id + offset
        label = _extract_final_answer(str(row["answer"]))
        for variant_id, order in enumerate(orders):
            prompt = _build_gsm8k_prompt(row["question"], support, order)
            examples.append(
                VariantExample(
                    group_id=group_id,
                    variant_id=variant_id,
                    prompt=prompt,
                    label=label,
                    task="gsm8k",
                )
            )
    return examples


def _squad_answer_text(row: dict) -> str:
    ans = row.get("answers", {})
    texts = ans.get("text", []) if isinstance(ans, dict) else []
    return texts[0].strip() if texts else ""


def _build_squad_variants(
    split_rows: List[dict],
    groups: int,
    variants: int,
    docs_per_prompt: int,
    seed: int,
    start_group_id: int = 0,
) -> List[VariantExample]:
    rng = random.Random(seed)
    examples: List[VariantExample] = []
    chosen_ids = rng.sample(list(range(len(split_rows))), k=min(groups, len(split_rows)))

    for offset, idx in enumerate(chosen_ids):
        row = split_rows[idx]
        # One relevant context and N-1 distractor contexts.
        distractors = _sample_without_index(split_rows, max(docs_per_prompt - 1, 1), idx, rng)
        docs = [str(row["context"])] + [str(d["context"]) for d in distractors]
        orders = _generate_orders(len(docs), variants, rng)
        group_id = start_group_id + offset
        label = _squad_answer_text(row)

        for variant_id, order in enumerate(orders):
            prompt = _build_squad_rag_prompt(str(row["question"]), docs, order)
            examples.append(
                VariantExample(
                    group_id=group_id,
                    variant_id=variant_id,
                    prompt=prompt,
                    label=label,
                    task="squad_v2",
                )
            )
    return examples


def build_grouped_examples(config: ExperimentConfig) -> Tuple[List[VariantExample], List[VariantExample]]:
    ds = _load_hf_dataset(config)
    train_split = list(ds[config.train_split_name])
    eval_split_name = _choose_eval_split(ds, config.eval_split_name)
    eval_split = list(ds[eval_split_name])

    name = config.dataset_name.lower()
    if name == "sst2":
        train = _build_sst2_variants(
            split_rows=train_split,
            groups=config.train_groups,
            variants=config.num_order_variants,
            shots=config.sentiment_shots,
            seed=config.seed,
            start_group_id=0,
        )
        eval_data = _build_sst2_variants(
            split_rows=eval_split,
            groups=config.eval_groups,
            variants=config.num_order_variants,
            shots=config.sentiment_shots,
            seed=config.seed + 1,
            start_group_id=1_000_000,
        )
        return train, eval_data

    if name == "gsm8k":
        train = _build_gsm8k_variants(
            split_rows=train_split,
            groups=config.train_groups,
            variants=config.num_order_variants,
            shots=config.math_shots,
            seed=config.seed,
            start_group_id=0,
        )
        eval_data = _build_gsm8k_variants(
            split_rows=eval_split,
            groups=config.eval_groups,
            variants=config.num_order_variants,
            shots=config.math_shots,
            seed=config.seed + 1,
            start_group_id=1_000_000,
        )
        return train, eval_data

    if name in {"squad_v2", "squadv2", "squad2"}:
        train = _build_squad_variants(
            split_rows=train_split,
            groups=config.train_groups,
            variants=config.num_order_variants,
            docs_per_prompt=config.rag_docs,
            seed=config.seed,
            start_group_id=0,
        )
        eval_data = _build_squad_variants(
            split_rows=eval_split,
            groups=config.eval_groups,
            variants=config.num_order_variants,
            docs_per_prompt=config.rag_docs,
            seed=config.seed + 1,
            start_group_id=1_000_000,
        )
        return train, eval_data

    raise ValueError(
        f"Dataset {config.dataset_name} is not yet implemented in grouped builder."
    )


def split_for_dgao_warm_start(examples: List[VariantExample]) -> Tuple[List[VariantExample], List[VariantExample]]:
    by_group: Dict[int, List[VariantExample]] = {}
    for ex in examples:
        by_group.setdefault(ex.group_id, []).append(ex)

    group_ids = sorted(by_group.keys())
    cut = len(group_ids) // 2
    warm_ids = set(group_ids[:cut])

    warm: List[VariantExample] = []
    rl: List[VariantExample] = []
    for gid, records in by_group.items():
        if gid in warm_ids:
            warm.extend(records)
        else:
            rl.extend(records)
    return warm, rl


def group_examples(examples: Iterable[VariantExample]) -> Dict[int, List[VariantExample]]:
    grouped: Dict[int, List[VariantExample]] = {}
    for ex in examples:
        grouped.setdefault(ex.group_id, []).append(ex)
    return grouped


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
