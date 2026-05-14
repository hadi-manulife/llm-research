from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List


def normalize_text(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _group_predictions(group_ids: Iterable[int], preds: Iterable[str], labels: Iterable[str]):
    grouped = defaultdict(list)
    for gid, pred, label in zip(group_ids, preds, labels):
        grouped[int(gid)].append((normalize_text(pred), normalize_text(label)))
    return grouped


@dataclass
class GroupMetrics:
    accuracy: float
    consistency_rate: float
    overconfidence_rate: float


def compute_group_metrics(group_ids: List[int], preds: List[str], labels: List[str]) -> GroupMetrics:
    grouped = _group_predictions(group_ids, preds, labels)
    if not grouped:
        return GroupMetrics(accuracy=0.0, consistency_rate=0.0, overconfidence_rate=0.0)

    total = 0
    total_correct = 0
    consistency_sum = 0.0
    overconfidence_sum = 0.0

    for gid, values in grouped.items():
        del gid
        pred_list = [p for p, _ in values]
        label = values[0][1]

        for pred in pred_list:
            total += 1
            total_correct += int(pred == label)

        pred_counter = Counter(pred_list)
        most_common_pred, max_count = pred_counter.most_common(1)[0]
        consistency_sum += max_count / len(pred_list)

        wrong_counts = [count for pred, count in pred_counter.items() if pred != label]
        max_wrong = max(wrong_counts) if wrong_counts else 0
        overconfidence_sum += max_wrong / len(pred_list)

        # Keep explicit use to avoid lint warning for most_common_pred in strict linters.
        _ = most_common_pred

    m = len(grouped)
    return GroupMetrics(
        accuracy=100.0 * total_correct / total if total > 0 else 0.0,
        consistency_rate=100.0 * consistency_sum / m,
        overconfidence_rate=100.0 * overconfidence_sum / m,
    )
