from __future__ import annotations

from typing import Any, Dict, Iterable, List


def confusion_and_scores(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [row for row in results if row.get("label") in {"goodcase", "badcase"}]
    tp = sum(1 for row in rows if row["label"] == "badcase" and row["predicted_label"] == "badcase")
    tn = sum(1 for row in rows if row["label"] == "goodcase" and row["predicted_label"] == "goodcase")
    fp = sum(1 for row in rows if row["label"] == "goodcase" and row["predicted_label"] == "badcase")
    fn = sum(1 for row in rows if row["label"] == "badcase" and row["predicted_label"] == "goodcase")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0
    return {
        "count": len(rows),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }
