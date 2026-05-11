import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


GOOD_LABEL = "goodcase"
BAD_LABEL = "badcase"
OUTLINE_TASK_KEYWORD = "生成大纲"


@dataclass(frozen=True)
class MetadataRow:
    name: str
    source: str
    split: str | None
    label: str | None


@dataclass(frozen=True)
class TraceItem:
    meta: MetadataRow
    trace: dict[str, Any]


@dataclass(frozen=True)
class Prediction:
    name: str
    source: str
    split: str
    actual_label: str
    predicted_label: str
    detail: dict[str, Any]


def normalize_label(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().lower().replace(" ", "").replace("_", "")
    if not cleaned:
        return None
    if cleaned in {"good", "goodcase", "1", "true"}:
        return GOOD_LABEL
    if cleaned in {"bad", "badcase", "0", "false"}:
        return BAD_LABEL
    raise ValueError(f"Unknown label: {value}")


def normalize_split(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    if cleaned not in {"train", "test"}:
        raise ValueError(f"Unknown split: {value}")
    return cleaned


def load_metadata(path: Path) -> dict[str, MetadataRow]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if "name" not in set(reader.fieldnames or []):
            raise ValueError("Metadata CSV must include a name column.")

        rows: dict[str, MetadataRow] = {}
        for index, row in enumerate(reader, start=2):
            name = (row.get("name") or "").strip()
            if not name:
                raise ValueError(f"Metadata row {index} has empty name.")
            rows[name] = MetadataRow(
                name=name,
                source=(row.get("source") or "").strip() or "unknown",
                split=normalize_split(row.get("split")),
                label=normalize_label(row.get("label")),
            )
    return rows


def load_trace(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Trace JSON root must be an object: {path}")
    return data


def discover_trace_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".json"
        )
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def build_items(
    trace_files: list[Path],
    metadata: dict[str, MetadataRow] | None,
    split: str | None,
) -> list[TraceItem]:
    items: list[TraceItem] = []
    for trace_file in trace_files:
        meta = metadata.get(trace_file.name) if metadata else None
        if split is not None and (meta is None or meta.split != split):
            continue
        items.append(
            TraceItem(
                meta=meta
                or MetadataRow(
                    name=trace_file.name,
                    source="unknown",
                    split=None,
                    label=None,
                ),
                trace=load_trace(trace_file),
            )
        )
    return items


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def task_signature(task: dict[str, Any], include_args: bool = True) -> str:
    task_name = normalize_text(task.get("task_name"))
    command = task.get("command")
    command_name = ""
    command_args = ""
    if isinstance(command, dict):
        command_name = normalize_text(command.get("name"))
        if include_args:
            command_args = normalize_text(command.get("args"))
    return f"{task_name}||{command_name}||{command_args}"


def badcase_confusion(predictions: Iterable[Prediction]) -> dict[str, int]:
    matrix = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for prediction in predictions:
        actual_bad = prediction.actual_label == BAD_LABEL
        predicted_bad = prediction.predicted_label == BAD_LABEL
        if actual_bad and predicted_bad:
            matrix["tp"] += 1
        elif not actual_bad and predicted_bad:
            matrix["fp"] += 1
        elif actual_bad and not predicted_bad:
            matrix["fn"] += 1
        else:
            matrix["tn"] += 1
    return matrix


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def badcase_metrics(matrix: dict[str, int]) -> dict[str, float]:
    precision = safe_divide(matrix["tp"], matrix["tp"] + matrix["fp"])
    recall = safe_divide(matrix["tp"], matrix["tp"] + matrix["fn"])
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def print_predictions(predictions: list[Prediction], strategy: str) -> None:
    print("sample\tpredicted_label\tstrategy\treason")
    for prediction in predictions:
        print(
            f"{prediction.name}\t{prediction.predicted_label}\t{strategy}\t{prediction.detail.get('reason', '')}"
        )


def print_metrics(predictions: list[Prediction]) -> None:
    labeled = [
        prediction
        for prediction in predictions
        if prediction.actual_label in {GOOD_LABEL, BAD_LABEL}
    ]
    if not labeled:
        return
    matrix = badcase_confusion(labeled)
    metrics = badcase_metrics(matrix)
    print("")
    print("badcase_confusion\tpred_badcase\tpred_goodcase")
    print(f"actual_badcase\t{matrix['tp']}\t{matrix['fn']}")
    print(f"actual_goodcase\t{matrix['fp']}\t{matrix['tn']}")
    print(
        "badcase_metrics\tprecision={:.4f}\trecall={:.4f}\tf1={:.4f}".format(
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )
    )


def serialize_predictions(predictions: list[Prediction], strategy: str) -> list[dict[str, Any]]:
    return [
        {
            "sample": prediction.name,
            "source": prediction.source,
            "split": prediction.split,
            "actual_label": prediction.actual_label,
            "predicted_label": prediction.predicted_label,
            "strategy": strategy,
            "detail": prediction.detail,
        }
        for prediction in predictions
    ]
