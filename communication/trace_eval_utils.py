import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


GOOD_LABEL = "goodcase"
BAD_LABEL = "badcase"


@dataclass(frozen=True)
class MetadataRow:
    name: str
    label: str
    source: str
    split: str


@dataclass(frozen=True)
class TraceRecord:
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


def normalize_label(value: str) -> str:
    cleaned = value.strip().lower().replace(" ", "").replace("_", "")
    if cleaned in {"good", "goodcase", "1", "true"}:
        return GOOD_LABEL
    if cleaned in {"bad", "badcase", "0", "false"}:
        return BAD_LABEL
    raise ValueError(f"Unknown label: {value}")


def normalize_split(value: str) -> str:
    cleaned = value.strip().lower()
    if cleaned not in {"train", "test"}:
        raise ValueError(f"Unknown split: {value}")
    return cleaned


def load_metadata(path: Path) -> list[MetadataRow]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"name", "label", "source", "split"}
        fieldnames = set(reader.fieldnames or [])
        missing = required - fieldnames
        if missing:
            raise ValueError("Metadata CSV missing columns: " + ", ".join(sorted(missing)))

        rows: list[MetadataRow] = []
        for index, row in enumerate(reader, start=2):
            name = (row.get("name") or "").strip()
            if not name:
                raise ValueError(f"Metadata row {index} has empty name.")
            rows.append(
                MetadataRow(
                    name=name,
                    label=normalize_label(row.get("label") or ""),
                    source=(row.get("source") or "").strip() or "unknown",
                    split=normalize_split(row.get("split") or ""),
                )
            )
    return rows


def filter_metadata_by_split(rows: Iterable[MetadataRow], split: str | None) -> list[MetadataRow]:
    if split is None:
        return list(rows)
    return [row for row in rows if row.split == split]


def load_json_trace(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("trace JSON root must be an object")
    return data


def load_trace_records(trace_dir: Path, rows: Iterable[MetadataRow]) -> list[TraceRecord]:
    records: list[TraceRecord] = []
    for row in rows:
        trace_path = trace_dir / row.name
        if not trace_path.is_file():
            raise FileNotFoundError(f"Trace JSON not found for metadata name: {row.name}")
        records.append(TraceRecord(meta=row, trace=load_json_trace(trace_path)))
    return records


def format_count(correct: int, total: int) -> str:
    return f"{correct}/{total}"


def source_label_counts(predictions: list[Prediction]) -> dict[str, dict[str, tuple[int, int]]]:
    counts: dict[str, dict[str, list[int]]] = {}
    for prediction in predictions:
        source_counts = counts.setdefault(
            prediction.source,
            {GOOD_LABEL: [0, 0], BAD_LABEL: [0, 0]},
        )
        label_counts = source_counts[prediction.actual_label]
        label_counts[1] += 1
        if prediction.actual_label == prediction.predicted_label:
            label_counts[0] += 1

    return {
        source: {
            label: (values[0], values[1])
            for label, values in source_counts.items()
        }
        for source, source_counts in counts.items()
    }


def badcase_confusion(predictions: list[Prediction]) -> dict[str, int]:
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
    if denominator == 0:
        return 0.0
    return numerator / denominator


def badcase_metrics(matrix: dict[str, int]) -> dict[str, float]:
    precision = safe_divide(matrix["tp"], matrix["tp"] + matrix["fp"])
    recall = safe_divide(matrix["tp"], matrix["tp"] + matrix["fn"])
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def print_predictions(predictions: list[Prediction]) -> None:
    print("name\tsource\tsplit\tactual_label\tpredicted_label")
    for prediction in predictions:
        print(
            "\t".join(
                [
                    prediction.name,
                    prediction.source,
                    prediction.split,
                    prediction.actual_label,
                    prediction.predicted_label,
                ]
            )
        )


def write_predictions_tsv(predictions: list[Prediction], output: Path) -> None:
    detail_keys = sorted({key for prediction in predictions for key in prediction.detail})
    fieldnames = [
        "name",
        "source",
        "split",
        "actual_label",
        "predicted_label",
        *detail_keys,
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for prediction in predictions:
            row = {
                "name": prediction.name,
                "source": prediction.source,
                "split": prediction.split,
                "actual_label": prediction.actual_label,
                "predicted_label": prediction.predicted_label,
                **prediction.detail,
            }
            writer.writerow(row)


def build_metrics_markdown(predictions: list[Prediction], title: str) -> str:
    source_counts = source_label_counts(predictions)
    matrix = badcase_confusion(predictions)
    metrics = badcase_metrics(matrix)

    lines = [
        f"# {title}",
        "",
        f"- 样本数：{len(predictions)}",
        f"- badcase 查准：{metrics['precision']:.4f}",
        f"- badcase 召回：{metrics['recall']:.4f}",
        f"- badcase F1-score：{metrics['f1']:.4f}",
        "",
        "## 按 source 统计",
        "",
        "| source | goodcase 正确/总数 | badcase 正确/总数 |",
        "| --- | ---: | ---: |",
    ]
    for source in sorted(source_counts):
        good_correct, good_total = source_counts[source][GOOD_LABEL]
        bad_correct, bad_total = source_counts[source][BAD_LABEL]
        lines.append(
            f"| {source} | {format_count(good_correct, good_total)} | {format_count(bad_correct, bad_total)} |"
        )

    lines.extend(
        [
            "",
            "## badcase 混淆矩阵",
            "",
            "|  | 预测 badcase | 预测 goodcase |",
            "| --- | ---: | ---: |",
            f"| 实际 badcase | {matrix['tp']} | {matrix['fn']} |",
            f"| 实际 goodcase | {matrix['fp']} | {matrix['tn']} |",
            "",
            "## badcase 指标",
            "",
            "| precision | recall | F1-score |",
            "| ---: | ---: | ---: |",
            f"| {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |",
            "",
        ]
    )
    return "\n".join(lines)


def print_metrics_summary(predictions: list[Prediction]) -> None:
    source_counts = source_label_counts(predictions)
    print("")
    print("By source:")
    print("source\tgoodcase_correct/total\tbadcase_correct/total")
    for source in sorted(source_counts):
        good_correct, good_total = source_counts[source][GOOD_LABEL]
        bad_correct, bad_total = source_counts[source][BAD_LABEL]
        print(
            f"{source}\t{format_count(good_correct, good_total)}\t{format_count(bad_correct, bad_total)}"
        )

    matrix = badcase_confusion(predictions)
    metrics = badcase_metrics(matrix)
    print("")
    print("Badcase confusion matrix:")
    print("actual\\predicted\tbadcase\tgoodcase")
    print(f"badcase\t{matrix['tp']}\t{matrix['fn']}")
    print(f"goodcase\t{matrix['fp']}\t{matrix['tn']}")
    print(
        f"badcase precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )


def write_metrics_markdown(predictions: list[Prediction], output: Path, title: str) -> None:
    output.write_text(build_metrics_markdown(predictions, title), encoding="utf-8-sig")
