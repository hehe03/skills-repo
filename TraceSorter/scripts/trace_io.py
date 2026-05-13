from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class TraceRecord:
    name: str
    path: Path
    trace: Any
    label: Optional[str] = None
    source: Optional[str] = None
    split: Optional[str] = None
    parse_error: Optional[str] = None


def normalize_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = value.strip().lower()
    if not text:
        return None
    aliases = {
        "good": "goodcase",
        "good_case": "goodcase",
        "goodcase": "goodcase",
        "bad": "badcase",
        "bad_case": "badcase",
        "badcase": "badcase",
    }
    return aliases.get(text, text)


def read_metadata(path: Optional[str | Path]) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}
    metadata_path = Path(path)
    rows: Dict[str, Dict[str, str]] = {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "name" not in reader.fieldnames:
            raise ValueError("metadata CSV must contain a 'name' column")
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                continue
            row = {key: (value or "").strip() for key, value in row.items()}
            row["label"] = normalize_label(row.get("label")) or ""
            row["split"] = (row.get("split") or "").lower()
            rows[name] = row
    return rows


def iter_trace_paths(input_path: str | Path) -> List[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"trace path does not exist: {path}")
    return sorted(p for p in path.rglob("*.json") if p.is_file())


def load_trace_file(path: Path) -> tuple[Any, Optional[str]]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle), None
    except Exception as exc:  # noqa: BLE001 - preserve parse failures as classifiable records.
        return {"_parse_error": str(exc)}, str(exc)


def load_records(
    input_path: str | Path,
    metadata_path: Optional[str | Path] = None,
) -> List[TraceRecord]:
    metadata = read_metadata(metadata_path)
    records: List[TraceRecord] = []
    for path in iter_trace_paths(input_path):
        trace, parse_error = load_trace_file(path)
        meta = metadata.get(path.name, {})
        records.append(
            TraceRecord(
                name=path.name,
                path=path,
                trace=trace,
                label=normalize_label(meta.get("label")),
                source=meta.get("source") or None,
                split=(meta.get("split") or "").lower() or None,
                parse_error=parse_error,
            )
        )
    return records


def split_records(records: Iterable[TraceRecord], split: str) -> List[TraceRecord]:
    split = split.lower()
    return [record for record in records if (record.split or "").lower() == split]


def records_with_labels(records: Iterable[TraceRecord]) -> List[TraceRecord]:
    return [record for record in records if record.label in {"goodcase", "badcase"}]
