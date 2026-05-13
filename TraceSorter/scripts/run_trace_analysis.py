from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from features import extract_features
from rule_engine import classify_features, load_rules
from trace_io import load_records


SCRIPT_DIR = Path(__file__).resolve().parent
GENERAL_RULES = SCRIPT_DIR / "rules" / "static" / "general_rules.json"
UNLABELED_RULES = SCRIPT_DIR / "rules" / "dynamic" / "unlabeled_rules.json"
LABELED_RULES = SCRIPT_DIR / "rules" / "dynamic" / "labeled_rules.json"
LLM_RULES = SCRIPT_DIR / "rules" / "dynamic" / "llm_rules.json"


def rule_paths_for_layer(layer: str) -> List[Path]:
    paths = [GENERAL_RULES]
    if layer in {"unlabeled", "all"}:
        paths.append(UNLABELED_RULES)
    if layer in {"labeled", "all"}:
        paths.append(LABELED_RULES)
    if layer in {"llm", "all"}:
        paths.append(LLM_RULES)
    return paths


def classify_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    records = load_records(args.trace_path, args.metadata)
    rules = load_rules(rule_paths_for_layer(args.rule_layer))
    results: List[Dict[str, Any]] = []
    for record in records:
        features = extract_features(record)
        prediction = classify_features(
            features,
            rules,
            bad_threshold=args.bad_threshold,
            good_threshold=args.good_threshold,
        )
        results.append(
            {
                "name": record.name,
                "path": str(record.path),
                "label": record.label,
                "source": record.source,
                "split": record.split,
                **prediction,
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify Agent traces as goodcase or badcase.")
    parser.add_argument("trace_path", help="Trace JSON file or directory containing JSON trace files.")
    parser.add_argument("--metadata", help="Optional metadata CSV with name,label,source,split columns.")
    parser.add_argument(
        "--rule-layer",
        choices=["general", "unlabeled", "labeled", "llm", "all"],
        default="general",
        help="Rule layer to load. Defaults to generic static rules.",
    )
    parser.add_argument("--bad-threshold", type=float, default=0.60)
    parser.add_argument("--good-threshold", type=float, default=0.50)
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = classify_records(args)
    if args.output:
        Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("name\tlabel\tpredicted_label\tbad_score\tgood_score\treason")
    for row in results:
        print(
            f"{row['name']}\t{row.get('label') or ''}\t{row['predicted_label']}\t"
            f"{row['bad_score']}\t{row['good_score']}\t{row['reason']}"
        )


if __name__ == "__main__":
    main()
