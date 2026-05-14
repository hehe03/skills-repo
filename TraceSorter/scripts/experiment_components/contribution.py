from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List

from experiment_components.registry import COMPONENT_DESCRIPTIONS
from experiment_components.rule_filters import annotate_rules


def summarize_rule_components(rules: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for rule in annotate_rules(rules):
        component = str(rule.get("component", "custom_rules"))
        entry = summary.setdefault(
            component,
            {
                "component": component,
                "description": COMPONENT_DESCRIPTIONS.get(component, ""),
                "rules": 0,
                "bad_rules": 0,
                "good_rules": 0,
                "weight_sum": 0.0,
            },
        )
        entry["rules"] += 1
        entry["weight_sum"] += float(rule.get("weight", 0.0))
        if rule.get("label") == "goodcase":
            entry["good_rules"] += 1
        else:
            entry["bad_rules"] += 1
    return sorted(summary.values(), key=lambda item: item["component"])


def summarize_result_components(results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    case_names_by_component: Dict[str, set[str]] = defaultdict(set)
    for row in results:
        actual = row.get("label")
        predicted = row.get("predicted_label")
        has_label = actual in {"goodcase", "badcase"}
        for hit in row.get("matched_rules", []) or []:
            component = str(hit.get("component", "custom_rules"))
            entry = summary.setdefault(
                component,
                {
                    "component": component,
                    "hits": 0,
                    "cases": 0,
                    "bad_score": 0.0,
                    "good_score": 0.0,
                    "correct_cases": 0,
                    "incorrect_cases": 0,
                },
            )
            entry["hits"] += 1
            label = hit.get("label")
            weight = float(hit.get("weight", 0.0))
            if label == "goodcase":
                entry["good_score"] += weight
            else:
                entry["bad_score"] += weight
            if row.get("name") not in case_names_by_component[component]:
                case_names_by_component[component].add(str(row.get("name")))
                entry["cases"] += 1
                if has_label and predicted == actual:
                    entry["correct_cases"] += 1
                elif has_label:
                    entry["incorrect_cases"] += 1
    return sorted(summary.values(), key=lambda item: item["component"])

