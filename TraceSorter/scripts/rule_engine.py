from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


@dataclass
class RuleHit:
    rule_id: str
    label: str
    weight: float
    description: str
    component: str = "custom_rules"
    feature_group: str = "unknown"
    source_method: str = "unknown"


def load_rules(paths: Iterable[str | Path]) -> List[Dict[str, Any]]:
    rules: List[Dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            data = data.get("rules", [])
        if not isinstance(data, list):
            raise ValueError(f"rule file must contain a list or {{'rules': [...]}}: {path}")
        rules.extend(data)
    return rules


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def condition_matches(features: Dict[str, Any], condition: Dict[str, Any]) -> bool:
    feature = condition.get("feature")
    op = condition.get("op", "==")
    expected = condition.get("value")
    actual = features.get(feature)
    if actual is None and isinstance(feature, str):
        if feature.startswith("field_exists:"):
            actual = False
        elif feature.startswith("field_text:"):
            actual = ""
        elif feature.startswith("field_count:"):
            actual = 0
        elif feature.startswith("field_nonempty_ratio:"):
            actual = 0.0

    if op == "truthy":
        return bool(actual)
    if op == "falsey":
        return not bool(actual)
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    if op in {">", ">=", "<", "<="}:
        left = _coerce_number(actual)
        right = _coerce_number(expected)
        if left is None or right is None:
            return False
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "<":
            return left < right
        return left <= right
    if op == "contains":
        return str(expected).lower() in str(actual).lower()
    if op == "regex":
        return re.search(str(expected), str(actual), re.IGNORECASE) is not None
    raise ValueError(f"unsupported rule operator: {op}")


def rule_matches(features: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    all_conditions: Sequence[Dict[str, Any]] = rule.get("all") or []
    any_conditions: Sequence[Dict[str, Any]] = rule.get("any") or []
    if all_conditions and not all(condition_matches(features, c) for c in all_conditions):
        return False
    if any_conditions and not any(condition_matches(features, c) for c in any_conditions):
        return False
    return bool(all_conditions or any_conditions)


def classify_features(
    features: Dict[str, Any],
    rules: Iterable[Dict[str, Any]],
    bad_threshold: float = 0.60,
    good_threshold: float = 0.50,
) -> Dict[str, Any]:
    hits: List[RuleHit] = []
    bad_score = 0.0
    good_score = 0.0
    for rule in rules:
        if not rule_matches(features, rule):
            continue
        label = rule.get("label", "badcase")
        weight = float(rule.get("weight", 0.0))
        hit = RuleHit(
            rule_id=str(rule.get("id", "unnamed_rule")),
            label=label,
            weight=weight,
            description=str(rule.get("description", "")),
            component=str(rule.get("component", "custom_rules")),
            feature_group=str(rule.get("feature_group", "unknown")),
            source_method=str(rule.get("source_method", "unknown")),
        )
        hits.append(hit)
        if label == "goodcase":
            good_score += weight
        else:
            bad_score += weight

    if bad_score >= bad_threshold and bad_score >= good_score:
        predicted = "badcase"
    elif good_score >= good_threshold:
        predicted = "goodcase"
    else:
        predicted = "goodcase"

    reason = "; ".join(
        f"{hit.rule_id}({hit.label}, {hit.weight:g})" for hit in hits[:8]
    )
    if not reason:
        reason = "no rule matched; default to goodcase"

    return {
        "predicted_label": predicted,
        "bad_score": round(bad_score, 4),
        "good_score": round(good_score, 4),
        "matched_rules": [hit.__dict__ for hit in hits],
        "reason": reason,
    }
