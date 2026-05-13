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
    effective_weight: float
    group: str
    description: str


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
    aggregation: str = "weighted",
) -> Dict[str, Any]:
    hits: List[RuleHit] = []
    raw_bad_score = 0.0
    raw_good_score = 0.0
    grouped: Dict[tuple[str, str], Dict[str, Any]] = {}
    for rule in rules:
        if not rule_matches(features, rule):
            continue
        label = rule.get("label", "badcase")
        weight = float(rule.get("weight", 0.0))
        group = str(rule.get("group") or rule.get("id", "ungrouped"))
        group_cap = float(rule.get("group_cap", weight))
        hit = RuleHit(
            rule_id=str(rule.get("id", "unnamed_rule")),
            label=label,
            weight=weight,
            effective_weight=weight,
            group=group,
            description=str(rule.get("description", "")),
        )
        hits.append(hit)
        if label == "goodcase":
            raw_good_score += weight
        else:
            raw_bad_score += weight
        key = (label, group)
        if key not in grouped:
            grouped[key] = {"score": 0.0, "cap": group_cap}
        grouped[key]["score"] += weight
        grouped[key]["cap"] = min(grouped[key]["cap"], group_cap)

    if aggregation == "weighted":
        bad_score = raw_bad_score
        good_score = raw_good_score
    elif aggregation == "group_capped":
        bad_score = 0.0
        good_score = 0.0
        for (label, _group), data in grouped.items():
            contribution = min(float(data["score"]), float(data["cap"]))
            if label == "goodcase":
                good_score += contribution
            else:
                bad_score += contribution
        for hit in hits:
            group_data = grouped[(hit.label, hit.group)]
            group_raw = float(group_data["score"])
            group_effective = min(group_raw, float(group_data["cap"]))
            hit.effective_weight = round(
                hit.weight * group_effective / group_raw,
                4,
            ) if group_raw else 0.0
    else:
        raise ValueError(f"unsupported aggregation method: {aggregation}")

    if bad_score >= bad_threshold and bad_score >= good_score:
        predicted = "badcase"
    elif good_score >= good_threshold:
        predicted = "goodcase"
    else:
        predicted = "goodcase"

    reason = "; ".join(
        f"{hit.rule_id}({hit.label}, {hit.effective_weight:g})" for hit in hits[:8]
    )
    if not reason:
        reason = "no rule matched; default to goodcase"

    return {
        "predicted_label": predicted,
        "bad_score": round(bad_score, 4),
        "good_score": round(good_score, 4),
        "raw_bad_score": round(raw_bad_score, 4),
        "raw_good_score": round(raw_good_score, 4),
        "aggregation": aggregation,
        "matched_rules": [hit.__dict__ for hit in hits],
        "reason": reason,
    }
