from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Set

from experiment_components.registry import annotate_rule


def parse_component_csv(value: str | None) -> Set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def annotate_rules(rules: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [annotate_rule(rule) for rule in rules]


def filter_rules(
    rules: Iterable[Dict[str, Any]],
    *,
    enable_components: Set[str] | None = None,
    disable_components: Set[str] | None = None,
) -> List[Dict[str, Any]]:
    enable_components = enable_components or set()
    disable_components = disable_components or set()
    filtered: List[Dict[str, Any]] = []
    for rule in annotate_rules(rules):
        component = str(rule.get("component", "custom_rules"))
        if enable_components and component not in enable_components:
            continue
        if component in disable_components:
            continue
        filtered.append(rule)
    return filtered


def component_counts(rules: Iterable[Dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for rule in annotate_rules(rules):
        counts[str(rule.get("component", "custom_rules"))] += 1
    return counts

