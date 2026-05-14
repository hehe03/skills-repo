from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set

from experiment_components.rule_filters import component_counts, filter_rules


@dataclass(frozen=True)
class AblationVariant:
    name: str
    rules: List[Dict[str, Any]]
    disabled_components: tuple[str, ...] = ()
    enabled_components: tuple[str, ...] = ()


def build_ablation_variants(
    rules: Iterable[Dict[str, Any]],
    *,
    plan: str,
    enable_components: Set[str] | None = None,
    disable_components: Set[str] | None = None,
) -> List[AblationVariant]:
    enable_components = enable_components or set()
    disable_components = disable_components or set()
    baseline_rules = filter_rules(
        rules,
        enable_components=enable_components,
        disable_components=disable_components,
    )
    variants = [
        AblationVariant(
            name="baseline",
            rules=baseline_rules,
            enabled_components=tuple(sorted(enable_components)),
            disabled_components=tuple(sorted(disable_components)),
        )
    ]
    if plan == "none":
        return variants

    components = sorted(component_counts(baseline_rules))
    if plan == "leave_one_component_out":
        for component in components:
            disabled = set(disable_components)
            disabled.add(component)
            variants.append(
                AblationVariant(
                    name=f"without_{component}",
                    rules=filter_rules(
                        rules,
                        enable_components=enable_components,
                        disable_components=disabled,
                    ),
                    enabled_components=tuple(sorted(enable_components)),
                    disabled_components=tuple(sorted(disabled)),
                )
            )
        return variants

    if plan == "only_one_component":
        for component in components:
            variants.append(
                AblationVariant(
                    name=f"only_{component}",
                    rules=filter_rules(rules, enable_components={component}),
                    enabled_components=(component,),
                    disabled_components=(),
                )
            )
        return variants

    raise ValueError(f"Unsupported ablation plan: {plan}")

