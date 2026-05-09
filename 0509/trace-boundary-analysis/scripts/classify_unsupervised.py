import math
import statistics
from typing import Any

from common import BAD_LABEL, GOOD_LABEL, Prediction, TraceItem
from features import FEATURE_NAMES, extract_features


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def behavior_prior(features: dict[str, float]) -> float:
    length_penalty = abs(features["step_count_log"] - math.log1p(6)) / 3.0
    repeat_penalty = min((features["max_same_run"] - 1.0) / 3.0, 1.0)
    loop_penalty = min((features["max_loop_repeats"] - 1.0) / 3.0, 1.0)
    raw = 0.0
    raw += 1.15 * features["result_nonempty_ratio"]
    raw += 0.75 * features["unique_task_ratio"]
    raw += 0.55 * features["unique_command_ratio"]
    raw += 0.55 * min(features["avg_result_log"] / 5.0, 1.0)
    raw += 0.35 * min(features["final_result_log"] / 5.0, 1.0)
    raw += 0.50 * features["has_outline"]
    raw += 0.25 * features["query_task_overlap"]
    raw -= 1.15 * loop_penalty
    raw -= 0.95 * repeat_penalty
    raw -= 0.55 * features["missing_command_ratio"]
    raw -= 0.35 * min(length_penalty, 1.0)
    raw -= 1.35
    return sigmoid(raw)


def bad_risk_score(features: dict[str, float]) -> float:
    repeat_penalty = clamp((features["max_same_run"] - 1.0) / 2.0)
    loop_penalty = clamp((features["max_loop_repeats"] - 1.0) / 2.0)
    same_result_penalty = clamp((features["max_same_result_run"] - 1.0) / 2.0)
    low_final_result = 1.0 - clamp(features["final_result_log"] / math.log1p(80))
    low_avg_result = 1.0 - clamp(features["avg_result_log"] / math.log1p(60))
    low_task_diversity = 1.0 - features["unique_task_ratio"]
    low_result_diversity = 1.0 - features["unique_result_ratio"]
    no_outline = 1.0 - features["has_outline"]
    raw = 0.0
    raw += 1.50 * loop_penalty
    raw += 1.25 * repeat_penalty
    raw += 1.00 * same_result_penalty
    raw += 0.90 * (1.0 - features["result_nonempty_ratio"])
    raw += 0.75 * low_final_result
    raw += 0.50 * low_avg_result
    raw += 0.65 * low_task_diversity
    raw += 0.45 * low_result_diversity
    raw += 0.55 * features["revisit_ratio"]
    raw += 0.55 * features["missing_command_ratio"]
    raw += 0.55 * no_outline
    raw -= 0.35 * features["query_task_overlap"]
    raw -= 1.65
    return sigmoid(raw)


def euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def robust_values(records: list[dict[str, float]]) -> list[list[float]]:
    columns = {name: [record[name] for record in records] for name in FEATURE_NAMES}
    medians = {name: statistics.median(values) for name, values in columns.items()}
    iqrs: dict[str, float] = {}
    for name, values in columns.items():
        sorted_values = sorted(values)
        if len(sorted_values) < 4:
            iqrs[name] = max(statistics.pstdev(sorted_values), 1.0)
        else:
            q1 = sorted_values[len(sorted_values) // 4]
            q3 = sorted_values[(len(sorted_values) * 3) // 4]
            iqrs[name] = max(q3 - q1, 1e-6)
    return [[(record[name] - medians[name]) / iqrs[name] for name in FEATURE_NAMES] for record in records]


def centrality_scores(records: list[dict[str, float]]) -> list[float]:
    if len(records) < 3:
        return [0.5 for _ in records]
    scaled = robust_values(records)
    raw_scores: list[float] = []
    for index, vector in enumerate(scaled):
        distances = [
            euclidean(vector, other)
            for other_index, other in enumerate(scaled)
            if other_index != index
        ]
        raw_scores.append(1.0 / (1.0 + statistics.median(distances)))
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    if max_score - min_score < 1e-9:
        return [0.5 for _ in raw_scores]
    return [(score - min_score) / (max_score - min_score) for score in raw_scores]


def build_reason(summary: dict[str, Any]) -> str:
    reasons: list[str] = []
    if summary["max_loop_repeats"] >= 2:
        reasons.append(f"连续循环次数={summary['max_loop_repeats']}")
    if summary["max_same_run"] >= 2:
        reasons.append(f"连续重复次数={summary['max_same_run']}")
    if summary["max_same_result_run"] >= 2:
        reasons.append(f"连续重复结果次数={summary['max_same_result_run']}")
    if summary["outline_count"]:
        reasons.append(f"生成大纲任务数={summary['outline_count']}")
    if summary["missing_command_ratio"] > 0:
        reasons.append(f"缺失command比例={summary['missing_command_ratio']:.2f}")
    reasons.append(f"result非空比例={summary['result_nonempty_ratio']:.2f}")
    reasons.append(f"结果多样性={summary['unique_result_ratio']:.2f}")
    reasons.append(f"最终结果长度={summary['final_result_length']}")
    reasons.append(f"任务数={summary['step_count']}")
    return "；".join(reasons)


def classify(
    items: list[TraceItem],
    threshold: float = 0.55,
    bad_risk_threshold: float = 0.55,
    bad_risk_weight: float = 0.45,
    centrality_weight: float = 0.10,
) -> list[Prediction]:
    extracted = []
    for item in items:
        features, summary = extract_features(item.trace)
        extracted.append((item, features, summary))

    centralities = centrality_scores([features for _, features, _ in extracted])
    use_centrality = len(extracted) >= 3
    predictions: list[Prediction] = []
    for (item, features, summary), centrality in zip(extracted, centralities):
        prior = behavior_prior(features)
        risk = bad_risk_score(features)
        adjustment = centrality_weight * (centrality - 0.5) if use_centrality else 0.0
        score = clamp(prior + adjustment - bad_risk_weight * risk)
        predicted_label = BAD_LABEL if risk >= bad_risk_threshold else GOOD_LABEL if score >= threshold else BAD_LABEL
        predictions.append(
            Prediction(
                name=item.meta.name,
                source=item.meta.source,
                split=item.meta.split or "unknown",
                actual_label=item.meta.label or "unknown",
                predicted_label=predicted_label,
                detail={
                    "score": round(score, 4),
                    "behavior_prior": round(prior, 4),
                    "bad_risk": round(risk, 4),
                    "centrality": round(centrality, 4) if use_centrality else None,
                    "reason": build_reason(summary),
                },
            )
        )
    return predictions
