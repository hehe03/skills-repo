import math
import statistics
from typing import Any

from common import BAD_LABEL, GOOD_LABEL, Prediction, TraceItem
from features import FEATURE_NAMES, extract_features


SUCCESS_TASK_KEYWORD = "生成大纲"


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


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


def structure_risk(features: dict[str, float]) -> float:
    no_steps = 1.0 if features["step_count_log"] == 0 else 0.0
    missing_command = features["missing_command_ratio"]
    return clamp(0.70 * no_steps + 0.30 * missing_command)


def repeat_risk(features: dict[str, float]) -> float:
    repeat_penalty = clamp((features["max_same_run"] - 1.0) / 2.0)
    loop_penalty = clamp((features["max_loop_repeats"] - 1.0) / 2.0)
    same_result_penalty = clamp((features["max_same_result_run"] - 1.0) / 2.0)
    revisit_penalty = clamp(features["revisit_ratio"])
    return clamp(
        0.35 * loop_penalty
        + 0.30 * repeat_penalty
        + 0.20 * same_result_penalty
        + 0.15 * revisit_penalty
    )


def result_risk(features: dict[str, float]) -> float:
    low_nonempty = 1.0 - features["result_nonempty_ratio"]
    low_final = 1.0 - clamp(features["final_result_log"] / math.log1p(120))
    low_avg = 1.0 - clamp(features["avg_result_log"] / math.log1p(80))
    low_result_diversity = 1.0 - features["unique_result_ratio"]
    return clamp(
        0.35 * low_nonempty
        + 0.25 * low_final
        + 0.20 * low_avg
        + 0.20 * low_result_diversity
    )


def flow_risk(features: dict[str, float]) -> float:
    no_success_task = 1.0 - features["has_outline"]
    low_task_diversity = 1.0 - features["unique_task_ratio"]
    low_command_diversity = 1.0 - features["unique_command_ratio"]
    # “生成大纲”是较大权重特征，但不是绝对证据。
    return clamp(
        0.55 * no_success_task
        + 0.25 * low_task_diversity
        + 0.20 * low_command_diversity
    )


def semantic_risk(features: dict[str, float]) -> float:
    low_overlap = 1.0 - features["query_task_overlap"]
    low_task_entropy = 1.0 - clamp(features["task_entropy"] / math.log1p(8))
    low_command_entropy = 1.0 - clamp(features["command_entropy"] / math.log1p(8))
    return clamp(0.50 * low_overlap + 0.25 * low_task_entropy + 0.25 * low_command_entropy)


def good_evidence_score(features: dict[str, float], centrality: float, use_centrality: bool) -> float:
    centrality_bonus = 0.08 * (centrality - 0.5) if use_centrality else 0.0
    raw = 0.0
    raw += 1.15 * features["has_outline"]
    raw += 0.85 * features["result_nonempty_ratio"]
    raw += 0.55 * features["unique_task_ratio"]
    raw += 0.40 * features["unique_command_ratio"]
    raw += 0.35 * clamp(features["final_result_log"] / math.log1p(120))
    raw += 0.25 * features["query_task_overlap"]
    raw += centrality_bonus
    raw -= 1.35
    return sigmoid(raw)


def hybrid_bad_risk(features: dict[str, float]) -> dict[str, float]:
    risks = {
        "structure_risk": structure_risk(features),
        "repeat_risk": repeat_risk(features),
        "result_risk": result_risk(features),
        "flow_risk": flow_risk(features),
        "semantic_risk": semantic_risk(features),
    }
    risks["bad_risk"] = clamp(
        0.18 * risks["structure_risk"]
        + 0.22 * risks["repeat_risk"]
        + 0.24 * risks["result_risk"]
        + 0.26 * risks["flow_risk"]
        + 0.10 * risks["semantic_risk"]
    )
    return risks


def build_reason(summary: dict[str, Any], risks: dict[str, float], good_score: float) -> str:
    reasons: list[str] = []
    if summary["outline_count"]:
        reasons.append(f"{SUCCESS_TASK_KEYWORD}任务数={summary['outline_count']}")
    else:
        reasons.append(f"未出现{SUCCESS_TASK_KEYWORD}任务")
    if summary["max_loop_repeats"] >= 2:
        reasons.append(f"连续循环次数={summary['max_loop_repeats']}")
    if summary["max_same_run"] >= 2:
        reasons.append(f"连续重复次数={summary['max_same_run']}")
    if summary["max_same_result_run"] >= 2:
        reasons.append(f"连续重复结果次数={summary['max_same_result_run']}")
    if summary["missing_command_ratio"] > 0:
        reasons.append(f"缺失command比例={summary['missing_command_ratio']:.2f}")
    reasons.append(f"result非空比例={summary['result_nonempty_ratio']:.2f}")
    reasons.append(f"最终结果长度={summary['final_result_length']}")
    reasons.append(f"bad_risk={risks['bad_risk']:.3f}")
    reasons.append(f"good_score={good_score:.3f}")
    return "；".join(reasons)


def classify(
    items: list[TraceItem],
    threshold: float = 0.52,
    bad_risk_threshold: float = 0.62,
    good_margin: float = 0.08,
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
        risks = hybrid_bad_risk(features)
        good_score = good_evidence_score(features, centrality, use_centrality)
        centrality_adjustment = centrality_weight * (0.5 - centrality) if use_centrality else 0.0
        bad_score = clamp(risks["bad_risk"] + centrality_adjustment)

        predicted_label = (
            BAD_LABEL
            if bad_score >= bad_risk_threshold and bad_score - good_score >= good_margin
            else GOOD_LABEL
            if good_score >= threshold
            else BAD_LABEL
        )
        predictions.append(
            Prediction(
                name=item.meta.name,
                source=item.meta.source,
                split=item.meta.split or "unknown",
                actual_label=item.meta.label or "unknown",
                predicted_label=predicted_label,
                detail={
                    "bad_score": round(bad_score, 4),
                    "good_score": round(good_score, 4),
                    "structure_risk": round(risks["structure_risk"], 4),
                    "repeat_risk": round(risks["repeat_risk"], 4),
                    "result_risk": round(risks["result_risk"], 4),
                    "flow_risk": round(risks["flow_risk"], 4),
                    "semantic_risk": round(risks["semantic_risk"], 4),
                    "centrality": round(centrality, 4) if use_centrality else None,
                    "success_task_keyword": SUCCESS_TASK_KEYWORD,
                    "reason": build_reason(summary, risks, good_score),
                },
            )
        )
    return predictions
