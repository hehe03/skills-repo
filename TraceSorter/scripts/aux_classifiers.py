from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

from features import FinalAnswerConfig, extract_features
from trace_io import TraceRecord, records_with_labels


AUX_COMPONENTS = {"distance_aux", "cluster_aux"}
MAX_FEATURES = 160
MAX_CLUSTER_ITERS = 12


def parse_aux_components(value: str | Sequence[str] | None) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = list(value)
    components: List[str] = []
    for item in raw_items:
        text = str(item).strip().lower()
        if not text or text == "none":
            continue
        if text == "all":
            for component in ("distance_aux", "cluster_aux"):
                if component not in components:
                    components.append(component)
            continue
        if text not in AUX_COMPONENTS:
            raise ValueError(f"unsupported aux component: {item}")
        if text not in components:
            components.append(text)
    return components


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _is_usable_feature(key: str, value: Any) -> bool:
    if key in {
        "source",
        "split",
        "trace_field_paths",
        "final_answer_source",
        "final_answer_adopted_fields",
        "final_answer_evidence_source",
        "final_answer_evidence_strength",
    }:
        return False
    if key.startswith("field_text:"):
        return False
    return _numeric_value(value) is not None


def _select_feature_names(feature_rows: List[Dict[str, Any]]) -> List[str]:
    counts: Counter[str] = Counter()
    variances: Dict[str, float] = {}
    for row in feature_rows:
        for key, value in row.items():
            if _is_usable_feature(key, value):
                counts[key] += 1
    candidates = []
    for key, count in counts.items():
        values = [_numeric_value(row.get(key, 0.0)) or 0.0 for row in feature_rows]
        variance = statistics.pvariance(values) if len(values) > 1 else 0.0
        if variance <= 0:
            continue
        variances[key] = variance
        fixed_priority = 0
        if key.startswith(("behavior_", "schema_")):
            fixed_priority = 3
        elif key in {"step_count", "error_count", "empty_result_ratio", "repeated_action_count"}:
            fixed_priority = 2
        elif key.startswith("field_"):
            fixed_priority = 1
        candidates.append((fixed_priority, count, variance, key))
    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    return [item[3] for item in candidates[:MAX_FEATURES]]


def _mean_std(feature_rows: List[Dict[str, Any]], feature_names: Sequence[str]) -> tuple[List[float], List[float]]:
    means: List[float] = []
    stds: List[float] = []
    for feature in feature_names:
        values = [_numeric_value(row.get(feature, 0.0)) or 0.0 for row in feature_rows]
        mean = statistics.fmean(values) if values else 0.0
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        means.append(mean)
        stds.append(std if std > 1e-9 else 1.0)
    return means, stds


def _vectorize(features: Dict[str, Any], feature_names: Sequence[str], means: Sequence[float], stds: Sequence[float]) -> List[float]:
    vector: List[float] = []
    for feature, mean, std in zip(feature_names, means, stds):
        value = _numeric_value(features.get(feature, 0.0))
        vector.append(((value if value is not None else 0.0) - mean) / std)
    return vector


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / len(left))


def _centroid(vectors: Sequence[Sequence[float]]) -> List[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    return [statistics.fmean(vector[index] for vector in vectors) for index in range(width)]


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _initial_centers(vectors: List[List[float]], k: int) -> List[List[float]]:
    if not vectors:
        return []
    centers = [vectors[0]]
    while len(centers) < k:
        next_vector = max(vectors, key=lambda vector: min(_distance(vector, center) for center in centers))
        centers.append(next_vector)
    return [list(center) for center in centers]


def _kmeans(vectors: List[List[float]], k: int) -> List[Dict[str, Any]]:
    if not vectors:
        return []
    k = max(1, min(k, len(vectors)))
    centers = _initial_centers(vectors, k)
    assignments = [0] * len(vectors)
    for _ in range(MAX_CLUSTER_ITERS):
        changed = False
        for index, vector in enumerate(vectors):
            cluster_index = min(range(k), key=lambda item: _distance(vector, centers[item]))
            if assignments[index] != cluster_index:
                assignments[index] = cluster_index
                changed = True
        for cluster_index in range(k):
            members = [vector for vector, assignment in zip(vectors, assignments) if assignment == cluster_index]
            if members:
                centers[cluster_index] = _centroid(members)
        if not changed:
            break
    clusters: List[Dict[str, Any]] = []
    for cluster_index, center in enumerate(centers):
        members = [vector for vector, assignment in zip(vectors, assignments) if assignment == cluster_index]
        distances = [_distance(vector, center) for vector in members]
        clusters.append(
            {
                "center": center,
                "size": len(members),
                "radius_p90": _percentile(distances, 0.90),
            }
        )
    return clusters


def build_aux_model(
    records: Iterable[TraceRecord],
    final_answer_config: FinalAnswerConfig | None,
    *,
    training_scenario: str,
) -> Dict[str, Any] | None:
    train_records = list(records)
    if training_scenario == "no_train" or len(train_records) < 3:
        return None
    rows = [
        {
            "record": record,
            "label": record.label,
            "features": extract_features(record, final_answer_config),
        }
        for record in train_records
    ]
    feature_rows = [row["features"] for row in rows]
    feature_names = _select_feature_names(feature_rows)
    if not feature_names:
        return None
    means, stds = _mean_std(feature_rows, feature_names)
    vectors = [_vectorize(row["features"], feature_names, means, stds) for row in rows]
    labeled_rows = [row for row in rows if row["label"] in {"goodcase", "badcase"}]
    labeled = training_scenario == "labeled" and {row["label"] for row in labeled_rows} >= {"goodcase", "badcase"}

    model: Dict[str, Any] = {
        "training_scenario": training_scenario,
        "labeled": labeled,
        "feature_names": feature_names,
        "means": means,
        "stds": stds,
        "train_count": len(rows),
    }

    if labeled:
        vectors_by_label: Dict[str, List[List[float]]] = {"goodcase": [], "badcase": []}
        for row, vector in zip(rows, vectors):
            if row["label"] in vectors_by_label:
                vectors_by_label[row["label"]].append(vector)
        centroids = {label: _centroid(items) for label, items in vectors_by_label.items()}
        model["distance"] = {
            "centroids": centroids,
            "bad_margin": 0.03,
            "good_margin": 0.03,
        }
        prototypes: List[Dict[str, Any]] = []
        for label, label_vectors in vectors_by_label.items():
            k = max(1, min(3, round(math.sqrt(len(label_vectors)))))
            for cluster in _kmeans(label_vectors, k):
                cluster["label"] = label
                prototypes.append(cluster)
        model["cluster"] = {"prototypes": prototypes}
    else:
        center = _centroid(vectors)
        distances = [_distance(vector, center) for vector in vectors]
        clusters = _kmeans(vectors, max(2, min(4, round(math.sqrt(len(vectors))))))
        total = len(vectors)
        for cluster in clusters:
            cluster["size_ratio"] = cluster["size"] / total if total else 0.0
        model["distance"] = {
            "center": center,
            "p90": _percentile(distances, 0.90),
            "p75": _percentile(distances, 0.75),
        }
        model["cluster"] = {"clusters": clusters}
    return model


def _rule_hit(rule_id: str, label: str, weight: float, description: str, component: str) -> Dict[str, Any]:
    return {
        "rule_id": rule_id,
        "label": label,
        "weight": round(weight, 4),
        "description": description,
        "component": component,
        "feature_group": "aux_classifier",
        "source_method": "aux_classifier",
    }


def _distance_evidence(vector: List[float], model: Dict[str, Any]) -> List[Dict[str, Any]]:
    distance_model = model.get("distance") or {}
    if model.get("labeled"):
        centroids = distance_model.get("centroids") or {}
        good_center = centroids.get("goodcase") or []
        bad_center = centroids.get("badcase") or []
        if not good_center or not bad_center:
            return []
        good_distance = _distance(vector, good_center)
        bad_distance = _distance(vector, bad_center)
        margin = float(distance_model.get("bad_margin", 0.03))
        if bad_distance + margin < good_distance:
            gap = min(0.40, max(0.12, (good_distance - bad_distance) / max(good_distance, 1e-9) * 0.40))
            return [
                _rule_hit(
                    "distance_aux_near_bad_centroid",
                    "badcase",
                    gap,
                    "Auxiliary distance classifier: sample is closer to the badcase centroid than the goodcase centroid.",
                    "distance_aux",
                )
            ]
        if good_distance + float(distance_model.get("good_margin", 0.03)) < bad_distance:
            gap = min(0.25, max(0.08, (bad_distance - good_distance) / max(bad_distance, 1e-9) * 0.25))
            return [
                _rule_hit(
                    "distance_aux_near_good_centroid",
                    "goodcase",
                    gap,
                    "Auxiliary distance classifier: sample is closer to the goodcase centroid than the badcase centroid.",
                    "distance_aux",
                )
            ]
        return []

    center = distance_model.get("center") or []
    if not center:
        return []
    distance = _distance(vector, center)
    p90 = float(distance_model.get("p90", 0.0))
    p75 = float(distance_model.get("p75", 0.0))
    if p90 > 0 and distance >= p90:
        return [
            _rule_hit(
                "distance_aux_unlabeled_outlier_p90",
                "badcase",
                0.25,
                "Auxiliary distance classifier: sample is outside the unlabeled cohort p90 distance.",
                "distance_aux",
            )
        ]
    if p75 > 0 and distance >= p75:
        return [
            _rule_hit(
                "distance_aux_unlabeled_outlier_p75",
                "badcase",
                0.12,
                "Auxiliary distance classifier: sample is outside the unlabeled cohort p75 distance.",
                "distance_aux",
            )
        ]
    return []


def _cluster_evidence(vector: List[float], model: Dict[str, Any]) -> List[Dict[str, Any]]:
    cluster_model = model.get("cluster") or {}
    if model.get("labeled"):
        prototypes = cluster_model.get("prototypes") or []
        if not prototypes:
            return []
        nearest = min(prototypes, key=lambda item: _distance(vector, item["center"]))
        label = str(nearest.get("label") or "goodcase")
        distance = _distance(vector, nearest["center"])
        radius = float(nearest.get("radius_p90", 0.0))
        if radius > 0 and distance > radius * 1.50:
            return []
        weight = 0.24 if label == "badcase" else 0.16
        return [
            _rule_hit(
                f"cluster_aux_nearest_{label}_prototype",
                label,
                weight,
                f"Auxiliary cluster classifier: nearest labeled prototype is `{label}`.",
                "cluster_aux",
            )
        ]

    clusters = cluster_model.get("clusters") or []
    if not clusters:
        return []
    nearest = min(clusters, key=lambda item: _distance(vector, item["center"]))
    distance = _distance(vector, nearest["center"])
    radius = float(nearest.get("radius_p90", 0.0))
    size_ratio = float(nearest.get("size_ratio", 0.0))
    if size_ratio <= 0.15:
        return [
            _rule_hit(
                "cluster_aux_small_unlabeled_cluster",
                "badcase",
                0.20,
                "Auxiliary cluster classifier: sample belongs to a small unlabeled cluster.",
                "cluster_aux",
            )
        ]
    if radius > 0 and distance > radius * 1.75:
        return [
            _rule_hit(
                "cluster_aux_far_from_cluster",
                "badcase",
                0.18,
                "Auxiliary cluster classifier: sample is far from its nearest unlabeled cluster.",
                "cluster_aux",
            )
        ]
    return []


def aux_evidence_for_features(
    features: Dict[str, Any],
    model: Dict[str, Any] | None,
    enabled_components: Sequence[str],
) -> List[Dict[str, Any]]:
    components = parse_aux_components(enabled_components)
    if not model or not components:
        return []
    vector = _vectorize(features, model["feature_names"], model["means"], model["stds"])
    evidence: List[Dict[str, Any]] = []
    if "distance_aux" in components:
        evidence.extend(_distance_evidence(vector, model))
    if "cluster_aux" in components:
        evidence.extend(_cluster_evidence(vector, model))
    return evidence


def apply_ensemble_policy(
    base_prediction: Dict[str, Any],
    aux_hits: List[Dict[str, Any]],
    *,
    policy: str,
    bad_threshold: float,
    good_threshold: float,
) -> Dict[str, Any]:
    policy = (policy or "rules_only").strip().lower()
    if policy == "rules_only" or not aux_hits:
        return dict(base_prediction)

    bad_score = float(base_prediction.get("bad_score", 0.0))
    good_score = float(base_prediction.get("good_score", 0.0))
    for hit in aux_hits:
        if hit.get("label") == "goodcase":
            good_score += float(hit.get("weight", 0.0))
        else:
            bad_score += float(hit.get("weight", 0.0))

    aux_bad_hits = [hit for hit in aux_hits if hit.get("label") == "badcase"]
    predicted = str(base_prediction.get("predicted_label", "goodcase"))
    if policy == "aux_additive":
        if bad_score >= bad_threshold and bad_score >= good_score:
            predicted = "badcase"
        elif good_score >= good_threshold:
            predicted = "goodcase"
        else:
            predicted = "goodcase"
    elif policy == "precision_guard":
        if predicted == "badcase":
            if good_score > bad_score:
                predicted = "goodcase"
        elif len(aux_bad_hits) >= 2 and bad_score >= bad_threshold + 0.15 and bad_score >= good_score + 0.10:
            predicted = "badcase"
    else:
        raise ValueError(f"unsupported ensemble policy: {policy}")

    result = dict(base_prediction)
    result["predicted_label"] = predicted
    result["bad_score"] = round(bad_score, 4)
    result["good_score"] = round(good_score, 4)
    matched = list(result.get("matched_rules", [])) + aux_hits
    matched.append(
        _rule_hit(
            f"ensemble_policy_{policy}",
            predicted,
            0.0,
            f"Auxiliary ensemble policy applied: {policy}.",
            "ensemble_policy",
        )
    )
    result["matched_rules"] = matched
    aux_reason = "; ".join(f"{hit['rule_id']}({hit['label']}, {hit['weight']:g})" for hit in aux_hits)
    base_reason = str(result.get("reason") or "")
    result["reason"] = f"{base_reason}; {aux_reason}" if base_reason else aux_reason
    result["aux_components"] = ",".join(sorted({hit["component"] for hit in aux_hits}))
    result["ensemble_policy"] = policy
    return result
