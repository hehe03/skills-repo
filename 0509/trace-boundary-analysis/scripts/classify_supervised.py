import math
import statistics

from common import BAD_LABEL, GOOD_LABEL, Prediction, TraceItem
from features import FEATURE_NAMES, extract_features


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def fit_scaler(records: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    medians: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in FEATURE_NAMES:
        values = sorted(record[name] for record in records)
        medians[name] = statistics.median(values)
        if len(values) < 4:
            scales[name] = max(statistics.pstdev(values), 1.0)
        else:
            q1 = values[len(values) // 4]
            q3 = values[(len(values) * 3) // 4]
            scales[name] = max(q3 - q1, 1e-6)
    return medians, scales


def scale_features(features: dict[str, float], medians: dict[str, float], scales: dict[str, float]) -> list[float]:
    return [(features[name] - medians[name]) / scales[name] for name in FEATURE_NAMES]


def mean_vector(vectors: list[list[float]]) -> list[float]:
    return [statistics.mean(vector[index] for vector in vectors) for index in range(len(vectors[0]))]


def nearest_example(vector: list[float], examples: list[tuple[str, list[float]]]) -> tuple[str, float]:
    nearest_name = ""
    nearest_distance = float("inf")
    for name, example_vector in examples:
        distance = euclidean(vector, example_vector)
        if distance < nearest_distance:
            nearest_name = name
            nearest_distance = distance
    return nearest_name, nearest_distance


def classify(items: list[TraceItem], threshold: float = 0.85) -> list[Prediction]:
    train_items = [
        item
        for item in items
        if item.meta.split == "train" and item.meta.label in {GOOD_LABEL, BAD_LABEL}
    ]
    if not train_items:
        raise ValueError("supervised 方法需要 metadata 中存在带标签的 train 样本。")

    train_labels = {item.meta.label for item in train_items}
    if train_labels != {GOOD_LABEL, BAD_LABEL}:
        raise ValueError("supervised 方法需要 train 同时包含 goodcase 和 badcase。")

    train_feature_rows: list[tuple[str, str, dict[str, float]]] = []
    for item in train_items:
        features, _ = extract_features(item.trace)
        train_feature_rows.append((item.meta.name, item.meta.label or BAD_LABEL, features))

    medians, scales = fit_scaler([features for _, _, features in train_feature_rows])
    scaled_labeled = [
        (name, label, scale_features(features, medians, scales))
        for name, label, features in train_feature_rows
    ]
    good_examples = [(name, vector) for name, label, vector in scaled_labeled if label == GOOD_LABEL]
    bad_examples = [(name, vector) for name, label, vector in scaled_labeled if label == BAD_LABEL]
    good_centroid = mean_vector([vector for _, vector in good_examples])
    bad_centroid = mean_vector([vector for _, vector in bad_examples])
    distance_scale = max(
        statistics.mean(
            euclidean(vector, good_centroid if label == GOOD_LABEL else bad_centroid)
            for _, label, vector in scaled_labeled
        ),
        1.0,
    )

    predict_items = [item for item in items if item.meta.split != "train"] or items
    predictions: list[Prediction] = []
    for item in predict_items:
        features, _ = extract_features(item.trace)
        vector = scale_features(features, medians, scales)
        distance_to_good = euclidean(vector, good_centroid)
        distance_to_bad = euclidean(vector, bad_centroid)
        score = sigmoid((distance_to_bad - distance_to_good) / distance_scale)
        predicted_label = GOOD_LABEL if score >= threshold else BAD_LABEL
        nearest_good, _ = nearest_example(vector, good_examples)
        nearest_bad, _ = nearest_example(vector, bad_examples)
        predictions.append(
            Prediction(
                name=item.meta.name,
                source=item.meta.source,
                split=item.meta.split or "unknown",
                actual_label=item.meta.label or "unknown",
                predicted_label=predicted_label,
                detail={
                    "score": round(score, 4),
                    "confidence": round(abs(score - 0.5) * 2, 4),
                    "nearest_good": nearest_good,
                    "nearest_bad": nearest_bad,
                    "reason": f"监督法：nearest_good={nearest_good}, nearest_bad={nearest_bad}",
                },
            )
        )
    return predictions
