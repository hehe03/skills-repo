from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from common import (
    compact_record_payload,
    load_train_eval_records,
    result_score_parts,
    sample_labeled_records,
    timestamped_path,
    trace_to_openai_messages,
    write_jsonl,
)

from metrics import confusion_and_scores


RUBRIC_SCHEMA: Dict[str, Any] = {
    "title": "trace_rubric_set",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {
            "type": "string",
            "description": "Short summary of the discovered boundary between goodcase and badcase.",
        },
        "rubrics": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "goodcase_signal": {"type": "string"},
                    "badcase_signal": {"type": "string"},
                    "observable_evidence": {"type": "string"},
                    "false_positive_risk": {"type": "string"},
                    "weight": {"type": "number"},
                },
                "required": [
                    "id",
                    "name",
                    "description",
                    "goodcase_signal",
                    "badcase_signal",
                    "observable_evidence",
                    "false_positive_risk",
                    "weight",
                ],
            },
        },
    },
    "required": ["summary", "rubrics"],
}


def _json_from_text(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        stripped = stripped[start : end + 1]
    return json.loads(stripped)


def _invoke_json_model(*, model: str, prompt: str) -> Dict[str, Any]:
    try:
        from langchain.chat_models import init_chat_model
    except ImportError as exc:
        raise RuntimeError(
            "Rubric mining requires LangChain. Install LangChain and a model provider package, "
            "then rerun this script."
        ) from exc
    judge = init_chat_model(model=model)
    try:
        response = judge.with_structured_output(RUBRIC_SCHEMA).invoke(
            [{"role": "user", "content": prompt}]
        )
        if isinstance(response, dict):
            return response
        if hasattr(response, "model_dump"):
            return response.model_dump()
    except Exception:
        response = judge.invoke([{"role": "user", "content": prompt}])
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        return _json_from_text(str(content))
    raise RuntimeError("Rubric mining model did not return a JSON object.")


def _create_rubric_evaluator(*, model: str, rubrics: Dict[str, Any]):
    try:
        from agentevals.trajectory.llm import create_trajectory_llm_as_judge
    except ImportError as exc:
        raise RuntimeError(
            "Rubric evaluation requires the LangChain AgentEvals package. "
            "Install it in your environment, then rerun this script."
        ) from exc
    prompt = build_rubric_judge_prompt(rubrics)
    return create_trajectory_llm_as_judge(
        prompt=prompt,
        model=model,
        feedback_key="trace_rubric_goodcase",
        continuous=False,
        use_reasoning=True,
    )


def _safe_id(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_").lower()
    return cleaned[:80] or "rubric"


def normalize_rubrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    rubrics = payload.get("rubrics") or []
    normalized = []
    for index, item in enumerate(rubrics):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("id") or f"rubric_{index + 1}")
        normalized.append(
            {
                "id": str(item.get("id") or _safe_id(name)),
                "name": name,
                "description": str(item.get("description") or ""),
                "goodcase_signal": str(item.get("goodcase_signal") or ""),
                "badcase_signal": str(item.get("badcase_signal") or ""),
                "observable_evidence": str(item.get("observable_evidence") or ""),
                "false_positive_risk": str(item.get("false_positive_risk") or ""),
                "weight": float(item.get("weight") or 1.0),
            }
        )
    return {
        "summary": str(payload.get("summary") or ""),
        "rubrics": normalized,
    }


def build_mining_prompt(records: List[Any], *, max_trace_chars: int) -> str:
    examples = [
        compact_record_payload(record, max_trace_chars=max_trace_chars)
        for record in records
    ]
    return "\n".join(
        [
            "You are discovering task-specific rubrics for Agent trace goodcase/badcase classification.",
            "Do not rely on generic parse-empty-error rules unless the examples prove they separate labels.",
            "Compare labeled goodcase and badcase traces and infer observable criteria that distinguish them.",
            "Prefer criteria grounded in trace fields, tool choices, tool arguments, result usage, final outputs, and business semantics.",
            "Each rubric must be usable by a later LLM judge reading a single trace.",
            "Return only JSON matching this schema:",
            json.dumps(RUBRIC_SCHEMA, ensure_ascii=False, indent=2),
            "",
            "Labeled examples:",
            json.dumps(examples, ensure_ascii=False, indent=2),
        ]
    )


def build_refine_prompt(
    *,
    current_rubrics: Dict[str, Any],
    metrics: Dict[str, Any],
    mistakes: List[Dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "You are refining Agent trace classification rubrics.",
            "The current rubrics were evaluated against labeled traces and produced mistakes.",
            "Revise the rubrics to improve badcase precision and recall while avoiding overfitting to file names.",
            "Keep rubrics observable from a single trace. Remove weak or misleading criteria.",
            "Return only JSON matching this schema:",
            json.dumps(RUBRIC_SCHEMA, ensure_ascii=False, indent=2),
            "",
            "Current metrics:",
            json.dumps(metrics, ensure_ascii=False, indent=2),
            "",
            "Current rubrics:",
            json.dumps(current_rubrics, ensure_ascii=False, indent=2),
            "",
            "Misclassified examples:",
            json.dumps(mistakes, ensure_ascii=False, indent=2),
        ]
    )


def build_rubric_judge_prompt(rubrics: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are an expert evaluator for Agent trace quality.",
            "Score true if the trace should be classified as goodcase.",
            "Score false if the trace should be classified as badcase.",
            "Use the learned task-specific rubrics below. If evidence is mixed, prefer the label best supported by concrete trace evidence.",
            "",
            "<LearnedRubrics>",
            json.dumps(rubrics, ensure_ascii=False, indent=2),
            "</LearnedRubrics>",
            "",
            "Goodcase means the trajectory plausibly satisfies the task according to these learned rubrics.",
            "Badcase means the trajectory violates one or more important learned rubrics or fails to produce the expected task outcome.",
            "",
            "Grade this actual trajectory:",
            "",
            "<trajectory>",
            "{outputs}",
            "</trajectory>",
        ]
    )


def evaluate_records(
    records: List[Any],
    *,
    rubrics: Dict[str, Any],
    model: str,
    max_trace_chars: int,
) -> List[Dict[str, Any]]:
    evaluator = _create_rubric_evaluator(model=model, rubrics=rubrics)
    results: List[Dict[str, Any]] = []
    for record in records:
        outputs = trace_to_openai_messages(record, max_trace_chars=max_trace_chars)
        raw_result = evaluator(outputs=outputs)
        raw_score, predicted, reasoning, good_score, bad_score = result_score_parts(
            raw_result,
            threshold=0.5,
        )
        results.append(
            {
                "name": record.name,
                "label": record.label,
                "source": record.source,
                "split": record.split,
                "predicted_label": predicted,
                "raw_score": raw_score,
                "good_score": good_score,
                "bad_score": bad_score,
                "reason": reasoning,
                "raw_result": repr(raw_result),
            }
        )
    return results


def collect_mistakes(
    records_by_name: Dict[str, Any],
    results: List[Dict[str, Any]],
    *,
    max_items: int,
    max_trace_chars: int,
) -> List[Dict[str, Any]]:
    mistakes = []
    for row in results:
        if row.get("label") not in {"goodcase", "badcase"}:
            continue
        if row.get("label") == row.get("predicted_label"):
            continue
        record = records_by_name[row["name"]]
        mistakes.append(
            {
                "name": row["name"],
                "gold_label": row.get("label"),
                "predicted_label": row.get("predicted_label"),
                "judge_reason": row.get("reason"),
                "trace": compact_record_payload(record, max_trace_chars=max_trace_chars),
            }
        )
        if len(mistakes) >= max_items:
            break
    return mistakes


def write_iteration_report(
    path: Path,
    *,
    args: argparse.Namespace,
    train_source: str,
    eval_source: str,
    iteration_summaries: List[Dict[str, Any]],
    final_rubrics: Dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Iterative Rubric LLM-as-Judge Report",
        "",
        "- Method: `rubric_mining_iterative_agentevals_judge`",
        f"- Trace path: `{args.trace_path}`",
        f"- Metadata: `{args.metadata}`" if args.metadata else "- Metadata: none",
        f"- Train source: `{train_source}`",
        f"- Eval source: `{eval_source}`",
        f"- Model: `{args.model}`",
        f"- Iterations: {len(iteration_summaries)}",
        "",
        "## Iteration Metrics",
        "",
        "| iteration | train accuracy | train precision(bad) | train recall(bad) | train f1(bad) | eval accuracy | eval precision(bad) | eval recall(bad) | eval f1(bad) | train predictions | eval predictions | rubrics |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for item in iteration_summaries:
        train = item["train_metrics"]
        eval_ = item["eval_metrics"]
        lines.append(
            f"| {item['iteration']} | {train['accuracy']} | {train['precision']} | {train['recall']} | {train['f1']} | "
            f"{eval_['accuracy']} | {eval_['precision']} | {eval_['recall']} | {eval_['f1']} | "
            f"`{item['train_predictions']}` | `{item['eval_predictions']}` | `{item['rubrics_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Final Rubrics",
            "",
            final_rubrics.get("summary", ""),
            "",
            "| id | name | weight | goodcase signal | badcase signal | observable evidence | false positive risk |",
            "|---|---|---:|---|---|---|---|",
        ]
    )
    for rubric in final_rubrics.get("rubrics", []):
        cells = [
            f"`{rubric.get('id')}`",
            str(rubric.get("name", "")).replace("|", "\\|"),
            str(rubric.get("weight", "")),
            str(rubric.get("goodcase_signal", "")).replace("|", "\\|"),
            str(rubric.get("badcase_signal", "")).replace("|", "\\|"),
            str(rubric.get("observable_evidence", "")).replace("|", "\\|"),
            str(rubric.get("false_positive_risk", "")).replace("|", "\\|"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_iterations(args: argparse.Namespace) -> Path:
    train_records, eval_records, train_source, eval_source = load_train_eval_records(
        args.trace_path,
        args.metadata,
        train_trace_path=args.train_trace_path,
        train_metadata=args.train_metadata,
        train_split=args.train_split,
        eval_split=args.eval_split,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_records = sample_labeled_records(train_records, per_label=args.mine_samples_per_label)
    rubrics = normalize_rubrics(
        _invoke_json_model(
            model=args.model,
            prompt=build_mining_prompt(sample_records, max_trace_chars=args.max_trace_chars),
        )
    )

    records_by_name = {record.name: record for record in train_records}
    iteration_summaries: List[Dict[str, Any]] = []

    for iteration in range(args.iterations):
        rubrics_path = output_dir / f"rubrics_iter_{iteration}.json"
        rubrics_path.write_text(json.dumps(rubrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        train_results = evaluate_records(
            train_records,
            rubrics=rubrics,
            model=args.model,
            max_trace_chars=args.max_trace_chars,
        )
        eval_results = evaluate_records(
            eval_records,
            rubrics=rubrics,
            model=args.model,
            max_trace_chars=args.max_trace_chars,
        )
        train_predictions = output_dir / f"train_predictions_iter_{iteration}.jsonl"
        eval_predictions = output_dir / f"eval_predictions_iter_{iteration}.jsonl"
        write_jsonl(train_predictions, train_results)
        write_jsonl(eval_predictions, eval_results)
        train_metrics = confusion_and_scores(train_results)
        eval_metrics = confusion_and_scores(eval_results)
        iteration_summaries.append(
            {
                "iteration": iteration,
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
                "train_predictions": str(train_predictions),
                "eval_predictions": str(eval_predictions),
                "rubrics_path": str(rubrics_path),
            }
        )

        if iteration >= args.iterations - 1:
            break
        mistakes = collect_mistakes(
            records_by_name,
            train_results,
            max_items=args.refine_mistakes,
            max_trace_chars=args.max_trace_chars,
        )
        if not mistakes:
            break
        refine_prompt = build_refine_prompt(
            current_rubrics=rubrics,
            metrics=train_metrics,
            mistakes=mistakes,
        )
        rubrics = normalize_rubrics(_invoke_json_model(model=args.model, prompt=refine_prompt))

    report_path = Path(args.output) if args.output else timestamped_path(
        output_dir,
        "rubric_iteration",
        ".md",
    )
    return write_iteration_report(
        report_path,
        args=args,
        train_source=train_source,
        eval_source=eval_source,
        iteration_summaries=iteration_summaries,
        final_rubrics=rubrics,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Discover and iteratively refine task-specific rubrics, then run them with "
            "LangChain AgentEvals LLM-as-Judge. This is isolated from rule-generation methods."
        )
    )
    parser.add_argument("trace_path", help="Trace JSON file or directory containing eval records, or all records.")
    parser.add_argument("--metadata", help="Optional metadata CSV with name,label,source,split.")
    parser.add_argument("--train-trace-path", help="Optional separate training trace JSON file or directory.")
    parser.add_argument("--train-metadata", help="Optional training metadata CSV. Defaults to --metadata.")
    parser.add_argument("--train-split", help="Training split when using trace_path as the source.")
    parser.add_argument("--eval-split", help="Evaluation split when using trace_path as the source.")
    parser.add_argument(
        "--model",
        required=True,
        help="LangChain model identifier, e.g. openai:gpt-4.1.",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--mine-samples-per-label", type=int, default=8)
    parser.add_argument("--refine-mistakes", type=int, default=12)
    parser.add_argument("--max-trace-chars", type=int, default=5000)
    parser.add_argument("--output-dir", default="./results/llm_judge/rubric_iteration")
    parser.add_argument("--output", help="Optional Markdown report path.")
    return parser


def main(argv: List[str] | None = None) -> None:
    report = run_iterations(build_parser().parse_args(argv))
    print(f"Wrote report: {report}")


if __name__ == "__main__":
    main()
