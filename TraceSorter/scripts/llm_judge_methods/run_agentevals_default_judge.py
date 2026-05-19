from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import (
    load_eval_records,
    result_score_parts,
    timestamped_path,
    trace_to_openai_messages,
    write_jsonl,
    write_llm_judge_report,
)


def _create_default_evaluator(*, model: str, continuous: bool, use_reasoning: bool):
    try:
        from agentevals.trajectory.llm import (
            TRAJECTORY_ACCURACY_PROMPT,
            create_trajectory_llm_as_judge,
        )
    except ImportError as exc:
        raise RuntimeError(
            "This method requires the LangChain AgentEvals package. "
            "Install it in your environment, then rerun this script."
        ) from exc
    return create_trajectory_llm_as_judge(
        prompt=TRAJECTORY_ACCURACY_PROMPT,
        model=model,
        continuous=continuous,
        use_reasoning=use_reasoning,
    )


def run_default_judge(args: argparse.Namespace) -> Path:
    records = load_eval_records(args.trace_path, args.metadata, split=args.eval_split)
    evaluator = _create_default_evaluator(
        model=args.model,
        continuous=args.continuous,
        use_reasoning=not args.no_reasoning,
    )
    results: List[Dict[str, Any]] = []
    for record in records:
        outputs = trace_to_openai_messages(record, max_trace_chars=args.max_trace_chars)
        raw_result = evaluator(outputs=outputs)
        raw_score, predicted, reasoning, good_score, bad_score = result_score_parts(
            raw_result,
            threshold=args.good_threshold,
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

    output_dir = Path(args.output_dir)
    jsonl_path = Path(args.predictions_output) if args.predictions_output else timestamped_path(
        output_dir,
        "agentevals_default_judge_predictions",
        ".jsonl",
    )
    write_jsonl(jsonl_path, results)
    report_path = Path(args.output) if args.output else timestamped_path(
        output_dir,
        "agentevals_default_judge",
        ".md",
    )
    notes = [
        "This method directly calls LangChain AgentEvals `create_trajectory_llm_as_judge()`.",
        "It uses AgentEvals' built-in no-reference trajectory accuracy prompt.",
        "No reference trajectory is provided.",
        "No TraceSorter-specific custom rubric is provided.",
        f"Predictions JSONL: `{jsonl_path}`",
    ]
    write_llm_judge_report(
        report_path,
        title="AgentEvals Default LLM-as-Judge",
        method="agentevals_default_no_reference_no_custom_rubric",
        trace_path=args.trace_path,
        metadata_path=args.metadata,
        model=args.model,
        results=results,
        notes=notes,
        max_rows=args.max_rows,
    )
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run LangChain AgentEvals' default trajectory LLM-as-Judge directly. "
            "This is intentionally separate from TraceSorter rule methods."
        )
    )
    parser.add_argument("trace_path", help="Trace JSON file or directory to evaluate.")
    parser.add_argument("--metadata", help="Optional metadata CSV with name,label,source,split.")
    parser.add_argument("--eval-split", help="Optional metadata split to evaluate.")
    parser.add_argument(
        "--model",
        required=True,
        help="LangChain model identifier for the judge, e.g. openai:gpt-4.1.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Ask AgentEvals for a continuous score from 0 to 1 instead of a boolean.",
    )
    parser.add_argument(
        "--good-threshold",
        type=float,
        default=0.5,
        help="For continuous scores, score >= threshold maps to goodcase.",
    )
    parser.add_argument("--no-reasoning", action="store_true", help="Do not ask the judge for reasoning.")
    parser.add_argument("--max-trace-chars", type=int, default=6000)
    parser.add_argument("--output-dir", default="./results/llm_judge")
    parser.add_argument("--output", help="Optional Markdown report path.")
    parser.add_argument("--predictions-output", help="Optional JSONL predictions path.")
    parser.add_argument("--max-rows", type=int, default=200)
    return parser


def main(argv: List[str] | None = None) -> None:
    report = run_default_judge(build_parser().parse_args(argv))
    print(f"Wrote report: {report}")


if __name__ == "__main__":
    main()
