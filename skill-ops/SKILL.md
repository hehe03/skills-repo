---
name: skill-ops
description: Optimize a task-specific Codex skill or a rule-based or LLM-assisted work skill when there are business rules, prediction outputs, partial human labels, and optional trace data. Use when Codex needs to audit label-vs-skill disagreement, flag likely label errors, redesign high-confidence vs low-confidence rules, add trace instrumentation, or improve agreement without relying on more human review.
---

# Skill Ops

Use this skill to improve another work skill under weak supervision. Treat the work skill as a system with four artifacts: business rules, skill implementation, prediction results, and optional human labels or trace logs.

## Workflow

1. Read the work skill and its evaluation artifacts before proposing changes.
2. Quantify disagreement first. Compare labels as sets, not raw strings, unless ordering is meaningful.
3. Separate issues into:
   - skill implementation defects
   - incomplete or brittle business rules
   - likely human labeling errors
   - low-confidence rule disagreements that should not dominate the main score
4. Use trace evidence to explain why a case was predicted, not just whether it mismatched.
5. Improve the pipeline in this order:
   - stabilize high-confidence rules
   - demote ambiguous rules to candidate outputs
   - add trace fields that make future audits cheaper
   - only then expand rule coverage

## Deliverables

Produce these outputs unless the user asks for a subset:

- a short insight report grounded in prior methods and the current project
- a reproducible audit script or workflow
- an updated skill or skill-ops folder with reusable instructions
- a discrepancy report that highlights likely label errors and likely skill issues
- a recommended trace schema for the work skill

## Decision Rules

Use a high-confidence or low-confidence split when at least one rule depends on weak proxies, aggregated fields, or missing intermediate evidence.

Mark a case as likely label error when:

- the trace and raw fields strongly support the prediction
- the human label is unsupported by the current evidence
- the disagreement recurs in a consistent pattern across multiple rows

Mark a case as likely skill issue when:

- the business rule is clearly satisfied but the skill misses it
- the skill predicts a label with no supporting trace evidence
- disagreement clusters around one implementation branch

Treat low-confidence rule gaps separately from the main agreement rate. Keep them visible as candidate labels or review queues.

## Trace-First Audit

When trace data exists, load [references/trace-schema.md](./references/trace-schema.md) and compare the available trace against the recommended schema.

When trace data does not exist:

- add row or sample identifiers
- record stage-level decisions
- store evidence for each fired rule
- record candidate labels separately from confirmed labels
- save trace in JSONL

Use [references/playbook.md](./references/playbook.md) for the operational workflow and [references/insight-report.md](./references/insight-report.md) for the design rationale.

## Scripts

Use these bundled scripts when possible instead of rebuilding analysis from scratch:

- `scripts/label_audit.py`: audit Excel or CSV labels vs predictions, support low-confidence labels, and emit an audit workbook plus Markdown report
- `scripts/trace_jsonl_summary.py`: summarize JSONL trace files, count stages and labels, and emit a compact Markdown summary

## Output Shape

Keep the final recommendation compact and action-oriented:

- current agreement rate
- likely label-error count
- likely skill-defect count
- low-confidence disagreement count
- top rule branches to fix
- trace instrumentation gaps

## Example Triggers

- "Optimize a business skill because human labels do not match the skill output."
- "Abstract the current project into a reusable skill-ops skill."
- "Audit this rule-based agent skill and find likely mislabeled samples."
- "Add traces to this work skill and improve agreement without more human review."
