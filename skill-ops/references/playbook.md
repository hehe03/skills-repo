# Skill-Ops Playbook

## Inputs

Collect as many of these as exist:

- work skill folder or prompt
- rule document
- prediction result file
- human-labeled result file or columns
- trace JSONL or logs
- a few representative mismatches

## Standard Workflow

### 1. Build the evaluation table

- load the data
- identify prediction column and human column
- normalize multivalue labels into sets
- preserve raw columns used as evidence

### 2. Compute the first-pass audit

Run `scripts/label_audit.py` and capture:

- agreement rate
- mismatch types
- per-label counts
- likely label-error rows

### 3. Inspect recurring mismatch families

For each family, decide whether it is:

- implementation gap
- rule gap
- human label noise
- low-confidence disagreement

Do not jump directly from disagreement to rule change.

### 4. Apply the confidence split

Create two output layers:

- confirmed labels: direct and auditable
- candidate labels: weak or incomplete evidence

Use confirmed labels for the main consistency metric.

### 5. Improve trace quality

If current trace is thin, add:

- per-sample identifiers
- condition booleans per rule
- evidence values used in each decision
- confirmed and candidate labels
- stage summary text

### 6. Patch the work skill

Change the work skill in this order:

- parsing and normalization
- rule implementation fidelity
- result formatting
- business-rule design

### 7. Re-run and compare

Report:

- before/after agreement
- rows moved from mismatch to match
- remaining likely label errors
- remaining low-confidence gaps

## Escalation Guidance

Pause and explicitly call out tradeoffs when:

- improving agreement would require weakening a high-confidence rule
- a business rule conflicts with repeated trace evidence
- the user appears to optimize for agreement rather than correctness

## Output Template

- baseline agreement: X
- improved agreement: Y
- likely label errors: N rows
- likely skill issues: M rows
- low-confidence disagreements: K rows
- next rule changes: 1-3 items
- next trace changes: 1-3 items
