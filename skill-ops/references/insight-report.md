# Skill-Ops Insight Report

## Goal

Optimize a task skill when four things can all be wrong at once:

- business rules may be incomplete or contradictory
- the skill implementation may drift from those rules
- human labels may contain noise
- traces may be missing or too weak to explain outcomes

The practical objective is not only to raise agreement, but to raise trustworthy agreement.

## Methods To Borrow

### 1. Component-wise diagnosis

Borrow the RAG-style idea of decomposing failures by stage instead of treating every mismatch as one class of bug.

Map that idea to skill optimization:

- rule definition quality
- implementation fidelity
- label quality
- trace observability

This prevents "edit rules until they match labels" from becoming the default behavior.

### 2. Trace-based root cause analysis

Borrow the AgentTrace and AgentDiagnose pattern: inspect the decision path, not just final outputs.

For work skills, useful trace units are:

- sample id
- stage
- rule checked
- condition truth values
- evidence fields
- confirmed labels
- candidate labels
- confidence

This makes it possible to distinguish:

- correct prediction vs wrong label
- wrong prediction vs right label
- ambiguous prediction vs missing evidence

### 3. Causal or dependency thinking

Borrow the multi-step RCA idea from agent tracing work: a later error often inherits from an earlier weak decision.

For skill optimization, common chains look like:

- weak raw field parsing -> wrong condition evaluation -> wrong label
- lossy aggregation -> ambiguous rule firing -> unstable labels
- no trace evidence -> impossible audit -> false rule edits

### 4. Noisy-label handling

Treat human labels as useful but fallible supervision. This suggests:

- compare normalized label sets
- cluster mismatches by pattern
- separate high-confidence from low-confidence labels
- identify repeated unsupported human tags as likely annotation issues

## Recommended Project Pattern

### Phase 1. Baseline and segmentation

- compute agreement by normalized label sets
- count per-label gaps
- group mismatches into repeated patterns

### Phase 2. High-confidence stabilization

- identify labels with direct, auditable evidence
- make those labels the primary output set
- stop low-confidence rules from dominating the score

### Phase 3. Trace enrichment

- emit JSONL traces
- store rule-level evidence
- preserve candidate labels and stage decisions

### Phase 4. Rule and skill improvement

- patch clear implementation bugs first
- revise rules only when repeated evidence shows the rule itself is weak
- keep ambiguous branches as candidate outputs until evidence improves

## Practical Heuristics

- If a mismatch repeats with the same field pattern, it is usually a rule or implementation issue, not random annotation noise.
- If a human-only label has no supporting evidence across many rows, it is likely a labeling standard problem.
- If a model-only label is supported by direct trace evidence, treat it as a likely human miss before weakening the rule.
- If a label depends on aggregate substitute fields, free-text relations, or missing intermediate checks, demote it to low confidence.

## What A Good Skill-Ops System Produces

- reproducible audit outputs
- case-level explanations
- clear separation of confirmed vs candidate labels
- an explicit trace schema
- change recommendations with evidence
