# Recommended Trace Schema For Work Skills

Use JSONL with one object per sample or decision unit.

## Minimum Fields

```json
{
  "sample_id": "row-17",
  "stage": "rule-evaluation",
  "confirmed_labels": ["label-a"],
  "candidate_labels": ["label-b"],
  "trace_digest": "label-a: cond1, cond2",
  "rule_hits": [
    {
      "label": "label-a",
      "hit": true,
      "conditions": {
        "cond1": true,
        "cond2": true
      },
      "evidence": {
        "field_x": 10,
        "field_y": 3
      },
      "reason": "short explanation"
    }
  ]
}
```

## Recommended Additions

- `input_snapshot`: raw fields used in the decision
- `confidence`: numeric or bucketed confidence
- `source_artifacts`: rule doc path, skill version, prompt version
- `tool_calls`: any external tools used during scoring
- `timing_ms`: runtime per sample or stage
- `error`: parsing or execution failure detail

## Why This Matters

This schema supports:

- row-level RCA
- label-quality audits
- high-confidence vs low-confidence output separation
- reproducible before/after comparisons

## Minimum Join Keys

The trace must share at least one stable key with the dataset:

- `sample_id`
- or source row index
- or a deterministic hash over business keys

Without a stable join key, trace analysis becomes anecdotal and cannot drive systematic optimization.
