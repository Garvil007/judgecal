# Bring your own data

JudgeCal loads human-labeled datasets from CSV or JSONL files via
`llmjudgecal.datasets.loaders`. Two modes are supported: **pairwise**
(human picked "A", "B", or "tie" between two responses) and **score**
(human gave one response a numeric score on a bounded scale).

Tiny worked examples live in `examples/pairwise_sample.jsonl` and
`examples/score_sample.csv` (10 rows each, synthetic text).

## Pairwise format

Required columns:

| column | meaning |
|---|---|
| `id` | unique item id |
| `prompt` | the question/instruction shown to both responses |
| `response_a_id`, `response_a_text` | first response |
| `response_b_id`, `response_b_text` | second response |
| `human_label` | `"A"`, `"B"`, or `"tie"` |
| `criterion` | what's being judged, e.g. `"overall"`, `"factuality"` |

Optional columns: `response_a_source_model`, `response_a_source_family`,
`response_b_source_model`, `response_b_source_family` (model provenance,
needed for self-preference bias checks), and `metadata` (a JSON object,
or a JSON-object string in CSV).

```python
from llmjudgecal.datasets.loaders import load_pairwise_csv, load_pairwise_jsonl

dataset = load_pairwise_jsonl("examples/pairwise_sample.jsonl")
dataset = load_pairwise_csv("my_pairwise_data.csv")
```

## Score format

Required columns:

| column | meaning |
|---|---|
| `id` | unique item id |
| `prompt` | the question/instruction |
| `response_id`, `response_text` | the response being scored |
| `human_score` | numeric score |
| `scale_min`, `scale_max` | the scale `human_score` is on (e.g. 1 and 10) |
| `criterion` | what's being judged |

Optional columns: `response_source_model`, `response_source_family`,
`metadata`.

```python
from llmjudgecal.datasets.loaders import load_score_csv, load_score_jsonl

dataset = load_score_csv("examples/score_sample.csv")
```

## Different column names: `column_map`

If your file uses different column names, pass a `column_map` from our
canonical field name to your column name. You only need to list the
columns that differ:

```python
dataset = load_pairwise_csv(
    "my_data.csv",
    column_map={
        "id": "uid",
        "prompt": "question",
        "human_label": "winner",
    },
)
```

## Generic entry point

`load_dataset(path, mode, column_map=None)` picks the right loader from
`mode` ("pairwise" or "score") and the file's `.csv`/`.jsonl` suffix:

```python
from llmjudgecal.datasets.loaders import load_dataset

dataset = load_dataset("examples/score_sample.csv", mode="score")
```

## Errors

Every parsing problem raises `DatasetLoadError` with the file path, the
1-indexed row number, and the field name, e.g.:

```
examples/score_sample.csv, row 3, field 'human_score': 'great' is not a valid number
```

Common causes: a missing required column, an empty required value, a
`human_label` that isn't `"A"`/`"B"`/`"tie"`, a `human_score` outside
`[scale_min, scale_max]`, duplicate `id` values, or a file with no data
rows.

## Notes

- JSONL files are parsed line by line (one row's worth of memory at a
  time), so large files don't need to be loaded as a single JSON array.
- All resulting items are validated against the same pydantic schema as
  the rest of JudgeCal (`llmjudgecal.datasets.schema`), so a dataset that
  loads successfully is guaranteed to satisfy every validator there too
  (non-empty text, label/scale bounds, unique ids).
