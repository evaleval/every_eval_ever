# Data Model

Every Eval Ever stores model evaluation results as aggregate JSON records, with optional instance-level JSONL for per-sample traces. The bundled aggregate schema version is `0.2.2`; the bundled instance-level schema version is `instance_level_eval_0.2.2`.

## Datastore Layout

Evaluation files in the Hugging Face datastore use this shape:

```text
data/
  {benchmark_name}/
    {developer_name}/
      {model_name}/
        {uuid}.json
        {uuid}_samples.jsonl
```

The JSON filename is a UUID. The optional samples file shares the UUID and adds `_samples.jsonl`.

## Aggregate Record

An aggregate `.json` file is validated as `EvaluationLog`. Required top-level fields are:

| Field | Purpose |
| --- | --- |
| `schema_version` | Schema version used by the record. |
| `evaluation_id` | Unique run identifier, normally `benchmark/model/retrieved_timestamp`. |
| `retrieved_timestamp` | Unix epoch timestamp for when the record was created or retrieved. |
| `source_metadata` | Source type, source organization, and evaluator relationship. |
| `model_info` | Model name, model id, developer, and inference details when known. |
| `eval_library` | Evaluation framework or source library name and version. |
| `evaluation_results` | One or more metric results for the model. |

Each `evaluation_results[]` entry must include `evaluation_name`, `source_data`, `metric_config`, and `score_details`.

```json
{
  "schema_version": "0.2.2",
  "evaluation_id": "my_eval/meta-llama/Llama-3.1-8B-Instruct/1760492095.8105888",
  "retrieved_timestamp": "1760492095.8105888",
  "source_metadata": {
    "source_type": "evaluation_run",
    "source_organization_name": "Example Lab",
    "evaluator_relationship": "third_party"
  },
  "model_info": {
    "name": "Llama 3.1 8B Instruct",
    "id": "meta-llama/Llama-3.1-8B-Instruct",
    "developer": "Meta",
    "inference_engine": {
      "name": "vllm",
      "version": "0.6.0"
    }
  },
  "eval_library": {
    "name": "lm_eval",
    "version": "unknown"
  },
  "evaluation_results": [
    {
      "evaluation_result_id": "my_eval/accuracy",
      "evaluation_name": "my_eval",
      "source_data": {
        "dataset_name": "my_eval",
        "source_type": "hf_dataset",
        "hf_repo": "example/my_eval"
      },
      "metric_config": {
        "metric_id": "accuracy",
        "metric_name": "Accuracy",
        "metric_kind": "accuracy",
        "metric_unit": "proportion",
        "lower_is_better": false,
        "score_type": "continuous",
        "min_score": 0.0,
        "max_score": 1.0
      },
      "score_details": {
        "score": 0.78
      }
    }
  ]
}
```

## Source Data

`source_data` is stored per evaluation result. Use one of three explicit variants:

- `source_type: "url"` with a non-empty `url` array.
- `source_type: "hf_dataset"` with the Hugging Face dataset repo and optional split/sample metadata.
- `source_type: "other"` for private or custom data that cannot be represented by a public URL or Hugging Face repo.

## Inference Metadata

Use `model_info.inference_platform` when the model was evaluated through a remote API or hosted platform. Use `model_info.inference_engine` when a local runtime produced the result, and include both `name` and `version` when available.

## Instance-Level Data

Instance-level rows are validated as `InstanceLevelEvaluationLog`. Use them when the source contains per-sample outputs, scoring details, transcripts, tool calls, or token/performance metadata.

The aggregate record points to the companion file through `detailed_evaluation_results`:

```json
{
  "detailed_evaluation_results": {
    "format": "jsonl",
    "file_path": "00000000-0000-4000-8000-000000000000_samples.jsonl",
    "hash_algorithm": "sha256",
    "total_rows": 1000
  }
}
```

Instance rows support `single_turn`, `multi_turn`, and `agentic` interaction types. If one sample contributes to multiple aggregate metrics, emit one instance row per aggregate `evaluation_result_id` rather than binding one row to several metrics.
