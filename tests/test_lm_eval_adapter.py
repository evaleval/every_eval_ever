import tempfile
from pathlib import Path

from eval_converters.lm_eval.adapter import LMEvalAdapter
from eval_converters.lm_eval.instance_level_adapter import LMEvalInstanceLevelAdapter
from eval_converters.lm_eval.utils import parse_model_args, find_samples_file
from eval_types import (
    EvaluationLog,
    EvaluatorRelationship,
    SourceDataHf,
)

DATA_DIR = Path("tests/data/lm_eval")
RESULTS_FILE = DATA_DIR / "results_2026-01-21T03-44-18.458309.json"
SAMPLES_FILE = DATA_DIR / "samples_math_perturbed_full_2026-01-21T03-44-18.458309.jsonl"


def _make_metadata_args(**overrides):
    args = {
        "source_organization_name": "TestOrg",
        "evaluator_relationship": EvaluatorRelationship.first_party,
    }
    args.update(overrides)
    return args


# ── Utility tests ──────────────────────────────────────────────────────

def test_parse_model_args_basic():
    result = parse_model_args("pretrained=EleutherAI/pythia-160m,dtype=float16")
    assert result == {"pretrained": "EleutherAI/pythia-160m", "dtype": "float16"}


def test_parse_model_args_empty():
    assert parse_model_args("") == {}
    assert parse_model_args(None) == {}


def test_parse_model_args_complex():
    result = parse_model_args(
        "pretrained=RylanSchaeffer/mem_Qwen3-93M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1,trust_remote_code=True"
    )
    assert result["pretrained"] == "RylanSchaeffer/mem_Qwen3-93M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1"
    assert result["trust_remote_code"] == "True"


def test_find_samples_file():
    found = find_samples_file(DATA_DIR, "math_perturbed_full")
    assert found is not None
    assert found.name.startswith("samples_math_perturbed_full")


def test_find_samples_file_missing():
    assert find_samples_file(DATA_DIR, "nonexistent_task") is None


# ── Adapter: transform_from_file ───────────────────────────────────────

def test_transform_from_file_returns_two_tasks():
    adapter = LMEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    assert len(logs) == 2
    for log in logs:
        assert isinstance(log, EvaluationLog)


def test_transform_from_file_model_info():
    adapter = LMEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    model = logs[0].model_info

    assert model.name == "RylanSchaeffer/mem_Qwen3-93M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1"
    assert model.id == model.name
    assert model.developer == "RylanSchaeffer"
    assert model.inference_engine.name == "transformers"
    assert model.additional_details.num_parameters == 93069280
    assert model.additional_details.dtype == "torch.bfloat16"


def test_transform_from_file_source_metadata():
    adapter = LMEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    src = logs[0].source_metadata

    assert src.source_name == "lm-evaluation-harness"
    assert src.source_type.value == "evaluation_run"
    assert src.source_organization_name == "TestOrg"


def test_transform_from_file_source_data():
    adapter = LMEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())

    # Both tasks should have HF source data
    for log in logs:
        assert isinstance(log.evaluation_results[0].source_data, SourceDataHf)

    perturbed = logs[0].evaluation_results[0].source_data
    assert perturbed.hf_repo == "stellaathena/math_perturbed_5000"
    assert perturbed.hf_split == "test"


def test_transform_from_file_evaluation_results():
    adapter = LMEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())

    # First task: math_perturbed_full with exact_match = 0.0
    perturbed_results = logs[0].evaluation_results
    assert len(perturbed_results) == 1
    assert perturbed_results[0].score_details.score == 0.0
    assert perturbed_results[0].metric_config.evaluation_description == "exact_match"
    assert perturbed_results[0].metric_config.lower_is_better is False
    assert perturbed_results[0].metric_config.min_score == 0.0
    assert perturbed_results[0].metric_config.max_score == 1.0

    # Second task: math_rephrased_full with exact_match = 0.0004
    rephrased_results = logs[1].evaluation_results
    assert rephrased_results[0].score_details.score == 0.0004


def test_transform_from_file_uncertainty():
    adapter = LMEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())

    uncertainty = logs[1].evaluation_results[0].score_details.uncertainty
    assert uncertainty is not None
    assert uncertainty.standard_error.value == 0.0002828144211304471
    assert uncertainty.standard_error.method == "bootstrap"
    assert uncertainty.num_samples == 5000


def test_transform_from_file_generation_config():
    adapter = LMEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())

    gen = logs[0].evaluation_results[0].generation_config
    assert gen is not None
    assert gen.generation_args.temperature == 0.0
    assert gen.generation_args.max_tokens == 512
    assert gen.additional_details.num_fewshot == 0


def test_transform_from_file_eval_timestamp():
    adapter = LMEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    assert logs[0].evaluation_timestamp == "1768964383"


# ── Adapter: transform_from_directory ──────────────────────────────────

def test_transform_from_directory():
    adapter = LMEvalAdapter()
    logs = adapter.transform_from_directory(DATA_DIR, _make_metadata_args())
    assert len(logs) == 2
    task_names = {r.evaluation_name for log in logs for r in log.evaluation_results}
    assert "math_perturbed_full" in task_names
    assert "math_rephrased_full" in task_names


# ── Adapter: group placeholder filtering ───────────────────────────────

def test_get_tasks_skips_group_placeholders():
    adapter = LMEvalAdapter()
    raw = {
        "results": {
            "group_task": {"alias": "group_task", " ": ""},
            "real_task": {"alias": "real_task", "acc,none": 0.5},
        }
    }
    tasks = adapter._get_tasks(raw)
    assert tasks == ["real_task"]


# ── Adapter: inference engine override ─────────────────────────────────

def test_inference_engine_override():
    adapter = LMEvalAdapter()
    metadata = _make_metadata_args(inference_engine="vllm", inference_engine_version="0.6.0")
    logs = adapter.transform_from_file(RESULTS_FILE, metadata)
    assert logs[0].model_info.inference_engine.name == "vllm"
    assert logs[0].model_info.inference_engine.version == "0.6.0"


# ── Adapter: eval_metadata tracking ───────────────────────────────────

def test_eval_metadata_stored_after_transform():
    adapter = LMEvalAdapter()
    logs = adapter.transform_from_file(RESULTS_FILE, _make_metadata_args())
    for log in logs:
        meta = adapter.get_eval_metadata(log.evaluation_id)
        assert "task_name" in meta
        assert "parent_dir" in meta


# ── Instance-level adapter ─────────────────────────────────────────────

def test_instance_level_transform_samples():
    inst_adapter = LMEvalInstanceLevelAdapter()
    logs = inst_adapter.transform_samples(
        SAMPLES_FILE,
        evaluation_id="test/eval/123",
        model_id="test-model",
        task_name="math_perturbed_full",
    )
    assert len(logs) == 10

    first = logs[0]
    assert first.sample_id == 0
    assert first.evaluation_name == "math_perturbed_full"
    assert first.model_id == "test-model"
    assert first.input.reference == "3"
    assert first.evaluation.score == 0.0
    assert first.evaluation.is_correct is False
    assert first.input.choices is None  # generation task, not MC
    assert first.sample_hash  # non-empty hash


def test_instance_level_transform_and_save():
    inst_adapter = LMEvalInstanceLevelAdapter()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = inst_adapter.transform_and_save(
            SAMPLES_FILE,
            evaluation_id="test/eval/123",
            model_id="test-model",
            task_name="math_perturbed_full",
            output_dir=tmpdir,
            file_uuid="abc123",
        )
        assert result is not None
        assert result.total_rows == 10
        assert result.format.value == "jsonl"
        assert result.checksum  # non-empty sha256
        assert Path(result.file_path).exists()
        assert "abc123_samples.jsonl" in result.file_path


def test_instance_level_transform_and_save_no_output_dir():
    inst_adapter = LMEvalInstanceLevelAdapter()
    result = inst_adapter.transform_and_save(
        SAMPLES_FILE,
        evaluation_id="test/eval/123",
        model_id="test-model",
        task_name="math_perturbed_full",
        output_dir=None,
    )
    assert result is None
