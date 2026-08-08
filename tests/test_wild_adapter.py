"""Tests for the WILD-raw adapter (every_eval_ever/adapters/wild/adapter.py). No network — builds a
tiny local parquet and runs the adapter over it."""
import argparse
import hashlib
import json
import math
import sys

import pytest

pytest.importorskip(
    'pyarrow',
    reason='pyarrow not installed; the wild adapter needs it (uv sync --extra wild)',
)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from every_eval_ever.adapters.wild import adapter  # noqa: E402
from every_eval_ever.helpers.io import SourceRecordsError  # noqa: E402
from every_eval_ever.validate import validate_file  # noqa: E402


def _synth_parquet(path):
    rows = []
    for model in ["openai/gpt-x", "01-ai/Yi-1.5-34B-Chat"]:
        for subtask in ["algebra", "logic"]:
            for i in range(3):
                score = 1 if i % 2 == 0 else 0
                # the assistant turn is the model's FULL generation (chain-of-thought);
                # it deliberately DIFFERS from the extracted `answer` ("4"/"5") and the
                # scorer's parsed answer so tests can prove output.raw != extracted_value.
                gen = f"Step by step: two plus two is four. Final answer: {'4' if score else '5'}."
                convo = json.dumps([{"role": "user", "content": "What is 2+2?"},
                                    {"role": "assistant", "content": gen}])
                rows.append(dict(
                    model=model, task="mmlu", subtask=subtask,
                    item_id=f"{model[:3]}{subtask[:2]}{i}", score=score,
                    input_tokens=100 + i, output_tokens=20 + i, conversation=convo,
                    stop_reason="stop", target="4", answer="4" if score else "5",
                    scores=json.dumps({"match": {"value": "C" if score else "I",
                                                 "answer": "the answer is 4"}})))
    pq.write_table(pa.Table.from_pylist(rows), str(path))


def _out(tmp_path):
    # the adapter publishes into <base>/wild/<developer>/<model>, deriving <base>
    # from the output dir's parent, so the output dir must be a `data/<collection>`.
    return tmp_path / "data" / "wild"


def _args(parquet, out, **kw):
    base = dict(parquet=[str(parquet)], output_dir=out, limit_shards=None,
                models=None, include_instances=False, max_instances=None,
                retrieved_timestamp="1700000000.0", evaluation_timestamp="1780000000.0",
                revision=None, replace_existing=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_aggregates(tmp_path):
    pqt = tmp_path / "w.parquet"
    _synth_parquet(pqt)
    out = _out(tmp_path)
    n = adapter.run(_args(pqt, out))
    assert n == 2  # 2 models x 1 benchmark
    files = list(out.rglob("*.json"))
    assert len(files) == 2
    for f in files:
        report = validate_file(f)
        assert report.valid, report.errors
    log = json.loads(next((out / "openai" / "gpt-x").glob("*.json")).read_text())
    names = {r["evaluation_name"] for r in log["evaluation_results"]}
    assert names == {"wild.mmlu", "wild.mmlu.algebra", "wild.mmlu.logic"}
    overall = next(r for r in log["evaluation_results"] if r["evaluation_name"] == "wild.mmlu")
    # the registry's canonical global metric on every result; the task lives in
    # evaluation_name, so a cross-source accuracy join stays joinable
    assert {r["metric_config"]["metric_id"] for r in log["evaluation_results"]} == {"accuracy"}
    assert overall["metric_config"]["score_type"] == "continuous"
    assert (overall["metric_config"]["min_score"], overall["metric_config"]["max_score"]) == (0.0, 1.0)
    assert abs(overall["score_details"]["score"] - 2 / 3) < 1e-9
    # analytic proportion SE = sqrt(p(1-p)/n), p=2/3 over n=6 items — regression guard
    unc = overall["score_details"]["uncertainty"]
    assert abs(unc["standard_error"]["value"] - math.sqrt((2 / 3) * (1 / 3) / 6)) < 1e-9
    assert unc["num_samples"] == 6
    assert log["source_metadata"]["source_type"] == "evaluation_run"
    assert log["model_info"]["id"] == "openai/gpt-x"
    assert log["evaluation_id"] == "wild/openai_gpt-x/mmlu/1780000000.0"  # keyed on eval time
    assert log["retrieved_timestamp"] == "1700000000.0"       # record-creation time
    assert log["evaluation_timestamp"] == "1780000000.0"      # when the eval ran
    assert log["eval_library"]["name"] == "inspect_ai"
    # source_data points at the benchmark's dataset repo, not WILD-raw
    assert overall["source_data"]["source_type"] == "hf_dataset"
    assert overall["source_data"]["hf_repo"] == "cais/mmlu"


def test_single_subtask_dedup(tmp_path):
    # a task whose only subtask is "general" must emit ONLY wild.<task> (no dup leaf),
    # and instances must attach to the overall result id (task), not task::general.
    convo = json.dumps([{"role": "user", "content": "Q?"},
                        {"role": "assistant", "content": "ANSWER: C"}])
    rows = [dict(model="openai/gpt-x", task="arc_challenge", subtask="general",
                 item_id=f"i{i}", score=i % 2, input_tokens=10, output_tokens=2,
                 conversation=convo, stop_reason="stop", target="C", answer="C",
                 scores=json.dumps({"choice": {"value": "C", "answer": "ANSWER: C"}}))
            for i in range(4)]
    pqt = tmp_path / "w.parquet"
    pq.write_table(pa.Table.from_pylist(rows), str(pqt))
    out = _out(tmp_path)
    adapter.run(_args(pqt, out, include_instances=True))
    log = json.loads(next(out.rglob("*.json")).read_text())
    names = [r["evaluation_name"] for r in log["evaluation_results"]]
    assert names == ["wild.arc_challenge"]  # deduped: no wild.arc_challenge.general
    inst = json.loads(next(out.rglob("*_samples.jsonl")).read_text().splitlines()[0])
    assert inst["evaluation_result_id"] == "arc_challenge"          # FK resolves to overall
    assert inst["input"]["raw"] == "Q?"                              # answer NOT leaked in
    assert inst["output"]["raw"] == ["ANSWER: C"]                    # full generation = assistant turn
    assert inst["answer_attribution"][0]["extraction_method"] == "choice"  # real scorer
    assert "sample_hash" in inst
    # source_data for arc_challenge -> the AI2 ARC dataset
    assert log["evaluation_results"][0]["source_data"]["hf_repo"] == "allenai/ai2_arc"


def test_instances(tmp_path):
    pqt = tmp_path / "w.parquet"
    _synth_parquet(pqt)
    out = _out(tmp_path)
    adapter.run(_args(pqt, out, include_instances=True))
    samples = list(out.rglob("*_samples.jsonl"))
    assert len(samples) == 2
    for s in samples:
        report = validate_file(s)
        assert report.valid, report.errors
    # aggregate points at its sidecar
    agg = next((out / "openai" / "gpt-x").glob("*.json"))
    log = json.loads(agg.read_text())
    det = log["detailed_evaluation_results"]
    assert det["format"] == "jsonl" and det["total_rows"] == 6
    inst = json.loads(next((out / "openai" / "gpt-x").glob("*_samples.jsonl")).read_text().splitlines()[0])
    assert inst["interaction_type"] == "single_turn"
    assert inst["evaluation"]["is_correct"] in (True, False)
    assert inst["token_usage"]["total_tokens"] == inst["token_usage"]["input_tokens"] + inst["token_usage"]["output_tokens"]
    assert inst["evaluation_name"].startswith("wild.mmlu.")
    # output.raw is the model's FULL generation (assistant turn), NOT the parsed answer
    full = inst["output"]["raw"]
    assert len(full) == 1 and full[0].startswith("Step by step")
    ev = inst["answer_attribution"][0]["extracted_value"]
    assert ev in ("4", "5")
    assert ev != full[0]                              # generation != extracted answer (regression guard)
    # sample_hash uses the canonical cross-adapter recipe over (input.raw, reference)
    assert inst["sample_hash"] == adapter._sample_hash(inst["input"]["raw"], inst["input"]["reference"])
    # the sidecar link is the full repository-relative path, not the basename
    assert det["file_path"] == (
        f"data/wild/openai/gpt-x/{agg.stem}_samples.jsonl")
    assert det["checksum"] == hashlib.sha256(
        agg.with_name(f"{agg.stem}_samples.jsonl").read_bytes()).hexdigest()


def test_rerun_needs_replace_existing(tmp_path):
    # filenames are fresh uuid4s, so a second run into the same directory would add a
    # duplicate copy of every evaluation_id instead of replacing it
    pqt = tmp_path / "w.parquet"
    _synth_parquet(pqt)
    out = _out(tmp_path)
    adapter.run(_args(pqt, out, include_instances=True))
    published = sorted(p.name for p in out.rglob("*.json*"))
    with pytest.raises(SystemExit, match="--replace-existing"):
        adapter.run(_args(pqt, out, include_instances=True))
    assert sorted(p.name for p in out.rglob("*.json*")) == published  # untouched
    adapter.run(_args(pqt, out, include_instances=True, replace_existing=True))
    replaced = sorted(p.name for p in out.rglob("*.json*"))
    assert len(replaced) == len(published)          # replaced, not accumulated
    assert replaced != published                    # fresh uuids


def test_a_failed_publication_leaves_no_partial_output(tmp_path, monkeypatch):
    # a mid-batch failure must not leave half a refresh behind
    pqt = tmp_path / "w.parquet"
    _synth_parquet(pqt)
    out = _out(tmp_path)
    real = adapter.publish_evaluation_logs
    calls = []

    def flaky(*a, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return real(*a, **kw)

    monkeypatch.setattr(adapter, "publish_evaluation_logs", flaky)
    with pytest.raises(RuntimeError):
        adapter.run(_args(pqt, out, include_instances=True))
    assert not list(out.rglob("*.json*"))


def _one_task_parquet(path, model, task):
    convo = json.dumps([{"role": "user", "content": "Q?"},
                        {"role": "assistant", "content": "ANSWER: C"}])
    rows = [dict(model=model, task=task, subtask="main", item_id=f"i{i}",
                 score=i % 2, input_tokens=10, output_tokens=2, conversation=convo,
                 stop_reason="stop", target="C", answer="C",
                 scores=json.dumps({"choice": {"value": "C", "answer": "ANSWER: C"}}))
            for i in range(4)]
    pq.write_table(pa.Table.from_pylist(rows), str(path))


def test_replacement_supersedes_only_the_benchmarks_it_rewrites(tmp_path):
    # one model directory holds every benchmark that model was evaluated on, so a run
    # covering one of them must replace its own prior copy and leave the rest alone
    mmlu, arc = tmp_path / "mmlu.parquet", tmp_path / "arc.parquet"
    _one_task_parquet(mmlu, "openai/gpt-x", "mmlu")
    _one_task_parquet(arc, "openai/gpt-x", "arc_challenge")
    out = _out(tmp_path)
    adapter.run(_args(mmlu, out, include_instances=True))
    kept = {p.name for p in out.rglob("*.json*")}
    # a benchmark the target does not hold yet supersedes nothing, so it needs no flag
    adapter.run(_args(arc, out, include_instances=True))
    adapter.run(_args(arc, out, include_instances=True, replace_existing=True))
    after = {p.name for p in out.rglob("*.json*")}
    assert kept < after                      # the mmlu aggregate and sidecar survive
    assert len(after) == 2 * len(kept)       # arc replaced rather than accumulated
    assert sorted(json.loads(p.read_text())["evaluation_id"].split("/")[2]
                  for p in out.rglob("*.json")) == ["arc_challenge", "mmlu"]


def test_a_failed_replacement_leaves_the_previous_refresh_in_place(tmp_path, monkeypatch):
    # the prior records are removed only after the new ones exist, so a refresh that
    # dies mid-publication cannot leave the directory emptier than it started
    pqt = tmp_path / "w.parquet"
    _synth_parquet(pqt)
    out = _out(tmp_path)
    adapter.run(_args(pqt, out, include_instances=True))
    before = {p.name for p in out.rglob("*.json*")}
    real = adapter.publish_evaluation_logs
    calls = []

    def flaky(*a, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return real(*a, **kw)

    monkeypatch.setattr(adapter, "publish_evaluation_logs", flaky)
    with pytest.raises(RuntimeError):
        adapter.run(_args(pqt, out, include_instances=True, replace_existing=True))
    assert {p.name for p in out.rglob("*.json*")} == before


def test_output_dir_must_be_the_collection_directory(tmp_path):
    # publication derives <base>/wild/<developer>/<model> itself, so any other leaf
    # would write beside the directory asked for — and be scanned for replacement too
    pqt = tmp_path / "w.parquet"
    _synth_parquet(pqt)
    with pytest.raises(SystemExit, match="must end in 'wild'"):
        adapter.run(_args(pqt, tmp_path / "data" / "wild-v2"))
    assert not list((tmp_path / "data").rglob("*.json*"))
    assert adapter.resolve_base_output_dir(_out(tmp_path)) == tmp_path / "data"


def test_models_filter_matching_nothing_publishes_nothing(tmp_path, capsys):
    pqt = tmp_path / "w.parquet"
    _synth_parquet(pqt)
    out = _out(tmp_path)
    with pytest.raises(SystemExit, match="the source has none of them"):
        adapter.run(_args(pqt, out, models=["openai/gpt-y"]))
    assert not list(out.rglob("*.json*"))
    # a partly-matching selection is a warning, not an error: what matched is real
    adapter.run(_args(pqt, out, models=["openai/gpt-x", "openai/gpt-y"]))
    assert len(list(out.rglob("*.json"))) == 1
    assert "no source rows for 1 selected model(s): openai/gpt-y" in capsys.readouterr().out


def test_bare_models_flag_is_an_error(monkeypatch):
    # nargs='+', so `--models` with nothing after it cannot parse to [] and then
    # convert every model in the source
    monkeypatch.setattr(sys, "argv", ["adapter", "--models"])
    with pytest.raises(SystemExit):
        adapter.parse_args()


def test_symbolic_revision_cannot_stand_in_for_a_failed_pin(monkeypatch, capsys):
    # the lookup is what turns 'main' into a commit; if it fails, 'main' still moves
    # between the aggregate pass and the instance pass, so it is not a pin
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(SystemExit, match="40-character commit SHA"):
        adapter.resolve_source_revision("main", None)
    with pytest.raises(SystemExit, match="--revision"):
        adapter.resolve_source_revision(None, None)
    sha = "a" * 40
    # a SHA is already the pin; only the commit date is lost, and without it
    # resolve_eval_timestamp demands --evaluation-timestamp rather than guessing
    assert adapter.resolve_source_revision(sha, None) == (sha, None)
    assert "no commit date" in capsys.readouterr().out


def _unusable_score_parquet(path):
    convo = json.dumps([{"role": "user", "content": "Q?"},
                        {"role": "assistant", "content": "A"}])
    rows = [dict(model="openai/gpt-x", task="gsm8k", subtask="main",
                 item_id=f"i{i}", score=score, input_tokens=in_tok,
                 output_tokens=out_tok, conversation=convo, stop_reason="stop",
                 target="4", answer="4",
                 scores=json.dumps({"match": {"value": "C", "answer": "4"}}))
            for i, (score, in_tok, out_tok) in enumerate(
                [(1, 10, 2), (0, 20, 4), (None, 30, 6), (1, None, 8)])]
    pq.write_table(pa.Table.from_pylist(rows), str(path))


def test_unusable_score_is_reported_not_counted(tmp_path):
    # a missing score is not a wrong answer: it must leave the denominator, be named
    # in the failure report, skip the sidecar, and make the run exit non-zero
    pqt = tmp_path / "w.parquet"
    _unusable_score_parquet(pqt)
    out = _out(tmp_path)
    with pytest.raises(SourceRecordsError):
        adapter.run(_args(pqt, out, include_instances=True))
    log = json.loads(next(out.rglob("*.json")).read_text())
    overall = log["evaluation_results"][0]
    assert overall["score_details"]["uncertainty"]["num_samples"] == 3   # not 4
    assert abs(overall["score_details"]["score"] - 2 / 3) < 1e-9
    report = json.loads(
        adapter.default_failure_report_path(out).read_text())
    assert report["total_source_records"] == 4
    assert len(report["failed_records"]) == 1
    assert report["failed_records"][0]["source_ref"].endswith("gpt-x/gsm8k/i2")
    lines = next(out.rglob("*_samples.jsonl")).read_text().splitlines()
    assert [json.loads(line)["sample_id"] for line in lines] == ["i0", "i1", "i3"]
    assert log["detailed_evaluation_results"]["total_rows"] == 3


def test_incomplete_token_usage_is_omitted_not_zeroed(tmp_path):
    pqt = tmp_path / "w.parquet"
    _unusable_score_parquet(pqt)
    out = _out(tmp_path)
    with pytest.raises(SourceRecordsError):
        adapter.run(_args(pqt, out, include_instances=True))
    details = json.loads(next(out.rglob("*.json")).read_text())[
        "evaluation_results"][0]["metric_config"]["additional_details"]
    # i3 carries no input_tokens, so it is out of the mean rather than a zero in it
    assert details["n_items_with_token_usage"] == "2"
    assert details["mean_input_tokens"] == "15.0"
    rows = {json.loads(line)["sample_id"]: json.loads(line)
            for line in next(out.rglob("*_samples.jsonl")).read_text().splitlines()}
    assert "token_usage" not in rows["i3"]
    assert rows["i0"]["token_usage"]["total_tokens"] == 12


def test_iter_batches_bounds_rows_per_read(tmp_path):
    # the cap must bound the allocation too: a WILD shard is one 500k-row row group,
    # so reads are batched rather than per row group
    pqt = tmp_path / "w.parquet"
    _synth_parquet(pqt)
    sizes = [len(batch["model"]) for _, batch in adapter.iter_batches(
        [str(pqt)], ["model"], batch_size=4)]
    assert sizes == [4, 4, 4]


def test_missing_evaluation_timestamp_is_an_error():
    # evaluation_id is keyed on it, so a now() fallback would give identical reruns
    # different logical identities
    with pytest.raises(SystemExit, match="--evaluation-timestamp"):
        adapter.resolve_eval_timestamp(None, None)
    assert adapter.resolve_eval_timestamp(None, "1780000000.0") == "1780000000.0"


def test_sample_hash_is_canonical():
    # locks the recipe to the skill's templates/instance_sidecar._sample_hash
    expected = hashlib.sha256(
        json.dumps({"raw": "Q?", "reference": ["C"]}, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest()
    assert adapter._sample_hash("Q?", ["C"]) == expected


def test_split_conversation_separates_prompt_and_generation():
    convo = json.dumps([{"role": "system", "content": "sys"},
                        {"role": "user", "content": "Q?"},
                        {"role": "assistant", "content": "the full model answer"}])
    prompt, generation = adapter._split_conversation(convo)
    assert prompt == "sys\n\nQ?"                       # user + system only, no assistant
    assert generation == ["the full model answer"]     # assistant turn -> output.raw


def test_local_run_provenance_no_false_revision(tmp_path):
    # a local --parquet run must NOT stamp dataset_revision='main' (false remote provenance)
    pqt = tmp_path / "w.parquet"
    _synth_parquet(pqt)
    out = _out(tmp_path)
    adapter.run(_args(pqt, out))
    log = json.loads(next((out / "openai" / "gpt-x").glob("*.json")).read_text())
    ad = log["source_metadata"]["additional_details"]
    assert "dataset_revision" not in ad                # unknown for a local file
    assert "local" in ad.get("dataset_source", "").lower()
