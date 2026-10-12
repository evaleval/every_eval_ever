"""
Script to convert HuggingFace Open LLM Leaderboard v2 data to the EvalEval schema format.

Data sources, both pinned Hugging Face datasets:
- ``open-llm-leaderboard/contents`` at ``CONTENTS_REVISION``: the leaderboard
  table (scores, one row per model and precision). Reading it needs pyarrow.
- ``open-llm-leaderboard/results`` at ``RESULTS_REVISION``: the harness result
  dumps (harness settings, sample counts, split)

``--source-api`` reads the table from the leaderboard Space API instead, which
is unpinned and no longer served.

Usage:
    uv run python -m every_eval_ever.adapters.hfopenllm_v2.adapter
    uv run python -m every_eval_ever.adapters.hfopenllm_v2.adapter \
        --output-dir /tmp/smoke/data/hfopenllm_v2 \
        --contents-parquet <contents snapshot>/data/train-00000-of-00001.parquet \
        --results-dir <results snapshot>
"""

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from every_eval_ever.adapters.hfopenllm_v2.contents import (
    CONTENTS_REPO,
    CONTENTS_REVISION,
    download_contents,
    load_contents,
)
from every_eval_ever.adapters.hfopenllm_v2.dump_index import (
    RESULTS_PATTERN,
    RESULTS_REPO,
    RESULTS_REVISION,
    DumpIndex,
    HarnessGroup,
    ModelEvidence,
    build_dump_index,
)
from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationArgs,
    GenerationConfig,
    MetricConfig,
    ScoreDetails,
    ScoreType,
    SourceDataHf,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordFailure,
    default_failure_report_path,
    fetch_json,
    make_model_info,
    make_source_metadata,
    raw_capture,
    save_evaluation_logs,
    save_failure_report,
)

# Source URL
SOURCE_URL = 'https://open-llm-leaderboard-open-llm-leaderboard.hf.space/api/leaderboard/formatted'
OUTPUT_DIR = 'data/hfopenllm_v2'

# Evaluation name mapping from API keys to display names
EVALUATION_MAPPING = {
    'ifeval': 'IFEval',
    'bbh': 'BBH',
    'math': 'MATH Level 5',
    'gpqa': 'GPQA',
    'musr': 'MUSR',
    'mmlu_pro': 'MMLU-PRO',
}


# Evaluation descriptions
EVALUATION_DESCRIPTIONS = {
    'IFEval': 'Accuracy on IFEval',
    'BBH': 'Accuracy on BBH',
    'MATH Level 5': 'Exact Match on MATH Level 5',
    'GPQA': 'Accuracy on GPQA',
    'MUSR': 'Accuracy on MUSR',
    'MMLU-PRO': 'Accuracy on MMLU-PRO',
}

# Source data mapping: eval_key -> SourceDataHf
SOURCE_DATA_MAPPING = {
    'ifeval': SourceDataHf(
        dataset_name='IFEval',
        source_type='hf_dataset',
        hf_repo='google/IFEval',
    ),
    'bbh': SourceDataHf(
        dataset_name='BBH',
        source_type='hf_dataset',
        hf_repo='SaylorTwift/bbh',
    ),
    'math': SourceDataHf(
        dataset_name='MATH Level 5',
        source_type='hf_dataset',
        hf_repo='DigitalLearningGmbH/MATH-lighteval',
    ),
    'gpqa': SourceDataHf(
        dataset_name='GPQA',
        source_type='hf_dataset',
        hf_repo='Idavidrein/gpqa',
    ),
    'musr': SourceDataHf(
        dataset_name='MUSR',
        source_type='hf_dataset',
        hf_repo='TAUR-Lab/MuSR',
    ),
    'mmlu_pro': SourceDataHf(
        dataset_name='MMLU-PRO',
        source_type='hf_dataset',
        hf_repo='TIGER-Lab/MMLU-Pro',
    ),
}


# eval_key -> (metric_id, metric_name, metric_kind, metric_unit)
METRIC_MAPPING = {
    'ifeval': ('accuracy', 'Accuracy', 'accuracy', 'proportion'),
    'bbh': ('accuracy', 'Accuracy', 'accuracy', 'proportion'),
    'math': ('exact_match', 'Exact Match', 'exact_match', 'proportion'),
    'gpqa': ('accuracy', 'Accuracy', 'accuracy', 'proportion'),
    'musr': ('accuracy', 'Accuracy', 'accuracy', 'proportion'),
    'mmlu_pro': ('accuracy', 'Accuracy', 'accuracy', 'proportion'),
}

HARNESS_FORK = 'https://github.com/huggingface/lm-evaluation-harness/tree/adding_all_changess'


def _slug(name: str) -> str:
    return '_'.join(
        ''.join(c if c.isalnum() else ' ' for c in name.lower()).split()
    )


def _source_data(
    base: SourceDataHf, group: HarnessGroup | None
) -> SourceDataHf:
    """``base`` with the split and counts the dumps agree on.

    The dataset the harness loaded is recorded as ``harness_dataset_path``
    where it differs from ``base.hf_repo``.
    """
    if group is None:
        return base
    update: Dict[str, Any] = {
        'hf_split': group.hf_split,
        'samples_number': group.samples_number,
    }
    details = {}
    if group.dataset_path is not None and group.dataset_path != base.hf_repo:
        details['harness_dataset_path'] = group.dataset_path
    if group.dataset_size is not None:
        details['dataset_size'] = str(group.dataset_size)
    if details:
        update['additional_details'] = details
    return base.model_copy(update=update)


def _generation_config(group: HarnessGroup | None) -> GenerationConfig | None:
    """The harness settings the dumps agree on, if any."""
    if group is None:
        return None
    details = {
        name: value if isinstance(value, str) else json.dumps(value)
        for name, value in (
            ('output_type', group.output_type),
            ('num_fewshot', group.num_fewshot),
            ('do_sample', group.do_sample),
        )
        if value is not None
    }
    args = None
    if group.temperature is not None or group.max_tokens is not None:
        args = GenerationArgs(
            temperature=group.temperature, max_tokens=group.max_tokens
        )
    if args is None and not details:
        return None
    return GenerationConfig(
        generation_args=args, additional_details=details or None
    )


def convert_model(
    model_data: Dict[str, Any],
    retrieved_timestamp: str,
    *,
    source_ref: str | None = None,
    failures: list[SourceRecordFailure] | None = None,
    evidence: ModelEvidence | None = None,
) -> EvaluationLog:
    """Convert one model, optionally retaining unusable metric provenance.

    The strict public behavior is unchanged when ``failures`` is omitted:
    any unusable metric rejects the model. Batch conversion supplies a failure
    list so valid metrics from the same model can still be published.
    ``evidence`` is the dump evidence for this model and precision; without
    it the harness fields are left out.
    """
    model_id = model_data['model']['name']
    if '/' not in model_id:
        raise ValueError(f"Expected 'org/model' format, got: {model_id}")
    developer, model_name = model_id.split('/', 1)
    precision = model_data['model'].get('precision')
    if not isinstance(precision, str) or not precision.strip():
        raise ValueError(f'precision is missing; got {precision!r}')
    evaluation_id = (
        f'hfopenllm_v2/{developer}_{model_name}/{precision}/'
        f'{retrieved_timestamp}'
    )
    groups = evidence.groups if evidence is not None else {}

    # Build evaluation results
    eval_results: List[EvaluationResult] = []
    for eval_key, eval_data in model_data.get('evaluations', {}).items():
        try:
            if eval_data.get('value') is None:
                raise ValueError('score is missing')
            display_name = eval_data.get(
                'name', EVALUATION_MAPPING.get(eval_key, eval_key)
            )
            description = EVALUATION_DESCRIPTIONS.get(
                display_name, f'Accuracy on {display_name}'
            )
            source_data = SOURCE_DATA_MAPPING.get(eval_key)
            if source_data is None or eval_key not in METRIC_MAPPING:
                raise ValueError(
                    f"unknown evaluation key; add '{eval_key}' to "
                    'SOURCE_DATA_MAPPING and METRIC_MAPPING'
                )
            metric_id, metric_name, metric_kind, metric_unit = METRIC_MAPPING[
                eval_key
            ]
            group = groups.get(eval_key)

            eval_results.append(
                EvaluationResult(
                    evaluation_result_id=(
                        f'{evaluation_id}#{_slug(display_name)}#{metric_id}'
                    ),
                    evaluation_name=display_name,
                    source_data=_source_data(source_data, group),
                    metric_config=MetricConfig(
                        evaluation_description=description,
                        lower_is_better=False,
                        score_type=ScoreType.continuous,
                        min_score=0.0,
                        max_score=1.0,
                        metric_id=metric_id,
                        metric_name=metric_name,
                        metric_kind=metric_kind,
                        metric_unit=metric_unit,
                    ),
                    score_details=ScoreDetails(
                        score=round(float(eval_data['value']), 4),
                    ),
                    generation_config=_generation_config(group),
                )
            )
        except Exception as exc:
            if failures is None:
                raise ValueError(
                    f"Evaluation '{eval_key}' could not be converted: {exc}"
                ) from exc
            failures.append(
                SourceRecordFailure(
                    source_ref=(
                        f'{source_ref or model_id} evaluation {eval_key!r}'
                    ),
                    reason=str(exc),
                    source_record={
                        'model': model_data.get('model'),
                        'evaluation_key': eval_key,
                        'evaluation': eval_data,
                    },
                )
            )
    if not eval_results:
        raise ValueError('model has no usable evaluation results')

    # Build additional details
    additional_details = {}
    if 'precision' in model_data['model']:
        additional_details['precision'] = str(model_data['model']['precision'])
    if 'architecture' in model_data['model']:
        additional_details['architecture'] = str(
            model_data['model']['architecture']
        )
    if 'params_billions' in model_data.get('metadata', {}):
        additional_details['params_billions'] = str(
            model_data['metadata']['params_billions']
        )

    # Build model info
    model_info = make_model_info(
        model_name=model_name,
        developer=developer,
        inference_platform='unknown',
        additional_details=additional_details if additional_details else None,
    )

    library_details = {'fork': HARNESS_FORK}
    if evidence is not None:
        if evidence.git_hash is not None:
            library_details['git_hash'] = evidence.git_hash
        if evidence.transformers_version is not None:
            library_details['transformers_version'] = (
                evidence.transformers_version
            )

    return EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=evaluation_id,
        retrieved_timestamp=retrieved_timestamp,
        source_metadata=make_source_metadata(
            source_name='HF Open LLM v2',
            organization_name='Hugging Face',
            evaluator_relationship=EvaluatorRelationship.third_party,
        ),
        eval_library=EvalLibrary(
            name='lm-evaluation-harness',
            version='0.4.0',
            additional_details=library_details,
        ),
        model_info=model_info,
        evaluation_results=eval_results,
    )


def _lookup(
    dump_index: DumpIndex | None, model_data: Dict[str, Any]
) -> ModelEvidence | None:
    if dump_index is None:
        return None
    model = model_data.get('model') or {}
    return dump_index.lookup(model.get('name'), model.get('precision'))


def convert_models(
    models_data: List[Dict[str, Any]],
    retrieved_timestamp: str | None = None,
    dump_index: DumpIndex | None = None,
) -> SourceConversionResult[EvaluationLogOutput]:
    """Convert all usable models and preserve every rejected source row.

    Each row is joined to ``dump_index`` on its exact (model, precision).
    """
    timestamp = retrieved_timestamp or str(time.time())
    outputs = []
    failures: list[SourceRecordFailure] = []
    for index, model_data in enumerate(models_data):
        source_ref = f'model row {index}'
        failure_count_before = len(failures)
        try:
            model_id = model_data['model']['name']
            if '/' not in model_id:
                raise ValueError(
                    f"Expected 'org/model' format, got: {model_id}"
                )
            developer, model = model_id.split('/', 1)
            eval_log = convert_model(
                model_data,
                timestamp,
                source_ref=source_ref,
                failures=failures,
                evidence=_lookup(dump_index, model_data),
            )
            outputs.append(
                EvaluationLogOutput(
                    eval_log=eval_log,
                    base_dir=OUTPUT_DIR,
                    developer=developer,
                    model_name=model,
                )
            )
        except Exception as exc:
            # If every metric was already recorded as unusable, add one
            # model-level entry explaining why no evaluation file was emitted.
            if len(failures) > failure_count_before:
                reason = f'no output written: {exc}'
            else:
                reason = str(exc)
            failures.append(
                SourceRecordFailure(
                    source_ref=source_ref,
                    reason=reason,
                    source_record=model_data,
                )
            )
    return SourceConversionResult(
        source_name='HF Open LLM v2',
        total_records=len(models_data),
        records=outputs,
        failures=failures,
    )


def harness_gaps(
    models_data: List[Dict[str, Any]], dump_index: DumpIndex | None
) -> Dict[str, Any]:
    """Every harness field left null, with the reason, per model row."""
    unfilled = []
    for index, model_data in enumerate(models_data):
        model = model_data.get('model') or {}
        evidence = _lookup(dump_index, model_data)
        for eval_key in model_data.get('evaluations') or {}:
            if evidence is None:
                gaps = {'all': 'no_dump'}
            elif eval_key not in evidence.groups:
                gaps = {'all': 'task_group_not_run'}
            else:
                gaps = evidence.groups[eval_key].gaps
            for field_name, reason in sorted(gaps.items()):
                unfilled.append(
                    {
                        'source_ref': f'model row {index}',
                        'model': model.get('name'),
                        'precision': model.get('precision'),
                        'evaluation_key': eval_key,
                        'field': field_name,
                        'reason': reason,
                    }
                )
    counts = Counter(
        f'{entry["evaluation_key"]}.{entry["field"]}: {entry["reason"]}'
        for entry in unfilled
    )
    return {
        'results_dataset': RESULTS_REPO,
        'results_revision': (
            dump_index.revision if dump_index else RESULTS_REVISION
        ),
        'results_dir_unpinned': bool(dump_index and dump_index.unpinned),
        'dumps_read': dump_index.files_read if dump_index else 0,
        'dumps_rejected': dump_index.rejected if dump_index else [],
        'unfilled_counts': dict(sorted(counts.items())),
        'unfilled': unfilled,
    }


def process_models(
    models_data: List[Dict[str, Any]],
    output_dir: str | Path = OUTPUT_DIR,
    dump_index: DumpIndex | None = None,
    leaderboard_input: Dict[str, Any] | None = None,
) -> int:
    """Save valid models, report rejected rows and unfilled harness fields.

    The report is written whenever a row failed or a harness field stayed
    null; only failed rows signal incompleteness.
    """
    result = convert_models(models_data, dump_index=dump_index)
    outputs = [
        EvaluationLogOutput(
            eval_log=record.eval_log,
            base_dir=output_dir,
            developer=record.developer,
            model_name=record.model_name,
        )
        for record in result.records
    ]
    paths = save_evaluation_logs(outputs)
    for path in paths:
        print(f'Saved: {path}')
    gaps = harness_gaps(models_data, dump_index)
    if result.failures or gaps['unfilled'] or gaps['dumps_rejected']:
        report_path = save_failure_report(
            result,
            default_failure_report_path(output_dir),
        )
        report = json.loads(report_path.read_text(encoding='utf-8'))
        report['harness_evidence'] = gaps
        if leaderboard_input is not None:
            report['leaderboard_input'] = leaderboard_input
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
            + '\n',
            encoding='utf-8',
        )
        print(f'Failure report: {report_path}')
        result.raise_if_incomplete()
    return len(paths)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Convert the HF Open LLM Leaderboard v2 API to EEE records.'
        )
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(OUTPUT_DIR),
        help=f'Datastore collection directory (default: {OUTPUT_DIR}).',
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=None,
        help=(
            f'Local Hub snapshot of {RESULTS_REPO} (a directory named after '
            f'its commit sha, which must be {RESULTS_REVISION}). Default: '
            'download its results_*.json files once from the Hub.'
        ),
    )
    parser.add_argument(
        '--allow-unpinned-results-dir',
        action='store_true',
        help=(
            '(Debug) Accept a --results-dir at another commit, or with no '
            'commit in its path; the revision is recorded and reported as is.'
        ),
    )
    table = parser.add_mutually_exclusive_group()
    table.add_argument(
        '--contents-parquet',
        type=Path,
        default=None,
        help=(
            f'Local copy of {CONTENTS_REPO} data/train-00000-of-00001.parquet '
            'inside a Hub snapshot directory named after its commit sha, '
            f'which must be {CONTENTS_REVISION}. Default: download it once '
            'from the Hub.'
        ),
    )
    table.add_argument(
        '--source-api',
        action='store_true',
        help=(
            '(Legacy) Read the table from the leaderboard Space API, which '
            'is unpinned and no longer served, instead of the contents '
            'dataset.'
        ),
    )
    parser.add_argument(
        '--allow-unpinned-contents-parquet',
        action='store_true',
        help=(
            '(Debug) Accept a --contents-parquet at another commit, or with '
            'no commit in its path; the revision is recorded and reported '
            'as is.'
        ),
    )
    return parser.parse_args(argv)


_SHA = re.compile(r'[0-9a-f]{40}')


def _snapshot_revision(
    snapshot_dir: Path,
    *,
    option: str,
    repo: str,
    pinned: str,
    allow_unpinned: bool,
) -> tuple[str, bool]:
    name = snapshot_dir.name
    sha = name if _SHA.fullmatch(name) else None
    if sha == pinned:
        return sha, False
    if not allow_unpinned:
        found = sha or f'no commit sha (directory {name!r})'
        raise ValueError(
            f'{option} is at {found}, but this adapter is pinned to '
            f'{repo}@{pinned}. Pass a snapshot at that revision, or '
            f'--allow-unpinned-{option[2:]}.'
        )
    return sha or 'local', True


def results_dir_revision(
    results_dir: str | Path, *, allow_unpinned: bool = False
) -> tuple[str, bool]:
    """The commit a local results snapshot is at, and whether it is unpinned.

    The commit is the directory's own name, as in the Hub cache layout
    ``snapshots/<sha>``. A directory at another commit, or without one, is
    an error unless ``allow_unpinned``; it is then reported as that sha, or
    ``'local'``.
    """
    return _snapshot_revision(
        Path(results_dir).resolve(),
        option='--results-dir',
        repo=RESULTS_REPO,
        pinned=RESULTS_REVISION,
        allow_unpinned=allow_unpinned,
    )


def contents_parquet_revision(
    parquet: str | Path, *, allow_unpinned: bool = False
) -> tuple[str, bool]:
    """The commit a local contents parquet is at, and whether it is unpinned.

    The commit is the name of the snapshot directory holding ``data/``, as
    in the Hub cache layout ``snapshots/<sha>/data/<file>``; the file itself
    may be a symlink into the cache's blobs. Otherwise as
    ``results_dir_revision``.
    """
    return _snapshot_revision(
        Path(parquet).absolute().parent.parent,
        option='--contents-parquet',
        repo=CONTENTS_REPO,
        pinned=CONTENTS_REVISION,
        allow_unpinned=allow_unpinned,
    )


def download_results() -> Path:
    """Fetch the pinned results snapshot's dumps in one Hub download."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=RESULTS_REPO,
            repo_type='dataset',
            revision=RESULTS_REVISION,
            allow_patterns=RESULTS_PATTERN,
        )
    )


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    revision, unpinned = RESULTS_REVISION, False
    contents_revision, contents_unpinned = CONTENTS_REVISION, False
    try:
        if args.results_dir is not None:
            revision, unpinned = results_dir_revision(
                args.results_dir,
                allow_unpinned=args.allow_unpinned_results_dir,
            )
        if args.contents_parquet is not None:
            contents_revision, contents_unpinned = contents_parquet_revision(
                args.contents_parquet,
                allow_unpinned=args.allow_unpinned_contents_parquet,
            )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.source_api:
        print(f'Fetching data from {SOURCE_URL}...')
        all_models = fetch_json(SOURCE_URL)
        leaderboard_input = {'source_url': SOURCE_URL}
    else:
        raw_capture.record_hf_dataset(
            CONTENTS_REPO,
            revision=contents_revision,
            label='Open LLM Leaderboard v2 table',
        )
        parquet = args.contents_parquet or download_contents()
        print(f'Reading {parquet} ({contents_revision})...')
        all_models = load_contents(parquet)
        leaderboard_input = {
            'dataset': CONTENTS_REPO,
            'revision': contents_revision,
            'unpinned': contents_unpinned,
        }

    raw_capture.record_hf_dataset(
        RESULTS_REPO,
        revision=revision,
        label='Open LLM Leaderboard v2 harness result dumps',
    )
    results_dir = args.results_dir or download_results()
    print(f'Indexing harness dumps in {results_dir} ({revision})...')
    dump_index = build_dump_index(results_dir)
    dump_index.revision, dump_index.unpinned = revision, unpinned
    print(
        f'Read {dump_index.files_read} dumps for {len(dump_index.entries)} '
        f'(model, precision) pairs; rejected {len(dump_index.rejected)}.'
    )

    print(f'Processing {len(all_models)} models...')
    count = process_models(
        all_models, args.output_dir, dump_index, leaderboard_input
    )
    print(f'Done! Processed {count} models.')
    return count


if __name__ == '__main__':
    main()
