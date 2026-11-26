#!/usr/bin/env python3
"""Convert typed-decisions evaluation runs into Every Eval Ever records.

typed-decisions (https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
evaluates one case per request: the model receives one piece of unstructured
state and answers five typed questions about it — ``choice`` (pick among the
criteria keys), ``noul`` (true/false by probability), and ``score`` (a level
with a probability distribution). Every answer is a distribution, not a single
label, so one run over the 400-case test split yields 2,000 decisions.

The adapter consumes the two JSONL files a run produces:

* ``--rows``  — the rows the run drew cases from (one case per line with
  ``state``, ``questions``, and ``gold``; those three may be objects or
  inline JSON strings).
* ``--preds`` — the run's predictions (the same case ids plus ``pred``,
  the model's full typed answers with probabilities, and ``latency_ms``).

Correctness follows the benchmark's official scorer semantics exactly:

* ``choice`` — the predicted label must equal the gold label string;
* ``noul``   — the predicted ``noul`` probability at or above 0.5 means
  ``true``, compared with the gold ``true``/``false`` label;
* ``score``  — the predicted level is the argmax of the level probabilities
  (the rounded continuous score is only a fallback when every level
  probability is zero), compared with the gold level.

Run:
    uv run python -m every_eval_ever.adapters.typed_decisions.adapter \\
        --rows exp/typed_decisions_test.jsonl \\
        --preds runs/ft_n13_r4_en400_preds.preds.jsonl \\
        --output-dir /tmp/eee-typed-decisions/data/typed-decisions \\
        --include-instances

The input layout, the identity rules, and every flag are documented in
``every_eval_ever/adapters/README.md`` under "Typed Decisions".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from every_eval_ever.converters import SCHEMA_VERSION
from every_eval_ever.converters.common.publication import (
    publish_evaluation_logs,
)
from every_eval_ever.eval_types import (
    DetailedEvaluationResults,
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    Format,
    HashAlgorithm,
    InferenceEngine,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataHf,
    SourceMetadata,
    SourceType,
)
from every_eval_ever.helpers.io import (
    SourceConversionResult,
    SourceRecordFailure,
    datastore_output_dir,
    datastore_repo_file_path,
    default_failure_report_path,
    save_failure_report,
)
from every_eval_ever.instance_level_types import (
    AnswerAttributionItem,
    Evaluation,
    Input,
    InstanceLevelEvaluationLog,
    InteractionType,
    Output,
    Performance,
)

COLLECTION = 'typed-decisions'
MODEL_ID = 'Phocinae/Phocinae-Largha-150M-v1'
MODEL_NAME = 'Phocinae-Largha-150M-v1'
MODEL_DEVELOPER = 'Phocinae'
MODEL_ORG_URL = 'https://huggingface.co/Phocinae'
EVAL_LIBRARY_NAME = 'laya'
EVAL_LIBRARY_VERSION = '0.3.20'
DEFAULT_OUTPUT_DIR = Path(f'data/{COLLECTION}')
EXTRACTION_METHOD = 'custom_typed_interface'

#: Per-variant identity. ``en`` is the canonical English test split; ``zh``
#: is the machine-translated release set (gold labels byte-identical to the
#: English source) published at ``Phocinae/phocinae-zh400``.
VARIANTS: dict[str, dict[str, str]] = {
    'en': {
        'evaluation_name': 'typed-decisions-en',
        'dataset_repo': 'LocalLLaMA/typed-decisions',
        'dataset_name': 'LocalLLaMA/typed-decisions',
        'dataset_note': (
            'English test split of the typed-decisions benchmark.'
        ),
    },
    'zh': {
        'evaluation_name': 'typed-decisions-zh',
        'dataset_repo': 'Phocinae/phocinae-zh400',
        'dataset_name': 'Phocinae/phocinae-zh400',
        'dataset_note': (
            'Machine-translated (API, no human post-edit) Chinese release '
            'set; gold labels are byte-identical to the English test split.'
        ),
    },
}

DECISION_RULES = {
    'choice': 'extracted label must equal the gold label string',
    'noul': (
        'predicted probability >= 0.5 means true (official scorer threshold)'
    ),
    'score': (
        'level is the argmax of the level probabilities '
        '(official scorer; rounded score only if all probabilities are 0)'
    ),
}


@dataclass
class _Decision:
    """One scored typed decision, from one case and question."""

    case: dict[str, Any]
    qid: str
    question_type: str
    pred_entry: dict[str, Any]
    extracted: str
    reference: str
    is_correct: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--rows',
        type=Path,
        required=True,
        help='rows JSONL the run drew cases from (case state/questions/gold)',
    )
    parser.add_argument(
        '--preds',
        type=Path,
        required=True,
        help='predictions JSONL the run produced (pred/latency_ms per case)',
    )
    parser.add_argument(
        '--variant',
        choices=sorted(VARIANTS),
        default='en',
        help=(
            'dataset variant that names the evaluation '
            f'(default en, i.e. {VARIANTS["en"]["evaluation_name"]})'
        ),
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f'datastore collection directory (default: {DEFAULT_OUTPUT_DIR})',
    )
    parser.add_argument(
        '--include-instances',
        action='store_true',
        help='also write the per-decision *_samples.jsonl sidecar',
    )
    parser.add_argument(
        '--limit-cases',
        type=int,
        help='convert only the first N cases (smoke runs)',
    )
    parser.add_argument(
        '--evaluation-timestamp',
        help=(
            'when the evaluation ran, as Unix epoch seconds or an ISO-8601 '
            'timestamp (default: the --preds file modification time)'
        ),
    )
    parser.add_argument(
        '--retrieved-timestamp',
        help='override the record-creation timestamp (epoch seconds)',
    )
    return parser


def _epoch_from_value(value: str, flag: str) -> float:
    try:
        return float(value)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(
            f'{flag}: expected Unix epoch seconds or an ISO-8601 '
            f'timestamp, got {value!r}'
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _timestamp_string(value: str) -> str:
    try:
        return f'{float(value):.3f}'
    except ValueError:
        return value


def _iso_utc(epoch: float) -> str:
    moment = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return moment.strftime('%Y-%m-%dT%H:%M:%SZ')


def _load_jsonl(path: Path) -> list[Any]:
    records: list[Any] = []
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f'{path}: line {line_number} is not valid JSON: {exc}'
                ) from exc
    return records


def _normalize_case(
    record: Any,
    path: Path,
    line_number: int,
    *,
    require_questions_and_gold: bool,
) -> dict[str, Any]:
    """Return a case with state/questions/gold available as objects.

    The rows and predictions files both carry those fields, but one may hold
    them as objects while the other holds inline JSON strings; the adapter
    accepts either form. Rows must carry ``questions`` and ``gold``;
    predictions rows may omit them (only ``pred`` is needed from those).
    """
    if not isinstance(record, dict):
        raise SystemExit(f'{path}: line {line_number} is not a JSON object')
    case = dict(record)
    for key in ('state', 'questions', 'gold'):
        value = case.get(key)
        if isinstance(value, str):
            try:
                case[key] = json.loads(value)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f'{path}: line {line_number}: {key} is an inline JSON '
                    f'string that does not parse: {exc}'
                ) from exc
    for key in ('questions', 'gold'):
        if not require_questions_and_gold and case.get(key) is None:
            continue
        if not isinstance(case.get(key), dict):
            raise SystemExit(
                f'{path}: line {line_number}: {key} must be an object '
                '(or an inline JSON string encoding one)'
            )
    if not isinstance(case.get('id'), str) or not case['id']:
        raise SystemExit(f'{path}: line {line_number}: case has no usable id')
    return case


def _load_cases(
    path: Path, *, require_questions_and_gold: bool
) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for line_number, record in enumerate(_load_jsonl(path), start=1):
        case = _normalize_case(
            record,
            path,
            line_number,
            require_questions_and_gold=require_questions_and_gold,
        )
        case_id = case['id']
        if case_id in cases:
            raise SystemExit(f'{path}: duplicate case id {case_id!r}')
        cases[case_id] = case
    return cases


def _choice_options(question: dict[str, Any]) -> list[str] | None:
    criteria = question.get('criteria')
    if isinstance(criteria, dict) and criteria:
        return [str(key) for key in criteria]
    if question.get('type') == 'noul':
        return ['false', 'true']
    return None


def _raw_input_text(
    case: dict[str, Any], qid: str, question: dict[str, Any]
) -> str:
    payload = {
        'state': case.get('state'),
        'question': {
            'id': qid,
            'type': question.get('type'),
            'criteria': question.get('criteria'),
        },
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    )


def _choice_answer(
    pred_entry: dict[str, Any], gold_entry: dict[str, Any]
) -> tuple[str, str]:
    extracted = str(pred_entry['choice'])
    if 'label' in gold_entry:
        reference = str(gold_entry['label'])
    else:
        # Soft-label fallback, as in the official scorer: derive the
        # reference from the highest-probability gold candidate.
        probabilities = gold_entry.get('probabilities', {})
        reference = (
            max(probabilities, key=probabilities.get) if probabilities else ''
        )
    return extracted, reference


def _noul_answer(
    pred_entry: dict[str, Any], gold_entry: dict[str, Any]
) -> tuple[str, str]:
    probability = float(pred_entry['noul'])
    extracted = 'true' if probability >= 0.5 else 'false'
    if 'label' in gold_entry:
        reference = str(gold_entry['label']).lower()
    else:
        gold_probability = gold_entry.get(
            'noul',
            gold_entry.get('probabilities', {}).get('true', 0.5),
        )
        reference = 'true' if float(gold_probability) >= 0.5 else 'false'
    return extracted, reference


def _score_answer(
    pred_entry: dict[str, Any],
    gold_entry: dict[str, Any],
    n_levels: int,
) -> tuple[str, str]:
    score = float(pred_entry.get('score', 0.0))
    probabilities = [
        float(pred_entry.get('probabilities', {}).get(str(i), 0.0) or 0.0)
        for i in range(n_levels)
    ]
    if sum(probabilities) > 0:
        # Ties resolve to the lowest level, exactly like np.argmax.
        level = probabilities.index(max(probabilities))
    else:
        level = int(round(score))
    if gold_entry.get('label') is not None:
        reference_level = int(float(gold_entry['label']))
    else:
        reference_level = int(round(float(gold_entry.get('score', 0.0))))
    return str(level), str(reference_level)


def _decide(
    question: dict[str, Any],
    gold_entry: dict[str, Any],
    pred_entry: dict[str, Any],
) -> tuple[str, str, bool]:
    question_type = question.get('type')
    if question_type == 'choice':
        extracted, reference = _choice_answer(pred_entry, gold_entry)
    elif question_type == 'noul':
        extracted, reference = _noul_answer(pred_entry, gold_entry)
    elif question_type == 'score':
        criteria = question.get('criteria')
        n_levels = len(criteria) if isinstance(criteria, list) else 0
        extracted, reference = _score_answer(pred_entry, gold_entry, n_levels)
    else:
        raise ValueError(f'unknown question type {question_type!r}')
    return extracted, reference, extracted == reference


def sample_hash(raw_input: str, references: list[str]) -> str:
    """Hash one instance the one canonical way (same recipe as openeval)."""
    payload = json.dumps(
        {'raw': raw_input, 'reference': references},
        sort_keys=True,
        separators=(',', ':'),
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_decisions(
    cases: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> tuple[list[_Decision], list[SourceRecordFailure]]:
    decisions: list[_Decision] = []
    failures: list[SourceRecordFailure] = []
    for case_id, case in cases.items():
        predicted_case = predictions.get(case_id)
        if predicted_case is None:
            failures.append(
                SourceRecordFailure(
                    source_ref=case_id,
                    reason='no predictions found for this case id',
                )
            )
            continue
        pred_by_question = predicted_case.get('pred')
        if not isinstance(pred_by_question, dict):
            failures.append(
                SourceRecordFailure(
                    source_ref=case_id,
                    reason='predictions row has no "pred" object',
                )
            )
            continue
        for qid, question in case['questions'].items():
            gold_entry = case['gold'].get(qid)
            pred_entry = pred_by_question.get(qid)
            ref = f'{case_id}::{qid}'
            if gold_entry is None or pred_entry is None:
                failures.append(
                    SourceRecordFailure(
                        source_ref=ref,
                        reason=(
                            'question is missing its '
                            + (
                                'gold entry'
                                if gold_entry is None
                                else 'prediction'
                            )
                        ),
                    )
                )
                continue
            try:
                extracted, reference, is_correct = _decide(
                    question, gold_entry, pred_entry
                )
            except (KeyError, TypeError, ValueError) as exc:
                failures.append(
                    SourceRecordFailure(
                        source_ref=ref,
                        reason=f'unscorable decision: {exc}',
                    )
                )
                continue
            decisions.append(
                _Decision(
                    case=case,
                    qid=qid,
                    question_type=question.get('type', 'unknown'),
                    pred_entry=pred_entry,
                    extracted=extracted,
                    reference=reference,
                    is_correct=is_correct,
                )
            )
    return decisions, failures


def _ratio(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def _aggregate_stats(
    decisions: list[_Decision], n_cases: int
) -> dict[str, Any]:
    per_type: dict[str, list[bool]] = {}
    per_workflow: dict[str, list[bool]] = {}
    for decision in decisions:
        per_type.setdefault(decision.question_type, []).append(
            decision.is_correct
        )
        workflow = decision.case.get('workflow', 'unknown')
        per_workflow.setdefault(workflow, []).append(decision.is_correct)
    n_correct = sum(1 for decision in decisions if decision.is_correct)
    return {
        'accuracy': _ratio(n_correct, len(decisions)),
        'n_correct': n_correct,
        'n_decisions': len(decisions),
        'n_cases': n_cases,
        'per_type': {
            key: _ratio(sum(values), len(values))
            for key, values in sorted(per_type.items())
        },
        'per_workflow': {
            key: _ratio(sum(values), len(values))
            for key, values in sorted(per_workflow.items())
        },
    }


def _stat_string(value: float) -> str:
    return str(round(value, 4))


def build_instance_logs(
    decisions: list[_Decision],
    *,
    evaluation_id: str,
    evaluation_name: str,
    evaluation_result_id: str,
) -> list[InstanceLevelEvaluationLog]:
    instances: list[InstanceLevelEvaluationLog] = []
    for decision in decisions:
        case = decision.case
        question = case['questions'][decision.qid]
        raw_input = _raw_input_text(case, decision.qid, question)
        references = [decision.reference]
        choices = _choice_options(question)
        if choices is None and decision.question_type == 'score':
            criteria = question.get('criteria')
            n_levels = len(criteria) if isinstance(criteria, list) else 0
            choices = [str(i) for i in range(n_levels)]
        metadata = {
            'case_id': case['id'],
            'workflow': case.get('workflow', 'unknown'),
            'question_type': decision.question_type,
            'decision_rule': DECISION_RULES.get(
                decision.question_type, 'unknown'
            ),
            'probability_source': (
                'native typed interface '
                '(probability distribution in output.raw JSON)'
            ),
        }
        latency = case.get('latency_ms')
        performance = None
        if isinstance(latency, (int, float)) and not isinstance(latency, bool):
            performance = Performance(latency_ms=float(latency))
            metadata['latency_scope'] = 'case-level (one request per case)'
        instances.append(
            InstanceLevelEvaluationLog(
                schema_version=SCHEMA_VERSION,
                evaluation_id=evaluation_id,
                evaluation_result_id=evaluation_result_id,
                model_id=MODEL_ID,
                evaluation_name=evaluation_name,
                sample_id=f'{case["id"]}::{decision.qid}',
                sample_hash=sample_hash(raw_input, references),
                interaction_type=InteractionType.single_turn,
                input=Input(
                    raw=raw_input,
                    reference=references,
                    choices=choices,
                ),
                output=Output(
                    raw=[
                        json.dumps(
                            decision.pred_entry,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(',', ':'),
                        )
                    ]
                ),
                answer_attribution=[
                    AnswerAttributionItem(
                        turn_idx=0,
                        source='output.raw',
                        extracted_value=decision.extracted,
                        extraction_method=EXTRACTION_METHOD,
                        is_terminal=True,
                    )
                ],
                evaluation=Evaluation(
                    score=1.0 if decision.is_correct else 0.0,
                    is_correct=decision.is_correct,
                ),
                performance=performance,
                metadata=metadata,
            )
        )
    return instances


def build_aggregate(
    *,
    variant: str,
    stats: dict[str, Any],
    evaluation_id: str,
    evaluation_result_id: str,
    evaluation_timestamp: float,
    retrieved_timestamp: str,
    rows_path: Path,
    preds_path: Path,
) -> EvaluationLog:
    variant_info = VARIANTS[variant]
    evaluation_name = variant_info['evaluation_name']
    details = {
        'n_cases': str(stats['n_cases']),
        'n_decisions': str(stats['n_decisions']),
        'accuracy_choice': _stat_string(stats['per_type'].get('choice', 0.0)),
        'accuracy_noul': _stat_string(stats['per_type'].get('noul', 0.0)),
        'accuracy_score': _stat_string(stats['per_type'].get('score', 0.0)),
    }
    for workflow, value in stats['per_workflow'].items():
        details[f'accuracy_{workflow}'] = _stat_string(value)
    source_details = {
        'run_context': (
            'Self-run evaluation, one request per case (state in, five typed '
            'answers out); scored with the benchmark official scorer '
            'semantics (choice exact match, noul 0.5 threshold, score '
            'probability argmax).'
        ),
        'model_disclosure': (
            'Specialist model fitted on typed-decisions train-split '
            'material; results are not zero-shot.'
        ),
        'rows_sha256': _sha256_file(rows_path),
        'preds_sha256': _sha256_file(preds_path),
    }
    if variant == 'zh':
        source_details['dataset_variant'] = (
            'Machine-translated rows (API translation, no human post-edit); '
            'gold labels are byte-identical to the English test split.'
        )
    return EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=evaluation_id,
        evaluation_timestamp=f'{evaluation_timestamp:.3f}',
        retrieved_timestamp=retrieved_timestamp,
        source_metadata=SourceMetadata(
            source_name=f'{MODEL_NAME} self-run ({evaluation_name} accuracy)',
            source_type=SourceType.evaluation_run,
            source_organization_name=MODEL_DEVELOPER,
            source_organization_url=MODEL_ORG_URL,
            evaluator_relationship=EvaluatorRelationship.first_party,
            additional_details=source_details,
        ),
        eval_library=EvalLibrary(
            name=EVAL_LIBRARY_NAME,
            version=EVAL_LIBRARY_VERSION,
            additional_details={
                'note': (
                    'Model served locally through the laya 0.3.20 runtime; '
                    'run orchestrated by the project official scorer script.'
                )
            },
        ),
        model_info=ModelInfo(
            name=MODEL_NAME,
            id=MODEL_ID,
            developer=MODEL_DEVELOPER,
            inference_platform='local',
            inference_engine=InferenceEngine(
                name=EVAL_LIBRARY_NAME,
                version=EVAL_LIBRARY_VERSION,
            ),
            additional_details={
                'deployment_type': 'self_deployed',
                'model_availability': 'open_weights',
            },
        ),
        evaluation_results=[
            EvaluationResult(
                evaluation_result_id=evaluation_result_id,
                evaluation_name=evaluation_name,
                source_data=SourceDataHf(
                    dataset_name=variant_info['dataset_name'],
                    source_type='hf_dataset',
                    hf_repo=variant_info['dataset_repo'],
                    hf_split='test',
                    samples_number=stats['n_cases'],
                    additional_details={
                        'dataset_note': variant_info['dataset_note'],
                    },
                ),
                evaluation_timestamp=f'{evaluation_timestamp:.3f}',
                metric_config=MetricConfig(
                    evaluation_description=(
                        f'Accuracy = agreement with the gold label over all '
                        f'typed decisions ({stats["n_cases"]} cases x 5 '
                        f'questions = {stats["n_decisions"]} decisions), '
                        f'one request per case.'
                    ),
                    metric_id='accuracy',
                    metric_name='Accuracy',
                    metric_kind='accuracy',
                    metric_unit='proportion',
                    lower_is_better=False,
                    score_type=ScoreType.continuous,
                    min_score=0.0,
                    max_score=1.0,
                ),
                score_details=ScoreDetails(
                    score=round(stats['accuracy'], 4),
                    details=details,
                ),
            )
        ],
    )


def convert(
    args: argparse.Namespace,
) -> tuple[
    list[InstanceLevelEvaluationLog],
    dict[str, Any],
    SourceConversionResult[EvaluationLog],
]:
    """Convert one run and return its instance rows, stats, and result."""
    all_cases = _load_cases(args.rows, require_questions_and_gold=True)
    all_case_ids = set(all_cases)
    predictions = _load_cases(args.preds, require_questions_and_gold=False)
    cases = all_cases
    if args.limit_cases is not None:
        cases = dict(list(cases.items())[: args.limit_cases])
    decisions, failures = _collect_decisions(cases, predictions)
    for case_id in predictions:
        if case_id not in all_case_ids:
            failures.append(
                SourceRecordFailure(
                    source_ref=case_id,
                    reason='prediction row has no matching case in --rows',
                )
            )
    stats = _aggregate_stats(decisions, len(cases))
    result: SourceConversionResult[EvaluationLog] = SourceConversionResult(
        source_name=COLLECTION,
        total_records=len(cases),
        records=[],
        failures=failures,
    )
    if failures:
        return [], stats, result
    evaluation_timestamp = (
        _epoch_from_value(args.evaluation_timestamp, '--evaluation-timestamp')
        if args.evaluation_timestamp
        else args.preds.stat().st_mtime
    )
    retrieved_timestamp = (
        _timestamp_string(args.retrieved_timestamp)
        if args.retrieved_timestamp
        else f'{time.time():.3f}'
    )
    variant_info = VARIANTS[args.variant]
    evaluation_name = variant_info['evaluation_name']
    evaluation_id = (
        f'{evaluation_name}/{MODEL_ID}/{_iso_utc(evaluation_timestamp)}'
    )
    evaluation_result_id = f'{MODEL_ID}#{evaluation_name}#accuracy'
    aggregate = build_aggregate(
        variant=args.variant,
        stats=stats,
        evaluation_id=evaluation_id,
        evaluation_result_id=evaluation_result_id,
        evaluation_timestamp=evaluation_timestamp,
        retrieved_timestamp=retrieved_timestamp,
        rows_path=args.rows,
        preds_path=args.preds,
    )
    result = SourceConversionResult(
        source_name=COLLECTION,
        total_records=len(cases),
        records=[aggregate],
        failures=[],
    )
    return (
        build_instance_logs(
            decisions,
            evaluation_id=evaluation_id,
            evaluation_name=evaluation_name,
            evaluation_result_id=evaluation_result_id,
        ),
        stats,
        result,
    )


def run(args: argparse.Namespace) -> int:
    """Convert one typed-decisions run and publish it."""
    if args.output_dir.name != COLLECTION:
        raise SystemExit(
            f'--output-dir must end in {COLLECTION!r}, the datastore '
            f'collection this adapter writes; got {args.output_dir}'
        )
    instances, stats, result = convert(args)
    if result.failures:
        report = save_failure_report(
            result, default_failure_report_path(args.output_dir)
        )
        print(f'Provenance report: {report}')
        result.raise_if_incomplete()
    aggregate = result.records[0]
    evaluation_result = aggregate.evaluation_results[0]
    print(
        f'{evaluation_result.evaluation_name}: accuracy '
        f'{evaluation_result.score_details.score} over '
        f'{stats["n_decisions"]} decisions '
        f'({stats["n_cases"]} cases)'
    )
    file_uuid = str(uuid.uuid4())
    with tempfile.TemporaryDirectory(prefix='eee-typed-decisions-') as staging:
        staging_root = Path(staging)
        if args.include_instances:
            sample_bytes = ''.join(
                json.dumps(
                    row.model_dump(mode='json', exclude_none=True),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + '\n'
                for row in instances
            ).encode('utf-8')
            sample_name = f'{file_uuid}_samples.jsonl'
            staged_dir = datastore_output_dir(
                staging_root,
                COLLECTION,
                MODEL_ID,
                MODEL_DEVELOPER,
            )
            staged_dir.mkdir(parents=True, exist_ok=True)
            (staged_dir / sample_name).write_bytes(sample_bytes)
            aggregate.detailed_evaluation_results = DetailedEvaluationResults(
                format=Format.jsonl,
                file_path=datastore_repo_file_path(
                    COLLECTION,
                    MODEL_ID,
                    MODEL_DEVELOPER,
                    sample_name,
                ),
                hash_algorithm=HashAlgorithm.sha256,
                checksum=hashlib.sha256(sample_bytes).hexdigest(),
                total_rows=len(instances),
            )
        published = publish_evaluation_logs(
            [aggregate],
            args.output_dir.parent,
            [file_uuid],
            staged_output_dir=staging_root,
            collection_override=COLLECTION,
        )
    for path in published:
        print(path)
    return len(published)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == '__main__':
    published_count = main()
    print(
        f'Wrote {published_count} typed-decisions evaluation log(s).',
        flush=True,
    )
