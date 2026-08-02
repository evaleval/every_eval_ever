"""Top-level CLI for conversion and validation."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from every_eval_ever.converters.common.publication import (
    publish_evaluation_logs,
)
from every_eval_ever.helpers.io import (
    SourceConversionResult,
    SourceRecordFailure,
    datastore_output_dir,
    default_failure_report_path,
    save_failure_report,
)

EVALUATOR_RELATIONSHIP_CHOICES = [
    'first_party',
    'third_party',
    'collaborative',
    'other',
]


def _common_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        'source_organization_name': args.source_organization_name,
        'evaluator_relationship': args.evaluator_relationship,
        'source_organization_url': args.source_organization_url,
        'source_organization_logo_url': args.source_organization_logo_url,
        'eval_library_name': args.eval_library_name,
        'eval_library_version': args.eval_library_version,
        'parent_eval_output_dir': args.output_dir,
    }


def _output_dir_for_log(base_output: Path, log: Any) -> Path:
    if not log.evaluation_results:
        raise ValueError(
            'evaluation_results must contain at least one result so the '
            'output collection can be determined'
        )
    source_data = log.evaluation_results[0].source_data
    if source_data is None:
        raise ValueError(
            'evaluation_results[0].source_data is required for the output path'
        )
    out_dir = datastore_output_dir(
        base_output,
        source_data.dataset_name,
        log.model_info.id,
        log.model_info.developer,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_partial_conversion_report(
    result: SourceConversionResult[Any] | None,
    output_dir: str | Path,
    name: str,
) -> None:
    if result is None or not (result.failures or result.exclusions):
        return
    report_path = save_failure_report(
        result,
        default_failure_report_path(Path(output_dir) / name),
    )
    print(f'Provenance report: {report_path}')


def _cmd_convert_lm_eval(args: argparse.Namespace) -> int:
    from every_eval_ever.converters.lm_eval.adapter import LMEvalAdapter
    from every_eval_ever.converters.lm_eval.instance_level_adapter import (
        LMEvalInstanceLevelAdapter,
    )
    from every_eval_ever.converters.lm_eval.utils import find_samples_file

    adapter = LMEvalAdapter()
    metadata = _common_metadata(args)
    if args.inference_engine:
        metadata['inference_engine'] = args.inference_engine
    if args.inference_engine_version:
        metadata['inference_engine_version'] = args.inference_engine_version

    log_path = Path(args.log_path)
    metadata['parent_eval_output_dir'] = str(
        log_path.parent if log_path.is_file() else log_path
    )
    input_result: SourceConversionResult[Any] | None = None
    if log_path.is_file():
        logs = adapter.transform_from_file(log_path, metadata)
    elif log_path.is_dir():
        input_result = adapter.transform_from_directory_result(
            log_path, metadata
        )
        logs = input_result.records
    else:
        raise FileNotFoundError(f'Path is not a file or directory: {log_path}')

    if not logs and input_result is None:
        raise ValueError(f'lm-eval conversion produced no logs from {log_path}')

    output_dir = Path(args.output_dir)
    eval_uuids = [str(uuid.uuid4()) for _ in logs]
    sample_result: SourceConversionResult[Any] | None = None
    with tempfile.TemporaryDirectory(prefix='eee-lm-eval-') as staging:
        staging_dir = Path(staging)
        publish_logs = []
        publish_uuids = []
        sample_successes = []
        sample_failures = []
        for log, eval_uuid in zip(logs, eval_uuids, strict=True):
            meta: dict[str, Any] = {}
            try:
                if not args.include_samples:
                    publish_logs.append(log)
                    publish_uuids.append(eval_uuid)
                    continue
                meta = adapter.get_eval_metadata(log.evaluation_id)
                parent_dir = meta.get('parent_dir')
                task_name = meta.get('task_name')
                if not parent_dir or not task_name:
                    raise RuntimeError(
                        'lm-eval converter lost the source location or task '
                        f'name for evaluation {log.evaluation_id!r}'
                    )
                samples_file = find_samples_file(Path(parent_dir), task_name)
                if samples_file is None:
                    raise FileNotFoundError(
                        '--include-samples was requested, but no upstream '
                        f'samples file was found for task {task_name!r} under '
                        f'{parent_dir}'
                    )
                detailed = LMEvalInstanceLevelAdapter().transform_and_save(
                    samples_path=samples_file,
                    evaluation_id=log.evaluation_id,
                    model_id=log.model_info.id,
                    task_name=task_name,
                    output_dir=str(_output_dir_for_log(staging_dir, log)),
                    file_uuid=eval_uuid,
                    collection=log.evaluation_results[
                        0
                    ].source_data.dataset_name,
                    developer=log.model_info.developer,
                )
                if detailed is None:
                    raise ValueError(
                        '--include-samples was requested, but the upstream '
                        f'samples file for task {task_name!r} contained no '
                        'usable rows'
                    )
                log.detailed_evaluation_results = detailed
                publish_logs.append(log)
                publish_uuids.append(eval_uuid)
                sample_successes.append(log)
            except Exception as exc:
                publish_logs.append(log)
                publish_uuids.append(eval_uuid)
                task_name = meta.get('task_name')
                sample_failures.append(
                    SourceRecordFailure(
                        source_ref=(
                            f'lm-eval evaluation {log.evaluation_id!r}'
                        ),
                        reason=str(exc),
                        source_record={
                            'evaluation_id': log.evaluation_id,
                            'searched_directory': meta.get('parent_dir'),
                            'task_name': task_name,
                            'expected_samples_pattern': (
                                f'samples_{task_name}_*.jsonl'
                                if task_name
                                else None
                            ),
                        },
                    )
                )
        if args.include_samples:
            sample_result = SourceConversionResult(
                source_name='lm-eval requested sample conversions',
                total_records=len(logs),
                records=sample_successes,
                failures=sample_failures,
            )
        paths = (
            publish_evaluation_logs(
                publish_logs,
                output_dir,
                publish_uuids,
                staged_output_dir=staging_dir,
            )
            if publish_logs
            else []
        )
    for path in paths:
        print(path)

    _save_partial_conversion_report(input_result, output_dir, 'lm_eval_inputs')
    _save_partial_conversion_report(
        sample_result, output_dir, 'lm_eval_samples'
    )
    if input_result is not None:
        input_result.raise_if_incomplete()
    if sample_result is not None:
        sample_result.raise_if_incomplete()
    print(f'Converted {len(paths)} evaluation log(s).')
    return 0


def _cmd_convert_inspect(args: argparse.Namespace) -> int:
    from every_eval_ever.converters.inspect.adapter import (
        InspectAIAdapter,
        list_eval_logs,
    )
    from every_eval_ever.converters.inspect.supplemental_eval_details import (
        SupplementalEvalDetails,
    )

    adapter = InspectAIAdapter()
    metadata = _common_metadata(args)
    supplemental_path = getattr(args, 'supplemental_eval_details_path', None)
    if supplemental_path:
        metadata['supplemental_eval_details'] = (
            SupplementalEvalDetails.model_validate_json(
                Path(supplemental_path).read_text(encoding='utf-8')
            )
        )
    log_path = Path(args.log_path)
    with tempfile.TemporaryDirectory(prefix='eee-inspect-') as staging:
        staging_dir = Path(staging)
        metadata['parent_eval_output_dir'] = str(staging_dir)
        eval_uuids: list[str]
        conversion_result: SourceConversionResult[Any] | None = None
        if log_path.is_file():
            eval_uuids = [str(uuid.uuid4())]
            metadata['file_uuid'] = eval_uuids[0]
            logs = [adapter.transform_from_file(log_path, metadata)]
        elif log_path.is_dir():
            eval_paths = sorted(
                list_eval_logs(log_path.absolute().as_posix()),
                key=lambda path: path.name,
            )
            if not eval_paths:
                raise ValueError(
                    f'Inspect conversion found no evaluation logs in {log_path}'
                )
            eval_uuids = [str(uuid.uuid4()) for _ in eval_paths]
            metadata['file_uuids'] = eval_uuids
            conversion_result = adapter.transform_from_directory_result(
                log_path, metadata
            )
            logs = [log for log, _ in conversion_result.records]
            eval_uuids = [
                file_uuid
                for _, file_uuid in conversion_result.records
                if file_uuid is not None
            ]
        else:
            raise FileNotFoundError(
                f'Path is not a file or directory: {log_path}'
            )

        if not logs and conversion_result is None:
            raise ValueError(
                f'Inspect conversion produced no logs from {log_path}'
            )
        if len(logs) != len(eval_uuids):
            raise RuntimeError(
                'Inspect conversion produced a different number of logs than '
                'the generated UUID list.'
            )
        paths = (
            publish_evaluation_logs(
                logs,
                args.output_dir,
                eval_uuids,
                staged_output_dir=staging_dir,
            )
            if logs
            else []
        )
    for path in paths:
        print(path)

    _save_partial_conversion_report(
        conversion_result, args.output_dir, 'inspect'
    )
    if conversion_result is not None:
        conversion_result.raise_if_incomplete()
    print(f'Converted {len(paths)} evaluation log(s).')
    return 0


def _cmd_convert_helm(args: argparse.Namespace) -> int:
    from every_eval_ever.converters.helm.adapter import HELMAdapter

    adapter = HELMAdapter()
    metadata = _common_metadata(args)
    log_path = Path(args.log_path)

    eval_uuids: list[str]
    if adapter._directory_contains_required_files(log_path):
        eval_uuids = [str(uuid.uuid4())]
        metadata['file_uuid'] = eval_uuids[0]
    elif log_path.is_dir():
        run_dirs = [
            entry.path
            for entry in os.scandir(log_path)
            if entry.is_dir()
            and adapter._directory_contains_required_files(entry.path)
        ]
        if not run_dirs:
            raise ValueError(
                f'HELM conversion found no valid run directories in {log_path}'
            )
        eval_uuids = [str(uuid.uuid4()) for _ in run_dirs]
        metadata['file_uuids'] = eval_uuids
    else:
        raise FileNotFoundError(f'Path is not a file or directory: {log_path}')

    with tempfile.TemporaryDirectory(prefix='eee-helm-') as staging:
        staging_dir = Path(staging)
        metadata['parent_eval_output_dir'] = str(staging_dir)
        conversion_result = adapter.transform_from_directory_result(
            log_path,
            output_path=str(staging_dir),
            metadata_args=metadata,
        )
        logs = [log for log, _ in conversion_result.records]
        eval_uuids = [
            file_uuid
            for _, file_uuid in conversion_result.records
            if file_uuid is not None
        ]

        if not logs and not conversion_result.failures:
            raise ValueError(
                f'HELM conversion produced no logs from {log_path}'
            )
        if len(logs) != len(eval_uuids):
            raise RuntimeError(
                'HELM conversion produced a different number of logs than '
                'the generated UUID list.'
            )
        paths = (
            publish_evaluation_logs(
                logs,
                args.output_dir,
                eval_uuids,
                staged_output_dir=staging_dir,
            )
            if logs
            else []
        )
    for path in paths:
        print(path)

    _save_partial_conversion_report(conversion_result, args.output_dir, 'helm')
    conversion_result.raise_if_incomplete()
    print(f'Converted {len(paths)} evaluation log(s).')
    return 0


def _cmd_convert_alpaca_eval(args: argparse.Namespace) -> int:
    from every_eval_ever.converters.alpaca_eval.adapter import (
        LEADERBOARDS,
        AlpacaEvalAdapter,
    )

    adapter = AlpacaEvalAdapter()
    versions = [args.version] if args.version else list(LEADERBOARDS.keys())
    output_dir = Path(args.output_dir)

    logs_to_publish = []
    eval_uuids = []
    conversion_results = []
    for version in versions:
        cfg_name = LEADERBOARDS[version]['source_name']
        print(f'\n=== {cfg_name} ===')
        try:
            conversion_result = adapter.fetch_leaderboard_result(version)
        except Exception as exc:
            conversion_result = SourceConversionResult(
                source_name=f'AlpacaEval {version}',
                total_records=1,
                records=[],
                failures=[
                    SourceRecordFailure(
                        source_ref=f'AlpacaEval leaderboard {version!r}',
                        reason=str(exc),
                        source_record={
                            'version': version,
                            'source_name': cfg_name,
                            'url': LEADERBOARDS[version]['url'],
                        },
                    )
                ],
            )
        conversion_results.append((version, conversion_result))
        if not conversion_result.records and not conversion_result.failures:
            conversion_result = SourceConversionResult(
                source_name=conversion_result.source_name,
                total_records=conversion_result.total_records,
                records=[],
                failures=[
                    SourceRecordFailure(
                        source_ref=f'AlpacaEval leaderboard {version!r}',
                        reason='conversion produced no logs',
                        source_record={
                            'version': version,
                            'source_name': cfg_name,
                            'url': LEADERBOARDS[version]['url'],
                        },
                    )
                ],
                exclusions=conversion_result.exclusions,
            )
            conversion_results[-1] = (version, conversion_result)

        for log in conversion_result.records:
            if args.source_organization_name != 'unknown':
                log.source_metadata.source_organization_name = (
                    args.source_organization_name
                )
            if args.source_organization_url is not None:
                log.source_metadata.source_organization_url = (
                    args.source_organization_url
                )
            if args.evaluator_relationship != 'third_party':
                from every_eval_ever.eval_types import EvaluatorRelationship

                log.source_metadata.evaluator_relationship = (
                    EvaluatorRelationship(args.evaluator_relationship)
                )
            if args.eval_library_name != 'alpaca_eval':
                log.eval_library.name = args.eval_library_name
            if args.eval_library_version != 'unknown':
                log.eval_library.version = args.eval_library_version

            logs_to_publish.append(log)
            eval_uuids.append(str(uuid.uuid4()))

    paths = publish_evaluation_logs(logs_to_publish, output_dir, eval_uuids)
    for path in paths:
        print(f'  {path}')
    for version, conversion_result in conversion_results:
        if conversion_result.failures or conversion_result.exclusions:
            report_path = save_failure_report(
                conversion_result,
                default_failure_report_path(
                    output_dir / f'alpaca_eval_{version}'
                ),
            )
            print(f'Provenance report: {report_path}')
    for _, conversion_result in conversion_results:
        conversion_result.raise_if_incomplete()
    print(f'\nConverted {len(paths)} model evaluation(s).')
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='every_eval_ever',
        description=(
            'CLI for validating and converting evaluation results into the '
            'Every Eval Ever schema.'
        ),
        epilog=(
            'Examples:\n'
            '  every_eval_ever convert lm_eval --log_path results.json --output_dir data\n'
            '  every_eval_ever convert inspect --log_path inspect_log.json --output_dir data\n'
            '  every_eval_ever convert helm --log_path helm_run_dir --output_dir data'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate JSON and JSONL files with Pydantic models',
        description=(
            'Validate aggregate .json and instance-level .jsonl files '
            'using the bundled Pydantic models.'
        ),
    )
    validate_parser.add_argument(
        'paths',
        nargs='+',
        help=(
            'Files or glob patterns to validate. Directory arguments are not '
            'supported; use a glob to select files.'
        ),
    )
    validate_parser.add_argument(
        '--max-errors',
        type=int,
        default=50,
        help='Maximum errors to report per JSONL file.',
    )
    validate_parser.add_argument(
        '--format',
        choices=['rich', 'json', 'github'],
        default='rich',
        dest='output_format',
        help='Output format.',
    )
    check_duplicates_parser = subparsers.add_parser(
        'check-duplicates',
        help='Detect duplicate evaluation JSON entries',
        description=(
            'Detect duplicate evaluation entries while ignoring scrape-specific '
            'keys (evaluation_id and retrieved_timestamp).'
        ),
    )
    check_duplicates_parser.add_argument(
        'paths',
        nargs='+',
        help='One or more JSON files or directories containing JSON files.',
    )

    convert_parser = subparsers.add_parser(
        'convert',
        help='Convert source eval logs to every_eval_ever',
        description='Convert outputs from supported eval frameworks into Every Eval Ever JSON.',
    )
    convert_subparsers = convert_parser.add_subparsers(
        dest='source', required=True
    )

    for source in ['lm_eval', 'inspect', 'helm', 'alpaca_eval']:
        source_parser = convert_subparsers.add_parser(
            source,
            help=f'Convert {source} logs',
            description=f'Convert {source} evaluation outputs to Every Eval Ever format.',
        )
        source_parser.add_argument(
            '--log_path',
            '--log-path',
            required=(source != 'alpaca_eval'),
            help='Path to source log file or directory to convert.',
        )
        source_parser.add_argument(
            '--output_dir',
            '--output-dir',
            default='data',
            help='Base output directory where converted files are written.',
        )
        source_parser.add_argument(
            '--source_organization_name',
            '--source-organization-name',
            default='unknown',
            help='Organization name for source_metadata.source_organization_name.',
        )
        source_parser.add_argument(
            '--evaluator_relationship',
            '--evaluator-relationship',
            default='third_party',
            choices=EVALUATOR_RELATIONSHIP_CHOICES,
            help='Relationship between evaluator and model developer.',
        )
        source_parser.add_argument(
            '--source_organization_url',
            '--source-organization-url',
            default=None,
            help='Optional organization URL for source metadata.',
        )
        source_parser.add_argument(
            '--source_organization_logo_url',
            '--source-organization-logo-url',
            default=None,
            help='Optional organization logo URL for source metadata.',
        )
        source_parser.add_argument(
            '--eval_library_name',
            '--eval-library-name',
            default=source,
            help='Evaluation library name recorded in eval_library.name.',
        )
        source_parser.add_argument(
            '--eval_library_version',
            '--eval-library-version',
            default='unknown',
            help='Evaluation library version recorded in eval_library.version.',
        )

        if source == 'alpaca_eval':
            source_parser.add_argument(
                '--version',
                choices=['v1', 'v2'],
                default=None,
                help=(
                    'Which leaderboard version to convert: v1 (AlpacaEval 1.0) '
                    'or v2 (AlpacaEval 2.0). Omit to convert both (default).'
                ),
            )

        if source == 'lm_eval':
            source_parser.add_argument(
                '--include_samples',
                '--include-samples',
                action='store_true',
                help='Also convert lm-eval sample JSONL into instance-level output.',
            )
            source_parser.add_argument(
                '--inference_engine',
                '--inference-engine',
                default=None,
                help='Override inferred inference engine (e.g. vllm, transformers).',
            )
            source_parser.add_argument(
                '--inference_engine_version',
                '--inference-engine-version',
                default=None,
                help='Inference engine version to record in model_info.inference_engine.version.',
            )
        if source == 'inspect':
            source_parser.add_argument(
                '--supplemental_eval_details_path',
                '--supplemental-eval-details-path',
                default=None,
                help=(
                    'Optional JSON file containing supplemental model and '
                    'evaluation details.'
                ),
            )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == 'validate':
        from every_eval_ever.validate import main as validate_main

        validate_args = [
            '--max-errors',
            str(args.max_errors),
            '--format',
            args.output_format,
        ]
        validate_args.extend(args.paths)
        return validate_main(validate_args)

    if args.command == 'check-duplicates':
        from every_eval_ever.check_duplicate_entries import (
            main as check_duplicates_main,
        )

        return check_duplicates_main(args.paths)

    if args.command == 'convert':
        if args.source == 'lm_eval':
            return _cmd_convert_lm_eval(args)
        if args.source == 'inspect':
            return _cmd_convert_inspect(args)
        if args.source == 'helm':
            return _cmd_convert_helm(args)
        if args.source == 'alpaca_eval':
            return _cmd_convert_alpaca_eval(args)

    parser.print_help()
    return 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
