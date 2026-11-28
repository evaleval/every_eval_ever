"""Instance-level companions from the Inspect logs Epoch links to its own runs.

Each Epoch-run row names its Inspect log in the ``Logs`` column (kept on the
record as ``source_metadata.additional_details['epoch_logs']``). This module
downloads that exact URL, converts the samples with the in-tree Inspect
instance-level converter, and attaches the result as the record's
``detailed_evaluation_results``.

A companion is attached only when the log demonstrably produced the published
score: the run completed, its model is the row's model, and exactly one
scorer reports the record's headline score. Anything else leaves the aggregate
record unchanged and is reported as a :class:`SourceRecordExclusion`, so a log
that cannot be fetched never costs the record itself.

Requires the ``inspect`` extra (``uv sync --extra inspect``).
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import requests

from every_eval_ever.eval_types import (
    DetailedEvaluationResults,
    EvaluationLog,
    Format,
    HashAlgorithm,
)
from every_eval_ever.helpers import SourceRecordExclusion, raw_capture
from every_eval_ever.helpers.io import (
    datastore_output_dir,
    datastore_repo_file_path,
)

# Samples converted per call to the Inspect instance-level converter, which
# holds one call's rows in memory; SWE-bench logs run to many GB.
SAMPLES_PER_CHUNK = 100
DEFAULT_MAX_LOG_MB = 1024

# Characters JSON leaves raw inside strings but str.splitlines() breaks on;
# the validator's companion check splits that way. Escaping them keeps every
# string identical once decoded.
_LINE_BREAKS = (
    ('\u2028'.encode(), b'\\u2028'),
    ('\u2029'.encode(), b'\\u2029'),
    ('\x85'.encode(), b'\\u0085'),
)


class LogUnavailable(Exception):
    """The log could not be used; the message is the exclusion reason."""


def download_log(url: str, dest: Path, max_bytes: int | None,
                 user_agent: str) -> str:
    """Stream *url* to *dest* and return its sha256.

    Raises LogUnavailable for a non-200 response or a log over *max_bytes*.
    """
    with requests.get(url, stream=True, timeout=(30, 300),
                      headers={'User-Agent': user_agent}) as response:
        if response.status_code != 200:
            raise LogUnavailable(
                f'log not downloadable (HTTP {response.status_code})')
        declared = int(response.headers.get('Content-Length') or 0)
        if max_bytes and declared > max_bytes:
            raise LogUnavailable(
                f'log is {declared / 1e6:.0f} MB, over the '
                f'{max_bytes / 1e6:.0f} MB limit (--max-log-mb)')
        digest = hashlib.sha256()
        written = 0
        with dest.open('wb') as handle:
            for chunk in response.iter_content(1 << 20):
                written += len(chunk)
                if max_bytes and written > max_bytes:
                    raise LogUnavailable(
                        f'log exceeds the {max_bytes / 1e6:.0f} MB limit '
                        '(--max-log-mb)')
                digest.update(chunk)
                handle.write(chunk)
    return digest.hexdigest()


def _decimals(value: float) -> int:
    text = repr(value)
    return 0 if 'e' in text else len(text.partition('.')[2])


def matching_scorers(header, score: float, percent: bool) -> list[str]:
    """Scorers with a metric equal to *score*, exactly or at its precision.

    Epoch publishes some scores rounded; a rounded value counts only when it
    keeps at least three decimals, so ``0.8`` does not vouch for ``0.837``.
    """
    places = _decimals(score)
    names = []
    for scorer in (header.results.scores if header.results else []):
        for metric_name, metric in scorer.metrics.items():
            if metric_name in ('stderr', 'std', 'var'):
                continue
            value = metric.value
            if not isinstance(value, (int, float)):
                continue
            value = value * 100 if percent else value
            if abs(value - score) <= 1e-9 or (
                places >= 3 and round(value, places) == score
            ):
                names.append(scorer.name)
                break
    return names


def _same_model(log_model: str, epoch_version: str,
                split_variant: Callable[[str], tuple[str, str | None]]) -> bool:
    base = split_variant(epoch_version)[0]
    return (log_model.rsplit('/', 1)[-1].casefold()
            == base.rsplit('/', 1)[-1].casefold())


def _reductions(path: Path):
    from inspect_ai.log import EvalSampleReductions

    with zipfile.ZipFile(path) as archive:
        if 'reductions.json' not in archive.namelist():
            return None
        data = json.loads(archive.read('reductions.json'))
    return [EvalSampleReductions.model_validate(item) for item in data]


def write_companion(log_path: Path, record: EvaluationLog, scorer: str,
                    dest: Path) -> int:
    """Convert every sample of *log_path* into JSONL rows at *dest*."""
    from inspect_ai.log import read_eval_log_samples

    from every_eval_ever.converters.inspect.instance_level_adapter import (
        InspectInstanceLevelDataAdapter,
    )

    result = record.evaluation_results[0]
    reductions = _reductions(log_path)
    rows = 0
    with tempfile.TemporaryDirectory() as scratch, dest.open('wb') as out:
        converter = InspectInstanceLevelDataAdapter(
            evaluation_id=record.evaluation_id,
            file_basename='chunk',
            format=Format.jsonl.value,
            hash_algorithm=HashAlgorithm.sha256.value,
            evaluation_dir=scratch,
        )

        def flush(batch):
            # the converter announces every file it writes; one log is many
            with contextlib.redirect_stdout(io.StringIO()):
                path, count = converter.convert_instance_level_logs(
                    result.evaluation_name, record.model_info.id, batch,
                    reductions, {scorer: [result.evaluation_result_id]},
                )
            content = Path(path).read_bytes()
            for raw, escaped in _LINE_BREAKS:
                content = content.replace(raw, escaped)
            out.write(content)
            return count

        batch = []
        for sample in read_eval_log_samples(log_path):
            batch.append(sample)
            if len(batch) == SAMPLES_PER_CHUNK:
                rows += flush(batch)
                batch = []
        if batch:
            rows += flush(batch)
    return rows


def attach_one(record: EvaluationLog, file_uuid: str, staged_root: Path,
               collection: str, workdir: Path, *, max_bytes: int | None,
               user_agent: str,
               split_variant: Callable[[str], tuple[str, str | None]],
               ) -> EvaluationLog:
    """Return *record* with its companion attached, or raise LogUnavailable."""
    from inspect_ai.log import read_eval_log

    url = record.source_metadata.additional_details['epoch_logs']
    log_path = workdir / f'{file_uuid}.eval'
    try:
        try:
            digest = download_log(url, log_path, max_bytes, user_agent)
        except requests.RequestException as exc:
            raise LogUnavailable(
                f'log download failed: {type(exc).__name__}') from exc
        raw_capture.record_pointer(
            kind='inspect_log', reference=url, revision=f'sha256:{digest}',
            url=url, label=record.evaluation_id,
        )
        try:
            header = read_eval_log(log_path, header_only=True)
        except Exception as exc:  # any unreadable archive
            raise LogUnavailable(
                f'log unreadable: {type(exc).__name__}: {exc}') from exc
        if header.status != 'success':
            raise LogUnavailable(f'log status is {header.status!r}')
        version = record.model_info.additional_details['epoch_model_version']
        if not _same_model(header.eval.model, version, split_variant):
            raise LogUnavailable(
                f'log model {header.eval.model!r} is not the row model '
                f'{version!r}')
        result = record.evaluation_results[0]
        scorers = matching_scorers(
            header, result.score_details.score,
            result.metric_config.metric_unit == 'percent')
        if len(scorers) != 1:
            raise LogUnavailable(
                'no scorer in the log reports the published score '
                f'{result.score_details.score}' if not scorers else
                f'several scorers report the published score: {scorers}')
        name = f'{file_uuid}_samples.jsonl'
        model_id = record.model_info.id
        developer = record.model_info.developer
        out_dir = datastore_output_dir(
            staged_root, collection, model_id, developer)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / name
        try:
            rows = write_companion(log_path, record, scorers[0], dest)
        except Exception as exc:  # a malformed sample in Epoch's log
            dest.unlink(missing_ok=True)
            raise LogUnavailable(
                f'log samples unreadable: {type(exc).__name__}: {exc}'
            ) from exc
        if rows == 0:
            dest.unlink()
            raise LogUnavailable('log has no samples')
        detailed = DetailedEvaluationResults(
            format=Format.jsonl,
            file_path=datastore_repo_file_path(
                collection, model_id, developer, name),
            hash_algorithm=HashAlgorithm.sha256,
            checksum=hashlib.sha256(dest.read_bytes()).hexdigest(),
            total_rows=rows,
        )
        return record.model_copy(
            update={'detailed_evaluation_results': detailed})
    finally:
        log_path.unlink(missing_ok=True)


def attach_instance_level(
    records: list[EvaluationLog], file_uuids: list[str], staged_root: Path,
    collection: str, *, max_bytes: int | None, user_agent: str,
    split_variant: Callable[[str], tuple[str, str | None]],
) -> tuple[list[EvaluationLog], list[SourceRecordExclusion]]:
    """Attach companions to every Epoch-run record that links a log."""
    out: list[EvaluationLog] = []
    skipped: list[SourceRecordExclusion] = []
    with tempfile.TemporaryDirectory() as workdir:
        for record, file_uuid in zip(records, file_uuids, strict=True):
            details = record.source_metadata.additional_details or {}
            if details.get('epoch_score_origin') != 'epoch_run' or not (
                details.get('epoch_logs')
            ):
                out.append(record)
                continue
            try:
                out.append(attach_one(
                    record, file_uuid, staged_root, collection,
                    Path(workdir), max_bytes=max_bytes,
                    user_agent=user_agent, split_variant=split_variant,
                ))
            except LogUnavailable as exc:
                out.append(record)
                skipped.append(SourceRecordExclusion(
                    source_ref=f'{record.evaluation_id}#inspect-log',
                    reason=f'no instance-level companion: {exc}',
                    source_record={'url': details['epoch_logs']},
                ))
    return out, skipped
