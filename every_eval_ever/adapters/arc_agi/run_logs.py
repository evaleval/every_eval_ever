"""Join ARC Prize leaderboard score rows to the published run logs.

ARC Prize publishes the outputs of its public-evaluation runs as two Hugging
Face datasets, one per public set. Each holds one directory per run, named
after the leaderboard ``modelId``, with one JSON file per task (a list of test
pairs, each mapping ``attempt_<n>`` to the logged request and its outcome) and
a ``results.json`` summarising the run.

A directory speaks for a leaderboard score only when it passes every gate in
:func:`read_run` and :func:`row_rejection`. What it then states (the output
cap every logged request carried, a temperature every request shared, the
number of tasks scored) is written onto that score result only.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from every_eval_ever.helpers import raw_capture


@dataclass(frozen=True)
class RunDataset:
    """One published run-output dataset, pinned to a commit."""

    dataset_id: str
    repo_id: str
    revision: str
    subdir: str

    @property
    def url(self) -> str:
        return f'https://huggingface.co/datasets/{self.repo_id}'


#: Leaderboard ``datasetId`` -> the dataset holding that set's run logs. Only
#: the public sets publish outputs; a directory fills its own set only.
RUN_DATASETS: dict[str, RunDataset] = {
    'v1_Public_Eval': RunDataset(
        dataset_id='v1_Public_Eval',
        repo_id='arcprize/arc_agi_v1_public_eval',
        revision='3e9c9d1a8402aff82356815c106cecaea65cb7d9',
        subdir='v1',
    ),
    'v2_Public_Eval': RunDataset(
        dataset_id='v2_Public_Eval',
        repo_id='arcprize/arc_agi_v2_public_eval',
        revision='026789c1c12a4c34580a32e84dcaf5630d7e8f31',
        subdir='v2',
    ),
}

#: Output-cap spellings a logged request may carry. A run states exactly one,
#: under the same key and value, on every recorded request.
CAP_KEYS = ('max_tokens', 'max_completion_tokens', 'max_output_tokens')
RESULTS_FILE = 'results.json'
_ATTEMPT_KEY = re.compile(r'^attempt_\d+$')


class StatedFloat(float):
    """A float parsed from JSON that keeps the text it was written as."""

    text: str

    def __new__(cls, text: str) -> StatedFloat:
        value = super().__new__(cls, text)
        value.text = text
        return value


def loads_stated(text: str) -> Any:
    """Parse JSON keeping each float's written form (see :class:`StatedFloat`)."""
    return json.loads(text, parse_float=StatedFloat)


def stated_decimal(value: float | int) -> Decimal:
    """Return ``value`` as a Decimal at the precision it was written with.

    A :class:`StatedFloat` keeps its JSON text, so ``0.470`` stays three
    decimals; any other number falls back to its shortest ``repr``.
    """
    if isinstance(value, StatedFloat):
        return Decimal(value.text)
    if isinstance(value, int):
        return Decimal(value)
    return Decimal(repr(float(value)))


class RunRejected(Exception):
    """A run directory failed a gate; ``reason`` names the gate."""

    def __init__(self, reason: str, detail: str = '') -> None:
        super().__init__(f'{reason}: {detail}' if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class PublishedRun:
    """One run as its logs state it, after the directory-level gates."""

    directory: str
    cap_key: str
    cap_value: int
    temperature: float | None
    score: float
    avg_cost_per_task: Any
    latest_end: datetime
    task_count: int
    attempt_count: int


@dataclass(frozen=True)
class RunLogSource:
    """A local checkout of one run dataset and, if known, its path history."""

    dataset: RunDataset
    root: Path
    revision: str
    history: dict[str, int]


@dataclass(frozen=True)
class JoinOutcome:
    """What one score row's join produced."""

    dataset_id: str
    model_id: str
    run: PublishedRun | None = None
    rejection: str | None = None
    detail: str = ''
    request_log: str = ''


@dataclass
class RunLogs:
    """Run-log sources keyed by leaderboard ``datasetId``, plus join outcomes."""

    sources: dict[str, RunLogSource]
    outcomes: list[JoinOutcome] = field(default_factory=list)
    unjoined: list[tuple[str, str]] = field(default_factory=list)
    _runs: dict[tuple[str, str], PublishedRun | RunRejected] = field(
        default_factory=dict
    )

    def join(
        self, dataset_id: str, row: dict[str, Any], retrieved: float
    ) -> JoinOutcome | None:
        """Join one chosen score row to its run directory.

        Returns ``None`` when the set publishes no logs or no directory is
        named after the row's ``modelId``.
        """
        source = self.sources.get(dataset_id)
        model_id = row.get('modelId')
        if source is None or not isinstance(model_id, str):
            return None
        directory = source.root / model_id
        if (
            not model_id
            or model_id.startswith('.')
            or '/' in model_id
            or not directory.is_dir()
        ):
            self.unjoined.append((dataset_id, model_id))
            return None
        key = (dataset_id, model_id)
        if key not in self._runs:
            try:
                self._runs[key] = read_run(directory, source.history)
            except RunRejected as exc:
                self._runs[key] = exc
        read = self._runs[key]
        request_log = f'{source.dataset.repo_id}@{source.revision}/{model_id}'
        if isinstance(read, RunRejected):
            outcome = JoinOutcome(
                dataset_id,
                model_id,
                rejection=read.reason,
                detail=read.detail,
                request_log=request_log,
            )
        else:
            reason = row_rejection(read, row, retrieved)
            outcome = JoinOutcome(
                dataset_id,
                model_id,
                run=None if reason else read,
                rejection=reason,
                request_log=request_log,
            )
        self.outcomes.append(outcome)
        return outcome


def rounds_to(value: float, stated: float | int) -> bool:
    """Whether ``value`` rounds to ``stated`` at the precision ``stated`` spells out."""
    target = stated_decimal(stated)
    half = Decimal(1).scaleb(target.as_tuple().exponent) / 2
    return abs(Decimal(repr(float(value))) - target) <= half


def _sums_to(value: float, stated: float, terms: int) -> bool:
    """Whether a sum of ``terms`` task scores equals one summed in another order."""
    spread = terms * sys.float_info.epsilon * max(abs(value), abs(stated), 1.0)
    return abs(value - stated) <= spread


def parse_timestamp(value: Any) -> datetime | None:
    """Parse an epoch number or ISO-8601 string to an aware UTC datetime."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _attempt_cap(kwargs: dict[str, Any], where: str) -> tuple[str, int]:
    stated = [key for key in CAP_KEYS if key in kwargs]
    if not stated:
        raise RunRejected('cap_absent', where)
    if len(stated) > 1:
        raise RunRejected('cap_ambiguous', f'{where}: {", ".join(stated)}')
    value = kwargs[stated[0]]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RunRejected('cap_not_an_integer', f'{where}: {value!r}')
    return stated[0], value


def read_history(repo_dir: Path, revision: str) -> dict[str, int]:
    """Count the commits up to ``revision`` that touched each path.

    Reads commits and trees only, so it works in a blobless clone without
    fetching file contents.
    """
    log = _git(
        '-C',
        str(repo_dir),
        'log',
        '--name-only',
        '--no-renames',
        '--pretty=format:',
        revision,
        '--',
    )
    return dict(Counter(line for line in log.splitlines() if line))


def read_run(directory: Path, history: dict[str, int]) -> PublishedRun:
    """Reconstruct one run from its directory, or raise :class:`RunRejected`.

    Gates: no path the directory ever held was touched by a second commit
    according to ``history``; every recorded attempt names this directory
    as its ``test_id``, its file's task id, and its pair index; every recorded
    request states one output cap under one key with one value; the task set,
    each task's score and attempt count, the total attempts and the aggregate
    all recompute from the task files to ``results.json``.

    A pair counts as solved when any attempt is correct, and a task scores the
    fraction of its pairs solved. An attempt recorded as ``null`` rejects the
    run: its request arguments, and so its cap, are not logged.
    """
    results_path = directory / RESULTS_FILE
    if not results_path.is_file():
        raise RunRejected('no_results_file')
    try:
        published = json.loads(results_path.read_text(encoding='utf-8'))
    except ValueError as exc:
        raise RunRejected('results_file_unreadable', str(exc)) from exc
    task_results = (
        published.get('task_results') if isinstance(published, dict) else None
    )
    if not isinstance(task_results, dict):
        raise RunRejected('results_file_without_task_results')

    prefix = f'{directory.name}/'
    rewritten = sorted(
        path
        for path, commits in history.items()
        if path.startswith(prefix) and commits > 1
    )
    if rewritten:
        raise RunRejected('run_rewritten', ', '.join(rewritten[:5]))

    task_paths = sorted(
        path
        for path in directory.glob('*.json')
        if path.is_file() and path.name != RESULTS_FILE
    )
    if not task_paths:
        raise RunRejected('no_task_files')

    scores: dict[str, float] = {}
    attempts_per_task: dict[str, int] = {}
    cap: tuple[str, int] | None = None
    temperatures: list[Any] = []
    latest_end: datetime | None = None

    for path in task_paths:
        task_id = path.stem
        try:
            pairs = json.loads(path.read_text(encoding='utf-8'))
        except ValueError as exc:
            raise RunRejected(
                'task_file_unreadable', f'{path.name}: {exc}'
            ) from exc
        if not isinstance(pairs, list) or not pairs:
            raise RunRejected('task_file_not_a_pair_list', path.name)
        solved = []
        task_attempts = 0
        for index, pair in enumerate(pairs):
            if not isinstance(pair, dict) or not pair:
                raise RunRejected('task_file_not_a_pair_list', path.name)
            correct = False
            for key, attempt in pair.items():
                where = f'{path.name}: {key}'
                if not _ATTEMPT_KEY.match(key):
                    raise RunRejected('attempt_key_unknown', where)
                task_attempts += 1
                if not isinstance(attempt, dict):
                    raise RunRejected('attempt_not_recorded', where)
                metadata = attempt.get('metadata')
                if not isinstance(metadata, dict):
                    raise RunRejected('attempt_without_metadata', where)
                if metadata.get('task_id') != task_id:
                    raise RunRejected(
                        'task_id_mismatch',
                        f'{where}: {metadata.get("task_id")!r}',
                    )
                if metadata.get('test_id') != directory.name:
                    raise RunRejected(
                        'test_id_mismatch',
                        f'{where}: {metadata.get("test_id")!r}',
                    )
                pair_index = metadata.get('pair_index')
                if (
                    isinstance(pair_index, bool)
                    or not isinstance(pair_index, int)
                    or pair_index != index
                ):
                    raise RunRejected(
                        'pair_index_mismatch',
                        f'{where}: {pair_index!r} at {index}',
                    )
                end = parse_timestamp(metadata.get('end_timestamp'))
                if end is None:
                    raise RunRejected('attempt_without_end_time', where)
                latest_end = end if latest_end is None else max(latest_end, end)
                kwargs = metadata.get('kwargs')
                if not isinstance(kwargs, dict):
                    raise RunRejected('attempt_without_kwargs', where)
                attempt_cap = _attempt_cap(kwargs, where)
                if cap is None:
                    cap = attempt_cap
                elif attempt_cap[0] != cap[0]:
                    raise RunRejected(
                        'cap_key_mixed', f'{cap[0]} vs {attempt_cap[0]}'
                    )
                elif attempt_cap[1] != cap[1]:
                    raise RunRejected(
                        'cap_conflict', f'{cap[1]} vs {attempt_cap[1]}'
                    )
                temperatures.append(kwargs.get('temperature'))
                if not isinstance(attempt.get('correct'), bool):
                    raise RunRejected('correct_not_recorded', where)
                correct = correct or attempt['correct']
            solved.append(correct)
        scores[task_id] = sum(solved) / len(solved)
        attempts_per_task[task_id] = task_attempts

    if cap is None or latest_end is None:
        raise RunRejected('no_recorded_attempts')
    if set(scores) != set(task_results):
        missing = len(set(task_results) - set(scores))
        extra = len(set(scores) - set(task_results))
        raise RunRejected(
            'task_set_mismatch',
            f'{missing} not published as files, {extra} not in results.json',
        )
    for task_id, score in sorted(scores.items()):
        stated = task_results[task_id]
        stated = stated if isinstance(stated, dict) else {}
        stated_score = stated.get('score')
        if not _is_number(stated_score) or not rounds_to(score, stated_score):
            raise RunRejected(
                'task_score_mismatch', f'{task_id}: {stated_score!r}'
            )
        stated_attempts = stated.get('attempts')
        if (
            stated_attempts is not None
            and stated_attempts != attempts_per_task[task_id]
        ):
            raise RunRejected(
                'task_attempts_mismatch',
                f'{task_id}: {stated_attempts!r} vs {attempts_per_task[task_id]}',
            )
    total_attempts = sum(attempts_per_task.values())
    for name, recomputed, terms in (
        ('total_tasks', len(scores), 0),
        ('total_attempts', total_attempts, 0),
        ('score', sum(scores.values()), len(scores)),
    ):
        stated = published.get(name)
        matches = _is_number(stated) and (
            _sums_to(float(recomputed), float(stated), terms)
            if terms
            else float(recomputed) == float(stated)
        )
        if not matches:
            raise RunRejected(
                'run_score_mismatch', f'{name}: {stated!r} vs {recomputed}'
            )

    temperature = None
    if (
        temperatures
        and all(_is_number(t) for t in temperatures)
        and len({float(t) for t in temperatures}) == 1
    ):
        temperature = float(temperatures[0])

    return PublishedRun(
        directory=directory.name,
        cap_key=cap[0],
        cap_value=cap[1],
        temperature=temperature,
        score=sum(scores.values()) / len(scores),
        avg_cost_per_task=published.get('avg_cost_per_task'),
        latest_end=latest_end,
        task_count=len(scores),
        attempt_count=total_attempts,
    )


def row_rejection(
    run: PublishedRun, row: dict[str, Any], retrieved: float
) -> str | None:
    """Name the gate a leaderboard row fails against a run, or ``None``.

    The run's recomputed score must round to the row's ``score`` and its
    ``avg_cost_per_task`` to the row's ``costPerTask``, each at the precision
    the row spells out, and the run must have ended before retrieval.
    """
    if not _is_number(row.get('score')):
        return 'score_not_stated'
    if not rounds_to(run.score, row['score']):
        return 'score_mismatch'
    if not _is_number(row.get('costPerTask')):
        return 'cost_not_stated'
    if not _is_number(run.avg_cost_per_task) or not rounds_to(
        run.avg_cost_per_task, row['costPerTask']
    ):
        return 'cost_mismatch'
    if run.latest_end.timestamp() > retrieved:
        return 'run_ended_after_retrieval'
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _git_head(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ['git', '-C', str(path), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _is_git_root(path: Path) -> bool:
    return (path / '.git').exists()


class RunLogsUnavailable(RuntimeError):
    """Local run logs cannot be read under the same gates as a live run."""


def load_history_file(path: Path, revision: str) -> dict[str, int]:
    """Read a path-history file: ``{"revision": sha, "paths": {path: commits}}``.

    The file is what :func:`read_history` returns for ``revision``, saved as
    JSON; its ``revision`` must equal the one the run logs are read at.
    """
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise RunLogsUnavailable(f'{path}: unreadable history file: {exc}')
    if not isinstance(data, dict) or data.get('revision') != revision:
        raise RunLogsUnavailable(
            f'{path}: history is not for revision {revision}'
        )
    paths = data.get('paths')
    if not isinstance(paths, dict) or not all(
        isinstance(k, str) and isinstance(v, int) for k, v in paths.items()
    ):
        raise RunLogsUnavailable(f'{path}: "paths" must map path to count')
    return paths


def local_run_logs(runs_dir: Path, history_dir: Path | None = None) -> RunLogs:
    """Use run logs already on disk under ``runs_dir/<v1|v2>``.

    A subdirectory that is itself a git checkout supplies its own commit and
    history. Otherwise the commit is taken to be the pinned one and its
    history must come from ``history_dir``: a clone at ``<v1|v2>/`` or a file
    ``<v1|v2>.json`` (see :func:`load_history_file`). A subdirectory with no
    history raises :class:`RunLogsUnavailable`, so an offline run applies the
    same gates as a live one. A missing subdirectory leaves that set unread.
    """
    sources = {}
    for dataset_id, dataset in RUN_DATASETS.items():
        root = Path(runs_dir) / dataset.subdir
        if not root.is_dir():
            print(
                f'ARC run logs: {root} not found; {dataset_id} not joined',
                file=sys.stderr,
            )
            continue
        revision = dataset.revision
        if _is_git_root(root):
            revision = _git_head(root) or dataset.revision
            history = read_history(root, revision)
        elif history_dir is None:
            raise RunLogsUnavailable(
                f'{root} is not a git checkout, so its history is unknown; '
                'pass --arc-history-dir with a clone or history file for '
                f'{dataset.subdir}'
            )
        elif _is_git_root(Path(history_dir) / dataset.subdir):
            history = read_history(Path(history_dir) / dataset.subdir, revision)
        elif (Path(history_dir) / f'{dataset.subdir}.json').is_file():
            history = load_history_file(
                Path(history_dir) / f'{dataset.subdir}.json', revision
            )
        else:
            raise RunLogsUnavailable(
                f'{history_dir} holds neither a {dataset.subdir}/ clone nor '
                f'{dataset.subdir}.json'
            )
        sources[dataset_id] = RunLogSource(dataset, root, revision, history)
    return RunLogs(sources)


def _git(*args: str, cwd: Path | None = None, stdin: str | None = None) -> str:
    env = {**os.environ, 'GIT_LFS_SKIP_SMUDGE': '1', 'GIT_TERMINAL_PROMPT': '0'}
    try:
        completed = subprocess.run(
            ['git', *args],
            cwd=cwd,
            input=stdin,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f'git {" ".join(args)} failed ({exc.returncode}): '
            f'{(exc.stderr or "").strip()}'
        ) from exc
    return completed.stdout


def fetch_run_logs(
    model_ids: dict[str, set[str]],
    work_dir: Path,
    datasets: dict[str, RunDataset] | None = None,
) -> RunLogs:
    """Clone each run dataset and check out its wanted directories at the
    pinned commit.

    One full ``--no-checkout`` clone per dataset (a single pack, so a pinned
    commit that is no longer HEAD still checks out), a cone-mode sparse
    checkout of ``model_ids[datasetId]`` that exist at the commit, then a
    detached checkout of it. The same clone's history backs the
    rewritten-path gate.
    """
    sources = {}
    for dataset_id, dataset in (datasets or RUN_DATASETS).items():
        dest = Path(work_dir) / dataset.subdir
        _git('clone', '--quiet', '--no-checkout', dataset.url, str(dest))
        present = set(
            _git(
                'ls-tree', '-d', '--name-only', dataset.revision, cwd=dest
            ).splitlines()
        )
        wanted = sorted(model_ids.get(dataset_id, set()) & present)
        _git(
            'sparse-checkout',
            'set',
            '--cone',
            '--stdin',
            cwd=dest,
            stdin=''.join(f'{name}\n' for name in wanted),
        )
        _git('checkout', '--quiet', '--detach', dataset.revision, cwd=dest)
        raw_capture.record_git_checkout(
            dataset.url,
            dest,
            ref=dataset.revision,
            label=f'{dataset.repo_id} run logs ({len(wanted)} directories)',
        )
        sources[dataset_id] = RunLogSource(
            dataset,
            dest,
            dataset.revision,
            read_history(dest, dataset.revision),
        )
    return RunLogs(sources)
