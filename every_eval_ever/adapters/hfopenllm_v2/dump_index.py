"""Index of the Open LLM Leaderboard v2 harness result dumps.

The leaderboard publishes the lm-evaluation-harness ``results_*.json`` dump of
every run in ``open-llm-leaderboard/results``. The index states, per
``(model, precision)`` and per leaderboard benchmark, the harness settings and
sample counts every dump for that pair agrees on. A field the dumps disagree
on, or do not state, is ``None`` with a reason in ``gaps``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RESULTS_REPO = 'open-llm-leaderboard/results'
RESULTS_REVISION = 'aa81ecc38fdc5708254b833923368970efdf5ef5'
RESULTS_PATTERN = '*/results_*.json'

DTYPE_PRECISION = {
    'torch.bfloat16': 'bfloat16',
    'torch.float16': 'float16',
}


@dataclass(frozen=True)
class TaskGroup:
    """The harness tasks that make up one leaderboard benchmark."""

    prefix: str
    subtasks: int
    generative: bool = False
    suffix: str = ''

    def tasks_of(self, configs: dict[str, Any]) -> dict[str, Any]:
        """The group's tasks among a dump's ``configs``."""
        exact = not self.prefix.endswith('_')
        return {
            task: config
            for task, config in configs.items()
            if (task == self.prefix if exact else task.startswith(self.prefix))
            and task.endswith(self.suffix)
        }


# Keyed on the leaderboard API's evaluation keys.
TASK_GROUPS: dict[str, TaskGroup] = {
    'ifeval': TaskGroup('leaderboard_ifeval', 1, generative=True),
    'bbh': TaskGroup('leaderboard_bbh_', 24),
    'math': TaskGroup('leaderboard_math_', 7, generative=True, suffix='_hard'),
    'gpqa': TaskGroup('leaderboard_gpqa_', 3),
    'musr': TaskGroup('leaderboard_musr_', 3),
    'mmlu_pro': TaskGroup('leaderboard_mmlu_pro', 1),
}

CONFIG_FIELDS = ('output_type', 'num_fewshot')
DECODING_FIELDS = ('temperature', 'max_tokens', 'do_sample')

_IDENTITY_ARGS = ('pretrained', 'peft', 'delta')


@dataclass
class HarnessGroup:
    """What every dump of one (model, precision) states for one benchmark.

    The harness settings (output type, few-shot count, decoding) resolve as
    one: a conflict on any of them leaves all of them null.
    """

    output_type: str | None = None
    num_fewshot: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    do_sample: bool | None = None
    samples_number: int | None = None
    dataset_size: int | None = None
    dataset_path: str | None = None
    hf_split: str | None = None
    gaps: dict[str, str] = field(default_factory=dict)


@dataclass
class ModelEvidence:
    """Dump evidence for one (model, precision)."""

    dumps: list[str]
    groups: dict[str, HarnessGroup]
    git_hash: str | None = None
    transformers_version: str | None = None


@dataclass
class DumpIndex:
    """All dumps of a results snapshot, keyed on ``(model, precision)``."""

    entries: dict[tuple[str, str], ModelEvidence]
    files_read: int = 0
    rejected: list[dict[str, Any]] = field(default_factory=list)
    revision: str = RESULTS_REVISION
    unpinned: bool = False

    def lookup(self, model: str, precision: str | None) -> ModelEvidence | None:
        """The evidence for an exact (model, precision), if any dump names it."""
        if precision is None:
            return None
        return self.entries.get((model, precision))


def _launched_models(model_args: Any) -> dict[str, str]:
    if not isinstance(model_args, str):
        return {}
    launched = {}
    for part in model_args.split(','):
        key, sep, value = part.partition('=')
        if sep and key in _IDENTITY_ARGS:
            launched[key] = value
    return launched


def dump_identity(dump: dict[str, Any]) -> tuple[str | None, str | None]:
    """The model a dump is about, or ``None`` and the reason it is rejected.

    Identity is the dump's own ``model_name``, which must equal the weights
    the run loaded: ``peft=`` or ``delta=`` when given, else ``pretrained=``.
    The directory a dump sits in is never used.
    """
    name = dump.get('model_name')
    if not isinstance(name, str) or not name:
        return None, 'no_model_name'
    launched = _launched_models((dump.get('config') or {}).get('model_args'))
    if 'pretrained' not in launched:
        return None, 'no_pretrained'
    if 'peft' in launched and 'delta' in launched:
        return None, 'ambiguous_adapter_args'
    expected = launched.get(
        'peft', launched.get('delta', launched['pretrained'])
    )
    if name != expected:
        return None, 'identity_contradicts_model_args'
    return name, None


def _unanimous(values: list[Any]) -> tuple[bool, Any]:
    distinct = {json.dumps(v, sort_keys=True) for v in values}
    if len(distinct) != 1:
        return False, None
    return True, values[0]


def _dump_statement(
    group: TaskGroup, dump: dict[str, Any]
) -> dict[str, Any] | None:
    """One dump's statement for one group, or ``None`` if it did not run it.

    Each field maps to ``(value, None)`` or ``(None, reason)``.
    """
    tasks = group.tasks_of(dump.get('configs') or {})
    if not tasks:
        return None
    if len(tasks) != group.subtasks:
        names = CONFIG_FIELDS + (DECODING_FIELDS if group.generative else ())
        return {
            name: (None, 'incomplete_task_group')
            for name in names + ('counts', 'dataset_path', 'hf_split')
        }

    statement: dict[str, tuple[Any, str | None]] = {}

    def config_of(task_config: dict[str, Any]) -> dict[str, Any]:
        kwargs = task_config.get('generation_kwargs') or {}
        config = {
            'output_type': task_config.get('output_type'),
            'num_fewshot': task_config.get('num_fewshot'),
        }
        if group.generative:
            config['temperature'] = kwargs.get('temperature')
            config['max_tokens'] = kwargs.get('max_gen_toks')
            config['do_sample'] = kwargs.get('do_sample')
        return config

    agreed, config = _unanimous(
        [config_of(c) for _, c in sorted(tasks.items())]
    )
    for name in CONFIG_FIELDS + (DECODING_FIELDS if group.generative else ()):
        if not agreed:
            statement[name] = (None, 'subtask_conflict')
        elif config[name] is None:
            statement[name] = (None, 'not_stated')
        else:
            statement[name] = (config[name], None)

    n_samples = dump.get('n-samples')
    counts: dict[str, tuple[int, int]] = {}
    count_gap = None
    if not isinstance(n_samples, dict):
        count_gap = 'n_samples_absent'
    else:
        for task in sorted(tasks):
            entry = n_samples.get(task)
            entry = entry if isinstance(entry, dict) else {}
            original, effective = entry.get('original'), entry.get('effective')
            if not all(
                isinstance(v, int) and not isinstance(v, bool) and v >= 1
                for v in (original, effective)
            ):
                count_gap = 'sample_count_absent'
                break
            counts[task] = (original, effective)
    if count_gap:
        statement['counts'] = (None, count_gap)
    else:
        statement['counts'] = (counts, None)

    paths = {c.get('dataset_path') for c in tasks.values()}
    path = next(iter(paths))
    if len(paths) != 1:
        statement['dataset_path'] = (None, 'subtask_conflict')
    elif not isinstance(path, str) or not path:
        statement['dataset_path'] = (None, 'not_stated')
    else:
        statement['dataset_path'] = (path, None)

    splits = {c.get('test_split') for c in tasks.values()}
    split = next(iter(splits))
    if any(not isinstance(s, str) or not s for s in splits):
        statement['hf_split'] = (None, 'not_stated')
    elif len(splits) != 1:
        statement['hf_split'] = (None, 'subtask_conflict')
    else:
        statement['hf_split'] = (split, None)
    return statement


def _resolve_group(statements: list[dict[str, Any]]) -> HarnessGroup:
    resolved = HarnessGroup()
    names = (
        CONFIG_FIELDS + DECODING_FIELDS + ('counts', 'dataset_path', 'hf_split')
    )
    for name in names:
        stated = [s[name] for s in statements if name in s]
        if not stated:
            continue
        reasons = {reason for _, reason in stated}
        agreed, value = _unanimous([v for v, _ in stated])
        if reasons == {None} and agreed:
            gap = None
        elif len(reasons) == 1:
            gap = reasons.pop() or 'cross_dump_conflict'
        else:
            gap = 'cross_dump_conflict'
        if name == 'counts':
            if gap:
                resolved.gaps['samples_number'] = gap
                continue
            original = sum(o for o, _ in value.values())
            effective = sum(e for _, e in value.values())
            resolved.samples_number = effective
            if effective != original:
                resolved.dataset_size = original
            continue
        if gap:
            resolved.gaps[name] = gap
        else:
            setattr(resolved, name, value)
    config_names = CONFIG_FIELDS + DECODING_FIELDS
    conflict = next(
        (
            resolved.gaps[name]
            for name in config_names
            if resolved.gaps.get(name, '').endswith('conflict')
        ),
        None,
    )
    if conflict:
        for name in config_names:
            if getattr(resolved, name) is not None or name in resolved.gaps:
                setattr(resolved, name, None)
                resolved.gaps[name] = conflict
    return resolved


def build_dump_index(results_dir: str | Path) -> DumpIndex:
    """Read every ``results_*.json`` under ``results_dir`` into a ``DumpIndex``."""
    root = Path(results_dir)
    index = DumpIndex(entries={})
    by_key: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = (
        defaultdict(list)
    )
    for path in sorted(root.rglob('results_*.json')):
        index.files_read += 1
        rel = path.relative_to(root).as_posix()
        try:
            dump = json.loads(path.read_text(encoding='utf-8'))
        except ValueError as exc:
            index.rejected.append(
                {'path': rel, 'reason': 'unreadable', 'detail': str(exc)}
            )
            continue
        if not isinstance(dump, dict):
            index.rejected.append({'path': rel, 'reason': 'not_an_object'})
            continue
        model, reason = dump_identity(dump)
        config = dump.get('config') or {}
        if model is None:
            index.rejected.append(
                {
                    'path': rel,
                    'reason': reason,
                    'model_name': dump.get('model_name'),
                    'model_args': config.get('model_args'),
                }
            )
            continue
        precision = DTYPE_PRECISION.get(config.get('model_dtype'))
        if precision is None:
            index.rejected.append(
                {
                    'path': rel,
                    'reason': 'unmapped_model_dtype',
                    'model_dtype': config.get('model_dtype'),
                }
            )
            continue
        by_key[(model, precision)].append((rel, dump))

    for key, dumps in sorted(by_key.items()):
        groups = {}
        for eval_key, group in TASK_GROUPS.items():
            statements = [
                s
                for s in (_dump_statement(group, dump) for _, dump in dumps)
                if s is not None
            ]
            if statements:
                groups[eval_key] = _resolve_group(statements)
        evidence = ModelEvidence(dumps=[rel for rel, _ in dumps], groups=groups)
        for name in ('git_hash', 'transformers_version'):
            values = {dump.get(name) for _, dump in dumps}
            value = next(iter(values))
            if len(values) == 1 and isinstance(value, str) and value:
                setattr(evidence, name, value)
        index.entries[key] = evidence
    return index
