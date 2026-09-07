"""Tests for the Hub-backed flat rebuild (cron.flat_rebuild)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import pytest
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from huggingface_hub.errors import EntryNotFoundError

from every_eval_ever.cron import flat_rebuild as fr
from every_eval_ever.cron import store

NOW = datetime(2026, 9, 6, 12, 0, 0, tzinfo=UTC)
OLD = NOW - timedelta(days=100)
OLDER = NOW - timedelta(days=101)
RECENT = NOW - timedelta(days=1)
SCHEMA = '0.2.2'


def record_bytes(uuid: str, benchmark: str) -> bytes:
    return json.dumps(
        {
            'schema_version': SCHEMA,
            'evaluation_id': f'{benchmark}/dev/model/{uuid}',
            'model_info': {'id': 'dev/model', 'developer': 'dev'},
            'evaluation_results': [
                {
                    'evaluation_name': benchmark,
                    'metric_config': {'metric_id': 'accuracy'},
                    'score_details': {'score': 1},
                }
            ],
        }
    ).encode()


@dataclass
class FakeRepoFile:
    path: str
    size: int


class FakeApi:
    """In-memory HfApi: enough surface for flat_rebuild, no network."""

    def __init__(self, files: dict[str, bytes], tmp_path: Path) -> None:
        self.files = dict(files)
        self.downloads = tmp_path / 'downloads'
        self.commits: list[str] = []
        self.commit_ops: list[list[Any]] = []
        self.fail_next = 0

    def list_repo_tree(self, repo_id: str, repo_type: str, recursive: bool):
        for path in sorted(self.files):
            yield FakeRepoFile(path, len(self.files[path]))

    def list_repo_files(
        self, repo_id: str, repo_type: str, revision: str | None = None
    ) -> list[str]:
        return sorted(self.files)

    def hf_hub_download(
        self, repo_id: str, repo_type: str, filename: str
    ) -> str:
        if filename not in self.files:
            raise EntryNotFoundError(filename)
        target = self.downloads / filename.replace('/', '_')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[filename])
        return str(target)

    def create_commit(
        self,
        repo_id: str,
        repo_type: str,
        operations: list[Any],
        commit_message: str,
        commit_description: str | None = None,
    ) -> None:
        if self.fail_next > 0:
            self.fail_next -= 1
            raise RuntimeError('simulated hub error')
        self.commits.append(commit_message)
        self.commit_ops.append(list(operations))
        for operation in operations:
            if isinstance(operation, CommitOperationAdd):
                data = operation.path_or_fileobj
                if isinstance(data, str):
                    data = Path(data).read_bytes()
                self.files[operation.path_in_repo] = bytes(data)
            elif isinstance(operation, CommitOperationDelete):
                self.files.pop(operation.path_in_repo, None)


def committed_paths(api: FakeApi) -> set[str]:
    return {
        operation.path_in_repo
        for operations in api.commit_ops
        for operation in operations
    }


def seed_manifest_files(
    files: dict[str, bytes], rows: list[fr.Row], created_at: str
) -> dict[str, Any]:
    """Write a snapshot's manifest and entries without moving the pointer."""
    manifest = fr.manifest_for(rows, created_at=created_at)
    files[manifest['entries_path']] = fr.jsonl_text(rows).encode()
    files[manifest['manifest_path']] = (
        json.dumps(manifest, indent=2, sort_keys=True) + '\n'
    ).encode()
    return manifest


def seed_pointer(files: dict[str, bytes], manifest: dict[str, Any]) -> None:
    files[fr.LATEST_MANIFEST_PATH] = (
        json.dumps(manifest, indent=2, sort_keys=True) + '\n'
    ).encode()


def seed_derived(files: dict[str, bytes], rows: list[fr.Row]) -> None:
    """Write the index files a published snapshot would have, so a run
    whose rows are unchanged is a clean no-op."""
    ordered = sorted(rows, key=lambda row: row.legacy_path)
    files[fr.BY_LEGACY_PATH] = fr.jsonl_text(ordered).encode()
    by_benchmark: dict[str, list[fr.Row]] = {}
    for row in ordered:
        by_benchmark.setdefault(row.benchmark, []).append(row)
    for benchmark, brows in by_benchmark.items():
        files[f'{fr.INDEXES_PREFIX}/{benchmark}.jsonl'] = fr.jsonl_text(
            brows
        ).encode()


def make_row(uuid: str, benchmark: str, *, legacy: str | None = None) -> fr.Row:
    path = legacy or f'data/{benchmark}/dev/model/{uuid}.json'
    data = record_bytes(uuid, benchmark)
    return fr.Row(
        object_uuid=uuid,
        object_path=fr.object_path_for(uuid, suffix='.json'),
        sha256=fr.sha256_bytes(data),
        size_bytes=len(data),
        legacy_path=path,
        benchmark=benchmark,
        eval_schema_version=SCHEMA,
    )


def seed_datastore(
    files: dict[str, bytes], benchmarks: dict[str, int]
) -> list[fr.Row]:
    """Populate data/ records and a snapshot one record behind them."""
    rows: list[fr.Row] = []
    for benchmark, count in benchmarks.items():
        for _ in range(count):
            uuid = str(uuid4())
            path = f'data/{benchmark}/dev/model/{uuid}.json'
            files[path] = record_bytes(uuid, benchmark)
            rows.append(make_row(uuid, benchmark))
    manifest = seed_manifest_files(files, rows[:-1], RECENT.isoformat())
    seed_pointer(files, manifest)
    return rows


# -- row and manifest shapes -----------------------------------------------


def test_row_round_trip() -> None:
    row = make_row(str(uuid4()), 'gsm8k')
    assert fr.Row.from_dict(row.to_dict()) == row


def test_row_dict_omits_absent_samples() -> None:
    row = make_row(str(uuid4()), 'gsm8k')
    assert 'instance_level_path' not in row.to_dict()


def test_manifest_shape_matches_builder() -> None:
    rows = [make_row(str(uuid4()), 'gsm8k')]
    manifest = fr.manifest_for(rows, created_at='2026-01-01T00:00:00+00:00')
    assert manifest['aggregate_file_count'] == 1
    assert manifest['benchmark_count'] == 1
    assert manifest['total_file_count'] == 1
    assert manifest['source'] == {'type': 'legacy_data_tree', 'path': 'data'}
    core = fr.manifest_core(manifest)
    assert 'created_at' not in core
    assert 'entries_path' not in core
    assert (
        fr.sha256_bytes(fr.stable_json_bytes(core))
        == (manifest['manifest_core_sha256'])
    )


def test_manifest_ignores_created_at() -> None:
    rows = [make_row(str(uuid4()), 'gsm8k')]
    one = fr.manifest_for(rows, created_at='2026-01-01T00:00:00+00:00')
    two = fr.manifest_for(rows, created_at='2027-01-01T00:00:00+00:00')
    assert fr.manifest_core(one) == fr.manifest_core(two)
    assert one['manifest_core_sha256'] == two['manifest_core_sha256']


# -- diff_against_snapshot ---------------------------------------------------


def test_diff_finds_new_removed_and_orphans() -> None:
    kept = str(uuid4())
    gone = str(uuid4())
    fresh = str(uuid4())
    old_rows = [make_row(kept, 'gsm8k'), make_row(gone, 'gone_coll')]
    listing = {
        f'data/gsm8k/dev/model/{kept}.json': 10,
        f'data/gsm8k/dev/model/{fresh}.json': 10,
        f'data/gsm8k/dev/model/{fresh}_samples.jsonl': 10,
        f'data/gone_coll/dev/{gone}_samples.jsonl': 10,
    }
    diff = fr.diff_against_snapshot(old_rows, listing)
    assert len(diff.new_paths) == 1
    assert diff.new_paths[0].endswith(f'{fresh}.json')
    assert diff.new_sample_paths[0].endswith(f'{fresh}_samples.jsonl')
    assert [row.object_uuid for row in diff.removed_rows] == [gone]
    assert list(diff.orphan_samples) == [
        f'data/gone_coll/dev/{gone}_samples.jsonl'
    ]


def test_diff_size_equal_is_not_new() -> None:
    row = make_row(str(uuid4()), 'gsm8k')
    diff = fr.diff_against_snapshot([row], {row.legacy_path: row.size_bytes})
    assert diff.new_paths == ()
    assert diff.removed_rows == ()


# -- build_rows ---------------------------------------------------------------


def test_build_new_aggregate_with_samples() -> None:
    uuid = str(uuid4())
    path = f'data/gsm8k/dev/model/{uuid}.json'
    samples_path = f'data/gsm8k/dev/model/{uuid}_samples.jsonl'
    contents = {path: record_bytes(uuid, 'gsm8k'), samples_path: b'{"x": 1}'}
    diff = fr.diff_against_snapshot(
        [], {path: len(contents[path]), samples_path: 8}
    )
    result = fr.build_rows([], diff, contents)
    assert not result.errors
    assert result.added == 1
    row = result.rows[0]
    assert row.instance_level_available
    assert row.instance_level_path == fr.object_path_for(
        uuid, suffix='_samples.jsonl'
    )
    assert uuid in result.upload_aggregates
    assert uuid in result.upload_samples


def test_build_move_repoints_without_uploads() -> None:
    uuid = str(uuid4())
    old_path = f'data/gsm8k/dev/model/{uuid}.json'
    new_path = f'data/gsm8k/other/{uuid}.json'
    data = record_bytes(uuid, 'gsm8k')
    old_row = make_row(uuid, 'gsm8k', legacy=old_path)
    diff = fr.diff_against_snapshot([old_row], {new_path: len(data)})
    result = fr.build_rows([old_row], diff, {new_path: data})
    assert not result.errors
    assert result.moved == 1
    assert result.added == 0
    assert not result.upload_aggregates
    assert result.rows[0].legacy_path == new_path


def test_same_uuid_semantic_change_becomes_conflict() -> None:
    old_row = make_row(str(uuid4()), 'gsm8k')
    changed = json.loads(record_bytes(old_row.object_uuid, 'gsm8k'))
    changed['evaluation_results'][0]['score_details']['score'] = 100
    contents = {old_row.legacy_path: json.dumps(changed).encode()}
    listing = {old_row.legacy_path: len(contents[old_row.legacy_path])}
    diff = fr.diff_against_snapshot([old_row], listing)
    old_object = record_bytes(old_row.object_uuid, 'gsm8k')
    result = fr.build_rows(
        [old_row],
        diff,
        contents,
        lambda object_path: (
            old_object if object_path == old_row.object_path else None
        ),
    )
    assert not result.errors
    assert len(result.conflicts) == 1
    assert 'immutable' in result.conflicts[0]
    assert result.excluded_paths == {old_row.legacy_path}
    assert result.rows[0] == old_row  # the published object stays
    assert not result.upload_aggregates

    # without the existing object there is nothing to compare against
    result = fr.build_rows([old_row], diff, contents)
    assert 'could not be fetched' in result.errors[0]


def test_reserialized_move_is_accepted() -> None:
    old_row = make_row(str(uuid4()), 'gsm8k')
    new_path = old_row.legacy_path.replace('/gsm8k/', '/other/')
    parsed = json.loads(record_bytes(old_row.object_uuid, 'gsm8k'))
    reserialized = json.dumps(parsed, indent=2, sort_keys=True).encode()
    diff = fr.diff_against_snapshot([old_row], {new_path: len(reserialized)})
    result = fr.build_rows(
        [old_row],
        diff,
        {new_path: reserialized},
        lambda object_path: (
            record_bytes(old_row.object_uuid, 'gsm8k')
            if object_path == old_row.object_path
            else None
        ),
    )
    assert not result.errors
    assert result.moved == 1
    assert not result.upload_aggregates
    assert result.rows[0].legacy_path == new_path
    assert result.rows[0].sha256 == old_row.sha256


def test_changed_in_place_reserialization_is_kept() -> None:
    old_row = make_row(str(uuid4()), 'gsm8k')
    parsed = json.loads(record_bytes(old_row.object_uuid, 'gsm8k'))
    reserialized = json.dumps(parsed, indent=2).encode()
    diff = fr.diff_against_snapshot(
        [old_row], {old_row.legacy_path: len(reserialized)}
    )
    result = fr.build_rows(
        [old_row],
        diff,
        {old_row.legacy_path: reserialized},
        lambda object_path: (
            record_bytes(old_row.object_uuid, 'gsm8k')
            if object_path == old_row.object_path
            else None
        ),
    )
    assert not result.errors
    assert result.rows[0] == old_row
    assert not result.upload_aggregates


def test_build_rejects_invalid_uuid_and_bad_json() -> None:
    bad_path = 'data/coll/dev/model/not-a-uuid.json'
    diff = fr.diff_against_snapshot([], {bad_path: 2})
    result = fr.build_rows([], diff, {bad_path: b'{}'})
    assert any('valid UUID' in error for error in result.errors)

    uuid = str(uuid4())
    path = f'data/coll/dev/model/{uuid}.json'
    diff = fr.diff_against_snapshot([], {path: 2})
    result = fr.build_rows([], diff, {path: b'[]'})
    assert any('object' in error for error in result.errors)


# -- plan_retire ----------------------------------------------------------------


def test_retire_moves_stale_and_keeps_live() -> None:
    listing = {
        'flat/indexes/by_collection/live.jsonl': 10,
        'flat/indexes/by_collection/dead.jsonl': 10,
        'flat/indexes/by_collection/legacy/aggregate.jsonl': 10,
    }
    moves = dict(fr.plan_retire(listing, {'live'}, today='20260906'))
    assert moves['flat/indexes/by_collection/dead.jsonl'] == (
        'flat/indexes/retired/dead.jsonl'
    )
    assert moves['flat/indexes/by_collection/legacy/aggregate.jsonl'] == (
        'flat/indexes/retired/legacy/aggregate.jsonl'
    )
    assert all('live' not in source for source in moves)


def test_retire_suffixes_on_collision() -> None:
    listing = {
        'flat/indexes/by_collection/dead.jsonl': 10,
        'flat/indexes/retired/dead.jsonl': 10,
    }
    moves = fr.plan_retire(listing, set(), today='20260906')
    assert moves[0][1] == 'flat/indexes/retired/dead.20260906.jsonl'


# -- plan_retention ---------------------------------------------------------------


def manifest_info(created_at: datetime, name: str) -> fr.ManifestInfo:
    digest = fr.sha256_bytes(name.encode())
    dir_path = f'flat/manifests/sha256_{digest}'
    return fr.ManifestInfo(
        manifest_path=f'{dir_path}/manifest.json',
        entries_path=f'{dir_path}/entries.jsonl',
        created_at=created_at.isoformat(),
        dir_path=dir_path,
    )


def test_retention_trims_only_fully_covered() -> None:
    live_uuid = str(uuid4())
    dead_uuid = str(uuid4())
    current = manifest_info(NOW, 'current')
    within = manifest_info(RECENT, 'within')
    trimmable = manifest_info(OLD, 'trim')
    pinned = manifest_info(OLDER, 'pin')
    uuids = {
        current.dir_path: {live_uuid},
        within.dir_path: {live_uuid},
        trimmable.dir_path: {live_uuid},
        pinned.dir_path: {dead_uuid, live_uuid},
    }
    plan = fr.plan_retention(
        [current, within, trimmable, pinned],
        new_uuids={live_uuid},
        current_manifest_path=current.manifest_path,
        now=NOW,
        retain_days=90,
        keep_newest=2,
        scan_budget=10,
        uuids_of=lambda info: uuids[info.dir_path],
    )
    assert plan.delete_dirs == (trimmable.dir_path,)
    assert plan.pinned == (pinned.dir_path,)
    assert plan.unscanned == ()


def test_retention_budget_keeps_unscanned() -> None:
    live_uuid = str(uuid4())
    current = manifest_info(NOW, 'current')
    old = [manifest_info(OLD - timedelta(days=i), f'm{i}') for i in range(3)]
    uuids = {current.dir_path: {live_uuid}} | {
        info.dir_path: {live_uuid} for info in old
    }
    plan = fr.plan_retention(
        [current, *old],
        new_uuids={live_uuid},
        current_manifest_path=current.manifest_path,
        now=NOW,
        retain_days=90,
        keep_newest=1,
        scan_budget=1,
        uuids_of=lambda info: uuids[info.dir_path],
    )
    assert len(plan.delete_dirs) == 1
    assert len(plan.unscanned) == 2


def test_retention_keeps_unreadable_snapshots() -> None:
    live_uuid = str(uuid4())
    current = manifest_info(NOW, 'current')
    broken = manifest_info(OLD, 'broken')

    def boom(info: fr.ManifestInfo) -> set[str]:
        raise RuntimeError('download failed')

    with pytest.warns(
        UserWarning, match='Keeping unreadable snapshot.*download failed'
    ):
        plan = fr.plan_retention(
            [current, broken],
            new_uuids={live_uuid},
            current_manifest_path=current.manifest_path,
            now=NOW,
            retain_days=90,
            keep_newest=1,
            scan_budget=10,
            uuids_of=boom,
        )
    assert plan.delete_dirs == ()
    assert broken.dir_path in plan.unscanned


# -- batch_units and publishing ----------------------------------------------------


def test_batch_units_never_splits() -> None:
    units = [[1] * 2, [2] * 3, [3] * 4]
    batches = fr.batch_units(units, 4)
    flattened = [item for batch in batches for item in batch]
    assert flattened == [1, 1, 2, 2, 2, 3, 3, 3, 3]
    for batch in batches:
        assert len(batch) <= 4


def _fake_add(n: int) -> CommitOperationAdd:
    return CommitOperationAdd(
        path_in_repo=f'flat/objects/{n}.json', path_or_fileobj=b'{}'
    )


def test_publisher_batches_with_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, 'wait_before_retry', lambda attempt: None)
    api = FakeApi({}, tmp_path)
    committed = fr.FlatPublisher(api, 'org/ds', batch_size=2).publish(
        [[_fake_add(n)] for n in range(5)],
        message='cron: flat rebuild 2026-09-06 (5 aggregate record(s))',
        description='d',
    )
    assert committed == 3
    assert api.commits[0].endswith('(1/3)')
    assert api.commits[-1].endswith('(3/3)')


def test_publisher_adopts_landed_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, 'wait_before_retry', lambda attempt: None)
    add = _fake_add(1)
    api = FakeApi({add.path_in_repo: b'{}'}, tmp_path)
    api.fail_next = 1
    committed = fr.FlatPublisher(api, 'org/ds').publish(
        [[add]], message='m', description='d'
    )
    assert committed == 1


def test_publisher_raises_when_landing_unverifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, 'wait_before_retry', lambda attempt: None)
    api = FakeApi({}, tmp_path)
    api.fail_next = 99

    def broken_listing(**kwargs: Any) -> list[str]:
        raise RuntimeError('hub unreachable')

    api.list_repo_files = broken_listing  # type: ignore[method-assign]
    with pytest.raises(fr.FlatRebuildError):
        fr.FlatPublisher(api, 'org/ds').publish(
            [[_fake_add(1)]], message='m', description='d'
        )


# -- orchestration end-to-end --------------------------------------------------------


def test_verify_rows_clean(tmp_path: Path) -> None:
    row = make_row(str(uuid4()), 'gsm8k')
    data = record_bytes(row.object_uuid, 'gsm8k')
    files = {row.legacy_path: data, row.object_path: data}
    api = FakeApi(files, tmp_path)
    checked, reserialized, drift = fr.verify_rows(
        api, 'org/ds', [row], {p: len(v) for p, v in files.items()}
    )
    assert (checked, reserialized, drift) == (1, 0, [])


def test_verify_rows_catches_same_length_drift(tmp_path: Path) -> None:
    row = make_row(str(uuid4()), 'gsm8k')
    parsed = json.loads(record_bytes(row.object_uuid, 'gsm8k'))
    parsed['evaluation_results'][0]['score_details']['score'] = 2
    mutated = json.dumps(parsed).encode()  # same length, different data
    assert len(mutated) == row.size_bytes
    files = {
        row.legacy_path: mutated,
        row.object_path: record_bytes(row.object_uuid, 'gsm8k'),
    }
    api = FakeApi(files, tmp_path)
    checked, reserialized, drift = fr.verify_rows(
        api, 'org/ds', [row], {p: len(v) for p, v in files.items()}
    )
    assert checked == 1 and reserialized == 0
    assert len(drift) == 1 and row.legacy_path in drift[0]


def test_verify_rows_accepts_reserialization(tmp_path: Path) -> None:
    row = make_row(str(uuid4()), 'gsm8k')
    parsed = json.loads(record_bytes(row.object_uuid, 'gsm8k'))
    reserialized = json.dumps(parsed, indent=2, sort_keys=True).encode()
    files = {
        row.legacy_path: reserialized,
        row.object_path: record_bytes(row.object_uuid, 'gsm8k'),
    }
    api = FakeApi(files, tmp_path)
    checked, reserialized_count, drift = fr.verify_rows(
        api, 'org/ds', [row], {p: len(v) for p, v in files.items()}
    )
    assert (checked, reserialized_count, drift) == (1, 1, [])


def test_verify_rows_reports_missing_inherited_objects(tmp_path: Path) -> None:
    row = make_row(str(uuid4()), 'gsm8k')
    files = {row.legacy_path: record_bytes(row.object_uuid, 'gsm8k')}
    api = FakeApi(files, tmp_path)
    checked, reserialized, drift = fr.verify_rows(
        api, 'org/ds', [row], {row.legacy_path: row.size_bytes}
    )
    assert (checked, reserialized) == (1, 0)
    assert len(drift) == 1
    assert 'inherited file is missing' in drift[0]


def test_orchestrate_refuses_bootstrap(tmp_path: Path) -> None:
    api = FakeApi({}, tmp_path)
    with pytest.raises(fr.FlatRebuildError, match='allow-bootstrap'):
        fr.orchestrate(api, 'org/ds', now=NOW)


def test_orchestrate_rebuild_then_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, 'wait_before_retry', lambda attempt: None)
    files: dict[str, bytes] = {}
    seed_datastore(files, {'gsm8k': 2, 'hle': 1})
    api = FakeApi(files, tmp_path)

    report = fr.orchestrate(api, 'org/ds', allow_bootstrap=True, now=NOW)
    assert not report.noop
    assert report.records == 3
    assert report.collections == 2
    assert report.commits >= 1
    latest = json.loads(api.files[fr.LATEST_MANIFEST_PATH])
    assert latest['aggregate_file_count'] == 3
    assert 'flat/indexes/by_collection/gsm8k.jsonl' in api.files
    assert 'flat/indexes/by_collection/hle.jsonl' in api.files
    assert 'flat/indexes/by_legacy_path.jsonl' in api.files
    assert any(
        message.startswith(
            'cron: flat rebuild 2026-09-06 (3 aggregate record(s))'
        )
        for message in api.commits
    )

    commits_before = len(api.commits)
    report = fr.orchestrate(api, 'org/ds', now=NOW)
    assert report.noop
    assert len(api.commits) == commits_before


def test_orchestrate_picks_up_new_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, 'wait_before_retry', lambda attempt: None)
    files: dict[str, bytes] = {}
    seed_datastore(files, {'gsm8k': 2})
    api = FakeApi(files, tmp_path)
    fr.orchestrate(api, 'org/ds', allow_bootstrap=True, now=NOW)

    uuid = str(uuid4())
    api.files[f'data/gsm8k/dev/model/{uuid}.json'] = record_bytes(uuid, 'gsm8k')
    commits_before = len(api.commits)
    report = fr.orchestrate(api, 'org/ds', now=NOW)
    assert not report.noop
    assert report.added_records == 1
    assert len(api.commits) > commits_before
    committed = committed_paths(api)
    assert fr.object_path_for(uuid, suffix='.json') in committed
    assert 'flat/indexes/by_collection/gsm8k.jsonl' in committed
    assert 'flat/indexes/by_legacy_path.jsonl' in committed


def test_orchestrate_dry_run_commits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, 'wait_before_retry', lambda attempt: None)
    files: dict[str, bytes] = {}
    seed_datastore(files, {'gsm8k': 1})
    api = FakeApi(files, tmp_path)
    pointer_before = api.files[fr.LATEST_MANIFEST_PATH]
    report = fr.orchestrate(
        api, 'org/ds', allow_bootstrap=True, dry_run=True, now=NOW
    )
    assert report.commits == 0
    assert report.uploads > 0
    assert api.files[fr.LATEST_MANIFEST_PATH] == pointer_before


def test_orchestrate_retires_stale_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, 'wait_before_retry', lambda attempt: None)
    files: dict[str, bytes] = {}
    seed_datastore(files, {'gsm8k': 1})
    files['flat/indexes/by_collection/live_bench.jsonl'] = b'{"old": true}\n'
    api = FakeApi(files, tmp_path)
    report = fr.orchestrate(api, 'org/ds', allow_bootstrap=True, now=NOW)
    assert 'flat/indexes/by_collection/live_bench.jsonl' in (
        report.indexes_retired
    )
    assert 'flat/indexes/retired/live_bench.jsonl' in api.files
    assert 'flat/indexes/by_collection/live_bench.jsonl' not in api.files


def test_orchestrate_trims_and_pins_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store, 'wait_before_retry', lambda attempt: None)
    files: dict[str, bytes] = {}
    seed_datastore(files, {'gsm8k': 2})
    api = FakeApi(files, tmp_path)
    fr.orchestrate(api, 'org/ds', allow_bootstrap=True, now=NOW, keep_newest=1)
    live_rows = list(fr.read_snapshot(api, 'org/ds')[1])

    # A snapshot holding a record that no longer exists in data/ is the
    # last index into that record and must be pinned; one whose records
    # are all still live may be trimmed. Snapshots need distinct rows,
    # since identical ones dedupe into the same content-addressed dir.
    dead_uuid = str(uuid4())
    dead_row = make_row(
        dead_uuid, 'gsm8k', legacy=f'data/gsm8k/dev/model/{dead_uuid}.json'
    )
    pinned = seed_manifest_files(
        api.files, [*live_rows, dead_row], OLD.isoformat()
    )
    trimmed = seed_manifest_files(api.files, [live_rows[0]], OLDER.isoformat())

    report = fr.orchestrate(api, 'org/ds', now=NOW, keep_newest=1)
    trimmed_dir = PurePosixPath(trimmed['manifest_path']).parent.as_posix()
    pinned_dir = PurePosixPath(pinned['manifest_path']).parent.as_posix()
    assert trimmed_dir in report.manifests_trimmed
    assert pinned_dir in report.manifests_pinned
    assert pinned['entries_path'] in api.files
    assert trimmed['entries_path'] not in api.files


@pytest.mark.parametrize('env_value', ['', 'org/custom'])
def test_cli_repository_environment(monkeypatch, env_value, capsys):
    monkeypatch.setenv('EEE_DATASTORE_REPO_ID', env_value)
    received = []

    def capture(api, repo_id, **kwargs):
        received.append(repo_id)
        return fr.RebuildReport(repo_id=repo_id, noop=True)

    monkeypatch.setattr(fr, 'orchestrate', capture)
    assert fr.main([]) == 0
    expected = env_value or fr.DEFAULT_DATASTORE_REPO
    assert received == [expected]
    assert expected in capsys.readouterr().out


def test_cli_explicit_empty_repository_fails(capsys):
    assert fr.main(['--repo-id', '']) == 1
    assert 'repo_id must not be empty' in capsys.readouterr().err


@pytest.mark.parametrize('path', [fr.LATEST_MANIFEST_PATH, fr.BY_LEGACY_PATH])
def test_publisher_does_not_adopt_failed_overwrite(tmp_path, path):
    api = FakeApi({path: b'old'}, tmp_path)
    api.fail_next = 1
    with pytest.raises(fr.FlatRebuildError, match='could not commit'):
        fr.FlatPublisher(api, 'org/ds').publish(
            [[CommitOperationAdd(path_in_repo=path, path_or_fileobj=b'new')]],
            message='m',
            description='d',
        )
    assert api.files[path] == b'old'


def test_publisher_retries_conflict_on_existing_path(tmp_path, monkeypatch):
    api = FakeApi({fr.LATEST_MANIFEST_PATH: b'old'}, tmp_path)
    api.fail_next = 1
    monkeypatch.setattr(store, 'is_commit_conflict', lambda exc: True)
    monkeypatch.setattr(store, 'wait_before_retry', lambda attempt: None)
    fr.FlatPublisher(api, 'org/ds').publish(
        [
            [
                CommitOperationAdd(
                    path_in_repo=fr.LATEST_MANIFEST_PATH, path_or_fileobj=b'new'
                )
            ]
        ],
        message='m',
        description='d',
    )
    assert api.files[fr.LATEST_MANIFEST_PATH] == b'new'


def test_publisher_adopts_actual_landed_overwrite(tmp_path):
    api = FakeApi({fr.LATEST_MANIFEST_PATH: b'old'}, tmp_path)
    commit = api.create_commit

    def landed(**kwargs):
        commit(**kwargs)
        raise TimeoutError('response lost')

    api.create_commit = landed
    fr.FlatPublisher(api, 'org/ds').publish(
        [
            [
                CommitOperationAdd(
                    path_in_repo=fr.LATEST_MANIFEST_PATH, path_or_fileobj=b'new'
                )
            ]
        ],
        message='m',
        description='d',
    )
    assert api.files[fr.LATEST_MANIFEST_PATH] == b'new'
    assert len(api.commits) == 1


def test_return_to_historical_snapshot_keeps_pointer_target(tmp_path):
    a, b, c = [make_row(str(uuid4()), 'gsm8k') for _ in range(3)]
    files = {a.legacy_path: record_bytes(a.object_uuid, 'gsm8k')}
    target = seed_manifest_files(files, [a], OLD.isoformat())
    seed_manifest_files(files, [a, b], RECENT.isoformat())
    seed_pointer(files, seed_manifest_files(files, [a, b, c], NOW.isoformat()))
    api = FakeApi(files, tmp_path)
    report = fr.orchestrate(api, 'org/ds', now=NOW)
    latest = json.loads(api.files[fr.LATEST_MANIFEST_PATH])
    assert latest['entries_path'] == target['entries_path']
    assert latest['entries_path'] in api.files
    assert latest['manifest_path'] in api.files
    assert (
        str(PurePosixPath(latest['manifest_path']).parent)
        not in report.manifests_trimmed
    )


@pytest.mark.parametrize('changed', [False, True])
@pytest.mark.parametrize('companion', [False, True])
def test_historical_objects_are_never_overwritten(tmp_path, changed, companion):
    row = make_row(str(uuid4()), 'gsm8k')
    data = record_bytes(row.object_uuid, 'gsm8k')
    sample = b'{"x":1}\n'
    if companion:
        row = fr._row_with_samples(row, sample)
    destination = row.instance_level_path if companion else row.object_path
    files = {row.legacy_path: data, row.object_path: data}
    if companion:
        files[fr.samples_path_for(row.legacy_path)] = sample
        files[destination] = sample
    original = files[destination]
    if changed:
        source = (
            fr.samples_path_for(row.legacy_path)
            if companion
            else row.legacy_path
        )
        parsed = json.loads(files[source])
        if companion:
            parsed['x'] = 100
        else:
            parsed['evaluation_results'][0]['score_details']['score'] = 100
        files[source] = json.dumps(parsed).encode()
    seed_manifest_files(files, [row], OLD.isoformat())
    seed_pointer(files, seed_manifest_files(files, [], RECENT.isoformat()))
    seed_derived(files, [row])
    api = FakeApi(files, tmp_path)
    report = fr.orchestrate(api, 'org/ds', now=NOW)
    if changed:
        assert report.conflicts
    else:
        assert not report.conflicts
    # a published object is adopted or conflicted, never overwritten
    assert destination not in committed_paths(api)
    assert api.files[destination] == original


def test_pre_flight_reports_every_immutable_mismatch(tmp_path):
    bad_agg = make_row(str(uuid4()), 'gsm8k')
    bad_sample = fr._row_with_samples(
        make_row(str(uuid4()), 'hle'), b'{"x":1}\n'
    )
    files = {
        bad_agg.legacy_path: record_bytes(bad_agg.object_uuid, 'gsm8k'),
        bad_agg.object_path: b'{}',
        bad_sample.legacy_path: record_bytes(bad_sample.object_uuid, 'hle'),
        fr.samples_path_for(bad_sample.legacy_path): b'{"x":1}\n',
        bad_sample.instance_level_path: b'{}',
    }
    seed_pointer(files, seed_manifest_files(files, [], RECENT.isoformat()))
    api = FakeApi(files, tmp_path)
    report = fr.orchestrate(api, 'org/ds', now=NOW)
    message = '\n'.join(report.conflicts)
    assert bad_agg.legacy_path in message
    assert fr.samples_path_for(bad_sample.legacy_path) in message
    # conflicting destinations keep their bytes; the clean record publishes
    assert api.files[bad_agg.object_path] == b'{}'
    assert api.files[bad_sample.instance_level_path] == b'{}'
    assert fr.exit_code_for(report) == 2


@pytest.mark.parametrize('change', ['add', 'remove', 'edit'])
def test_existing_aggregate_companion_changes(tmp_path, change):
    row = make_row(str(uuid4()), 'gsm8k')
    sample = b'{"x":1}\n'
    if change != 'add':
        row = fr._row_with_samples(row, sample)
    files = {row.legacy_path: record_bytes(row.object_uuid, 'gsm8k')}
    seed_pointer(files, seed_manifest_files(files, [row], RECENT.isoformat()))
    sample_path = fr.samples_path_for(row.legacy_path)
    if change != 'add':
        files[row.instance_level_path] = sample
    if change != 'remove':
        files[sample_path] = sample if change != 'edit' else b'{"x":2000}\n'
    seed_derived(files, [row])
    api = FakeApi(files, tmp_path)
    if change == 'edit':
        report = fr.orchestrate(api, 'org/ds', now=NOW)
        assert report.conflicts
        assert not api.commits
    else:
        report = fr.orchestrate(api, 'org/ds', now=NOW)
        assert not report.noop
        rebuilt = fr.read_snapshot(api, 'org/ds')[1][0]
        assert rebuilt.instance_level_available == (change == 'add')
        if change == 'add':
            assert api.files[rebuilt.instance_level_path] == sample
        else:
            assert rebuilt.instance_level_path is None


def test_retention_preserves_last_companion_reference(tmp_path):
    a, b, c = [make_row(str(uuid4()), 'gsm8k') for _ in range(3)]
    files = {
        r.legacy_path: record_bytes(r.object_uuid, 'gsm8k') for r in [a, b, c]
    }
    historical = seed_manifest_files(
        files, [fr._row_with_samples(a, b'{}')], OLD.isoformat()
    )
    seed_manifest_files(files, [a, b], RECENT.isoformat())
    seed_pointer(files, seed_manifest_files(files, [a, b, c], NOW.isoformat()))
    api = FakeApi(files, tmp_path)
    report = fr.orchestrate(api, 'org/ds', now=NOW)
    assert (
        str(PurePosixPath(historical['manifest_path']).parent)
        in report.manifests_pinned
    )
    assert historical['entries_path'] in api.files


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('same_content', [False, True])
def test_duplicate_uuid_paths_are_rejected(existing, same_content):
    row = make_row(str(uuid4()), 'gsm8k')
    other = row.legacy_path.replace('/gsm8k/', '/other/')
    data = record_bytes(row.object_uuid, 'gsm8k')
    contents = {
        row.legacy_path: data,
        other: data if same_content else data + b' ',
    }
    old = [row] if existing else []
    diff = fr.diff_against_snapshot(
        old, {p: len(v) for p, v in contents.items()}
    )
    result = fr.build_rows(old, diff, contents)
    assert any('duplicate UUID' in error for error in result.errors)


def test_moved_samples_compare_hash_not_only_size():
    old = fr._row_with_samples(make_row(str(uuid4()), 'gsm8k'), b'{"x":1}')
    path = old.legacy_path.replace('/gsm8k/', '/other/')
    sample_path = fr.samples_path_for(path)
    contents = {
        path: record_bytes(old.object_uuid, 'gsm8k'),
        sample_path: b'{"x":2}',
    }
    diff = fr.diff_against_snapshot(
        [old], {p: len(v) for p, v in contents.items()}
    )
    result = fr.build_rows(
        [old],
        diff,
        contents,
        lambda object_path: (
            b'{"x":1}' if object_path == old.instance_level_path else None
        ),
    )
    assert not result.errors
    assert len(result.conflicts) == 1
    assert 'immutable' in result.conflicts[0]
    # the moved row keeps its published samples object
    assert result.rows[0].instance_sha == old.instance_sha


def test_missing_index_is_repaired_even_when_snapshot_matches(tmp_path):
    files = {}
    seed_datastore(files, {'gsm8k': 1})
    api = FakeApi(files, tmp_path)
    fr.orchestrate(api, 'org/ds', now=NOW)
    del api.files[f'{fr.INDEXES_PREFIX}/gsm8k.jsonl']
    report = fr.orchestrate(api, 'org/ds', now=NOW)
    assert not report.noop
    assert f'{fr.INDEXES_PREFIX}/gsm8k.jsonl' in api.files


def test_publisher_unreadable_overwrite_outcome_stops(tmp_path, monkeypatch):
    api = FakeApi({fr.LATEST_MANIFEST_PATH: b'old'}, tmp_path)
    api.fail_next = 1

    def unreadable(**kwargs):
        raise RuntimeError('download unavailable')

    monkeypatch.setattr(api, 'hf_hub_download', unreadable)
    monkeypatch.setattr(store, 'is_commit_conflict', lambda exc: True)
    with pytest.raises(
        fr.FlatRebuildError, match='unverifiable.*download unavailable'
    ):
        fr.FlatPublisher(api, 'org/ds').publish(
            [
                [
                    CommitOperationAdd(
                        path_in_repo=fr.LATEST_MANIFEST_PATH,
                        path_or_fileobj=b'new',
                    )
                ]
            ],
            message='m',
            description='d',
        )
    assert not api.commits


def test_unreadable_retention_scan_consumes_budget():
    snapshots = [
        manifest_info(OLD - timedelta(days=i), f'broken{i}') for i in range(3)
    ]
    attempts = []

    def unreadable(info):
        attempts.append(info.dir_path)
        raise RuntimeError('unavailable')

    with pytest.warns(UserWarning, match='Keeping unreadable snapshot'):
        plan = fr.plan_retention(
            snapshots,
            new_uuids=set(),
            current_manifest_path=None,
            now=NOW,
            retain_days=90,
            keep_newest=0,
            scan_budget=1,
            uuids_of=unreadable,
        )
    assert len(attempts) == 1
    assert len(plan.unscanned) == 3
    assert not plan.delete_dirs


@pytest.mark.parametrize('mode', ['existing', 'moved', 'returning'])
@pytest.mark.parametrize('conflicting', [False, True])
def test_historical_companion_attachment_preserves_published_bytes(
    tmp_path, mode, conflicting
):
    row = make_row(str(uuid4()), 'gsm8k')
    published = b'{"x":1}\n{"x":2}\n'
    incoming = b'{ "x": 1 }\n{ "x": 2 }\n' if not conflicting else b'{"x":9}\n'
    historical = fr._row_with_samples(row, published)
    path = (
        row.legacy_path
        if mode != 'moved'
        else row.legacy_path.replace('/gsm8k/', '/other/')
    )
    files = {
        path: record_bytes(row.object_uuid, 'gsm8k'),
        row.object_path: record_bytes(row.object_uuid, 'gsm8k'),
        historical.instance_level_path: published,
        fr.samples_path_for(path): incoming,
    }
    seed_manifest_files(files, [historical], OLD.isoformat())
    current_rows = [] if mode == 'returning' else [row]
    seed_pointer(
        files, seed_manifest_files(files, current_rows, RECENT.isoformat())
    )
    seed_derived(files, current_rows)
    api = FakeApi(files, tmp_path)
    report = fr.orchestrate(api, 'org/ds', now=NOW, verify=True)
    assert bool(report.conflicts) == conflicting
    assert api.files[historical.instance_level_path] == published
    assert historical.instance_level_path not in committed_paths(api)
    rebuilt = fr.read_snapshot(api, 'org/ds')[1][0]
    assert rebuilt.instance_level_available == (not conflicting)
    if not conflicting:
        assert rebuilt.instance_sha == fr.sha256_bytes(published)
        assert rebuilt.instance_level_size_bytes == len(published)


@pytest.mark.parametrize(
    'incoming', [b'{"x":1}\n{"x":2}\n', b'{ "x": 1 }\n{ "x": 2 }\n']
)
def test_moved_multiline_companion_is_not_a_conflict(incoming):
    published = b'{"x":1}\n{"x":2}\n'
    row = fr._row_with_samples(make_row(str(uuid4()), 'gsm8k'), published)
    path = row.legacy_path.replace('/gsm8k/', '/other/')
    contents = {
        path: record_bytes(row.object_uuid, 'gsm8k'),
        fr.samples_path_for(path): incoming,
    }
    result = fr.build_rows(
        [row],
        fr.diff_against_snapshot(
            [row], {p: len(v) for p, v in contents.items()}
        ),
        contents,
        lambda path: published,
    )
    assert not result.errors
    assert not result.conflicts
    assert result.rows[0].instance_sha == row.instance_sha
    assert not result.upload_samples


@pytest.mark.parametrize(
    'incoming,accepted',
    [
        (b'{"x":2}\n{"x":2}\n', False),
        (b'{ "x": 1 }\n{ "x": 2 }\n', True),
        (b'{"x":2}\n{"x":1}\n', False),
    ],
)
def test_deep_verification_checks_companions(tmp_path, incoming, accepted):
    published = b'{"x":1}\n{"x":2}\n'
    row = fr._row_with_samples(make_row(str(uuid4()), 'gsm8k'), published)
    files = {
        row.legacy_path: record_bytes(row.object_uuid, 'gsm8k'),
        row.object_path: record_bytes(row.object_uuid, 'gsm8k'),
        row.instance_level_path: published,
        fr.samples_path_for(row.legacy_path): incoming,
    }
    api = FakeApi(files, tmp_path)
    checked, reserialized, drift = fr.verify_rows(
        api, 'org/ds', [row], {p: len(v) for p, v in files.items()}
    )
    assert checked == 2
    assert bool(drift) == (not accepted)
    assert reserialized == int(accepted)


def test_verify_orchestration_rejects_same_length_companion_drift(tmp_path):
    published = b'{"x":1}\n'
    row = fr._row_with_samples(make_row(str(uuid4()), 'gsm8k'), published)
    files = {
        row.legacy_path: record_bytes(row.object_uuid, 'gsm8k'),
        row.object_path: record_bytes(row.object_uuid, 'gsm8k'),
        row.instance_level_path: published,
        fr.samples_path_for(row.legacy_path): b'{"x":2}\n',
    }
    seed_pointer(files, seed_manifest_files(files, [row], RECENT.isoformat()))
    seed_derived(files, [row])
    api = FakeApi(files, tmp_path)
    with pytest.raises(fr.FlatRebuildError, match='no longer match'):
        fr.orchestrate(api, 'org/ds', now=NOW, verify=True)
    assert not api.commits


@pytest.mark.parametrize(
    'left,right',
    [
        (b'{"score":true}', b'{"score":1}'),
        (b'{"nested":[false]}', b'{"nested":[0]}'),
        (b'{"score":"1"}', b'{"score":1}'),
        (b'{"score":null}', b'{"score":false}'),
    ],
)
@pytest.mark.parametrize('jsonl', [False, True])
def test_semantic_comparison_preserves_types(left, right, jsonl):
    assert not fr.semantically_equal(left, right, jsonl=jsonl)


def test_semantic_comparison_preserves_record_order_and_count():
    assert not fr.semantically_equal(
        b'{"x":1}\n{"x":2}', b'{"x":2}\n{"x":1}', jsonl=True
    )
    assert not fr.semantically_equal(
        b'{"x":1}\n{"x":1}', b'{"x":1}', jsonl=True
    )
    assert fr.semantically_equal(
        b'{"b":2,"a":[true]}', b'{ "a": [true], "b": 2 }'
    )


def test_verify_skips_explicitly_pending_objects(tmp_path):
    row = make_row(str(uuid4()), 'gsm8k')
    api = FakeApi(
        {row.legacy_path: record_bytes(row.object_uuid, 'gsm8k')}, tmp_path
    )
    assert fr.verify_rows(
        api,
        'org/ds',
        [row],
        {row.legacy_path: row.size_bytes},
        pending_objects=frozenset({row.object_path}),
    ) == (0, 0, [])


def test_verify_bootstrap_allows_planned_new_objects(tmp_path):
    row = make_row(str(uuid4()), 'gsm8k')
    api = FakeApi(
        {
            row.legacy_path: record_bytes(row.object_uuid, 'gsm8k'),
            fr.samples_path_for(row.legacy_path): b'{"x":1}\n',
        },
        tmp_path,
    )
    report = fr.orchestrate(
        api, 'org/ds', now=NOW, verify=True, allow_bootstrap=True
    )
    assert not report.conflicts
    assert report.verified_files == 0
    assert row.object_path in api.files
