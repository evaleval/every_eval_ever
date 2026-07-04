from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest
import yaml
from rich.console import Console
from rich.progress import Progress

from every_eval_ever import cli


def _load_community_evals_converter():
    source = (
        Path(__file__).resolve().parents[1]
        / 'tools'
        / 'hf-community-evals'
        / 'community_evals_converter.py'
    )
    spec = importlib.util.spec_from_file_location(
        'community_evals_converter_under_test',
        source,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to load {source}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


community_evals_converter = _load_community_evals_converter()

FIXTURE_DIR = Path(__file__).parent / 'data' / 'community_evals_converter'


class FakeRepoInfo:
    def __init__(self, *, sha: str) -> None:
        self.sha = sha


class FakeHfApi:
    def __init__(
        self,
        *,
        datastore_sha: str = 'abc123',
        missing_models: set[str] | None = None,
        repo_files_by_revision: dict[tuple[str, str], list[str]] | None = None,
        dataset_files_by_revision: dict[tuple[str, str], list[str]] | None = None,
        discussions: dict[str, list[FakeDiscussion]] | None = None,
    ) -> None:
        self.datastore_sha = datastore_sha
        self.missing_models = missing_models or set()
        self.repo_files_by_revision = repo_files_by_revision or {}
        self.dataset_files_by_revision = dataset_files_by_revision or {}
        self.discussions = discussions or {}
        self.model_info_calls: list[str] = []
        self.repo_info_calls: list[dict] = []
        self.discussion_calls: list[str] = []
        self.commits: list[dict] = []

    def repo_info(self, **kwargs):
        self.repo_info_calls.append(kwargs)
        assert kwargs['repo_type'] == 'dataset'
        assert kwargs['revision'] == 'main'
        return FakeRepoInfo(sha=self.datastore_sha)

    def model_info(self, repo_id: str):
        self.model_info_calls.append(repo_id)
        if repo_id in self.missing_models:
            raise RuntimeError('missing model')
        return {'id': repo_id}

    def list_repo_files(
        self,
        repo_id: str,
        repo_type: str = 'model',
        revision: str | None = None,
    ):
        if repo_type == 'dataset':
            return self.dataset_files_by_revision.get(
                (repo_id, revision or 'main'), []
            )
        assert repo_type == 'model'
        return self.repo_files_by_revision.get((repo_id, revision or 'main'), [])

    def list_repo_tree(
        self,
        repo_id: str,
        path_in_repo: str | None = None,
        *,
        recursive: bool = False,
        expand: bool = False,
        revision: str | None = None,
        repo_type: str = 'model',
        token: bool | str | None = None,
    ):
        assert path_in_repo == '.eval_results'
        assert recursive is True
        assert expand is False
        assert repo_type == 'model'
        assert token is True
        for path in self.repo_files_by_revision.get((repo_id, revision or 'main'), []):
            yield FakeRepoFile(path=path, blob_id=f'{revision}:{path}')

    def get_repo_discussions(self, repo_id: str, **_kwargs):
        self.discussion_calls.append(repo_id)
        return self.discussions.get(repo_id, [])

    def create_commit(self, **kwargs):
        self.commits.append(kwargs)
        return FakeCommitInfo(
            pr_url=f'https://huggingface.co/{kwargs["repo_id"]}/discussions/1',
            commit_url=f'https://huggingface.co/{kwargs["repo_id"]}/commit/abc',
        )


class FakeCommitInfo:
    def __init__(self, *, pr_url: str, commit_url: str) -> None:
        self.pr_url = pr_url
        self.commit_url = commit_url

    def __str__(self) -> str:
        return self.pr_url


class FakeRepoFile:
    def __init__(self, *, path: str, blob_id: str) -> None:
        self.path = path
        self.rfilename = path
        self.blob_id = blob_id
        self.size = 1


class FakeDiscussion:
    def __init__(
        self,
        *,
        title: str = 'Add EvalEval community eval results',
        git_reference: str = 'refs/pr/1',
        url: str = 'https://huggingface.co/google/gemma-2b-it/discussions/1',
        num: int = 1,
    ) -> None:
        self.title = title
        self.git_reference = git_reference
        self.url = url
        self.num = num


class RecordingProgress(community_evals_converter.ReviewProgress):
    def __init__(self) -> None:
        self.descriptions: list[str] = []
        self.advances: list[int] = []
        self.task_initial_descriptions: dict[int, str] = {}
        self.advance_by_task: dict[int, int] = {}

    def add_task(self, description: str, total: int | None = None) -> int:
        task_id = len(self.task_initial_descriptions) + 1
        self.task_initial_descriptions[task_id] = description
        self.advance_by_task[task_id] = 0
        self.descriptions.append(description)
        return task_id

    def update(
        self,
        task_id: int,
        *,
        advance: int = 0,
        description: str | None = None,
        total: int | None = None,
    ) -> None:
        self.advances.append(advance)
        self.advance_by_task[task_id] = (
            self.advance_by_task.get(task_id, 0) + advance
        )
        if description is not None:
            self.descriptions.append(description)


def _aggregate(
    *,
    model_id: str = 'google/gemma-2b-it',
    score: float = 0.641,
) -> dict:
    return {
        'schema_version': '0.2.2',
        'evaluation_id': 'openeval/google_gemma-2b-it/123',
        'evaluation_timestamp': '2024-07-16T00:00:00Z',
        'retrieved_timestamp': '1234567890',
        'source_metadata': {
            'source_type': 'evaluation_run',
            'source_organization_name': 'EvalEval',
            'evaluator_relationship': 'third_party',
        },
        'eval_library': {'name': 'openeval', 'version': 'unknown'},
        'model_info': {
            'name': model_id.rsplit('/', 1)[-1],
            'id': model_id,
            'developer': model_id.split('/', 1)[0],
            'inference_platform': 'huggingface',
        },
        'evaluation_results': [
            {
                'evaluation_result_id': 'mmlu-pro::chain-of-thought-correctness',
                'evaluation_name': 'MMLU-Pro',
                'source_data': {
                    'dataset_name': 'MMLU-Pro',
                    'source_type': 'hf_dataset',
                    'hf_repo': 'TIGER-Lab/MMLU-Pro',
                },
                'metric_config': {
                    'lower_is_better': False,
                    'score_type': 'binary',
                    'metric_unit': 'proportion',
                    'min_score': 0.0,
                    'max_score': 1.0,
                },
                'score_details': {'score': score},
            }
        ],
    }


def _gpqa_aggregate(*, dataset_name: str = 'GPQA') -> dict:
    record = _aggregate()
    record['evaluation_results'] = [
        {
            'evaluation_result_id': 'gpqa::chain-of-thought-correctness',
            'evaluation_name': dataset_name,
            'source_data': {
                'dataset_name': dataset_name,
                'source_type': 'hf_dataset',
                'hf_repo': 'Idavidrein/gpqa',
            },
            'metric_config': {
                'lower_is_better': False,
                'score_type': 'binary',
                'metric_unit': 'proportion',
                'min_score': 0.0,
                'max_score': 1.0,
            },
            'score_details': {'score': 0.5},
        }
    ]
    return record


def _write_index_row(
    tmp_path: Path,
    record: dict,
    *,
    object_uuid: str = '676f4465-ce78-411a-9f5a-c97b3d2eac4f',
    row_overrides: dict | None = None,
) -> tuple[Path, Path]:
    datastore = tmp_path / 'datastore'
    object_path = (
        datastore
        / 'flat'
        / 'objects'
        / object_uuid[:2]
        / object_uuid[2:4]
        / f'{object_uuid}.json'
    )
    object_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(record).encode('utf-8')
    object_path.write_bytes(data)

    index_jsonl = tmp_path / 'aggregate.jsonl'
    row = {
        'benchmark': 'MMLU-Pro',
        'eval_schema_version': record['schema_version'],
        'legacy_path': f'data/MMLU-Pro/google/gemma-2b-it/{object_uuid}.json',
        'object_path': object_path.relative_to(datastore).as_posix(),
        'object_uuid': object_uuid,
        'record_type': 'aggregate',
        'sha256': hashlib.sha256(data).hexdigest(),
        'size_bytes': len(data),
    }
    if row_overrides:
        row.update(row_overrides)
    index_jsonl.write_text(json.dumps(row) + '\n', encoding='utf-8')
    return datastore, index_jsonl


def _fake_download(datastore: Path):
    def download_file(**kwargs) -> str:
        assert kwargs['repo_id'] == 'evaleval/EEE_datastore'
        assert kwargs['repo_type'] == 'dataset'
        assert kwargs['revision'] == 'abc123'
        path = datastore / kwargs['filename']
        if not path.exists():
            raise FileNotFoundError(path)
        return path.as_posix()

    return download_file


def _write_collection_rows(
    tmp_path: Path,
    records: list[dict],
    *,
    collection_name: str = 'MMLU-Pro',
    include_instance_level: bool = False,
) -> tuple[Path, Path]:
    datastore = tmp_path / 'datastore'
    rows = []
    for index, record in enumerate(records):
        object_uuid = f'676f4465-ce78-411a-9f5a-c97b3d2eac{index:03d}'
        object_path = (
            datastore
            / 'flat'
            / 'objects'
            / object_uuid[:2]
            / object_uuid[2:4]
            / f'{object_uuid}.json'
        )
        object_path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(record).encode('utf-8')
        object_path.write_bytes(data)
        row = {
            'benchmark': 'collection-benchmark',
            'eval_schema_version': record['schema_version'],
            'legacy_path': (
                f'data/{collection_name}/{record["model_info"]["id"]}/'
                f'{object_uuid}.json'
            ),
            'object_path': object_path.relative_to(datastore).as_posix(),
            'object_uuid': object_uuid,
            'record_type': 'aggregate',
            'sha256': hashlib.sha256(data).hexdigest(),
            'size_bytes': len(data),
            'instance_level_available': False,
        }
        if include_instance_level:
            instance_path = object_path.with_name(f'{object_uuid}_samples.jsonl')
            instance_data = (
                json.dumps(
                    {
                        'schema_version': 'instance_level_eval_0.2.2',
                        'evaluation_id': record['evaluation_id'],
                        'model_id': record['model_info']['id'],
                    }
                )
                + '\n'
            ).encode('utf-8')
            instance_path.write_bytes(instance_data)
            row.update(
                {
                    'instance_level_available': True,
                    'instance_level_path': (
                        instance_path.relative_to(datastore).as_posix()
                    ),
                    'instance_level_size_bytes': len(instance_data),
                    'instance_sha': hashlib.sha256(instance_data).hexdigest(),
                }
            )
        rows.append(row)

    collection_jsonl = (
        datastore
        / 'flat'
        / 'indexes'
        / 'by_collection'
        / f'{collection_name}.jsonl'
    )
    collection_jsonl.parent.mkdir(parents=True, exist_ok=True)
    collection_jsonl.write_text(
        ''.join(json.dumps(row) + '\n' for row in rows),
        encoding='utf-8',
    )
    return datastore, collection_jsonl


def _fake_download_with_model_files(
    datastore: Path,
    model_files: dict[tuple[str, str, str], Path],
):
    def download_file(**kwargs) -> str:
        if kwargs['repo_type'] == 'dataset':
            assert kwargs['repo_id'] == 'evaleval/EEE_datastore'
            assert kwargs['revision'] == 'abc123'
            path = datastore / kwargs['filename']
            if not path.exists():
                raise FileNotFoundError(path)
            return path.as_posix()
        if kwargs['repo_type'] == 'model':
            key = (kwargs['repo_id'], kwargs['revision'], kwargs['filename'])
            return model_files[key].as_posix()
        raise AssertionError(f'unexpected repo_type {kwargs["repo_type"]}')

    return download_file


def test_parse_benchmarks_aliases_and_rejects_unknown() -> None:
    assert community_evals_converter.parse_benchmarks('gpqa-diamond,mmlu_pro') == [
        'gpqa',
        'mmlu_pro',
    ]

    with pytest.raises(community_evals_converter.HFEvalsError, match='Unsupported benchmark'):
        community_evals_converter.parse_benchmarks('alphaxiv')


def test_parse_datastore_locator_accepts_optional_revision() -> None:
    assert community_evals_converter.parse_datastore_locator(
        'evaleval/EEE_datastore@abc123'
    ) == ('evaleval/EEE_datastore', 'abc123')
    assert community_evals_converter.parse_datastore_locator('evaleval/EEE_datastore') == (
        'evaleval/EEE_datastore',
        None,
    )

    with pytest.raises(
        community_evals_converter.HFEvalsError, match='<hf_dataset_repo>\\[@<revision>\\]'
    ):
        community_evals_converter.parse_datastore_locator('bad@repo@abc123')


def test_resolve_datastore_locator_uses_latest_commit_for_bare_repo() -> None:
    api = FakeHfApi(datastore_sha='resolvedabc')

    assert community_evals_converter.resolve_datastore_locator(
        'evaleval/EEE_datastore', api=api
    ) == ('evaleval/EEE_datastore', 'resolvedabc')
    assert api.repo_info_calls == [
        {
            'repo_id': 'evaleval/EEE_datastore',
            'repo_type': 'dataset',
            'revision': 'main',
        }
    ]


def test_build_collection_manifest_downloads_collection_jsonl_and_scans_results(
    tmp_path: Path,
) -> None:
    record = _aggregate()
    record['evaluation_results'].append(
        {
            'evaluation_result_id': 'gsm8k/exact_match',
            'evaluation_name': 'GSM8K',
            'source_data': {
                'dataset_name': 'GSM8K',
                'source_type': 'hf_dataset',
                'hf_repo': 'openai/gsm8k',
            },
            'metric_config': {
                'lower_is_better': False,
                'score_type': 'binary',
                'metric_unit': 'proportion',
                'min_score': 0.0,
                'max_score': 1.0,
            },
            'score_details': {'score': 0.72},
        }
    )
    datastore, _collection_jsonl = _write_collection_rows(
        tmp_path,
        [record],
        collection_name='MMLU-Pro',
        include_instance_level=True,
    )

    manifest = community_evals_converter.build_collection_manifest(
        collection_name='MMLU-Pro',
        datastore='evaleval/EEE_datastore',
        api=FakeHfApi(datastore_sha='abc123'),
        download_file=_fake_download(datastore),
    )

    assert manifest['source_url_mode'] == 'online_collection_index_jsonl'
    assert manifest['collection_jsonl'] == (
        'flat/indexes/by_collection/MMLU-Pro.jsonl'
    )
    assert {entry['benchmark'] for entry in manifest['entries']} == {
        'mmlu_pro',
        'gsm8k',
    }
    assert {entry['target_path'] for entry in manifest['entries']} == {
        '.eval_results/mmlu_pro.yaml',
        '.eval_results/gsm8k.yaml',
    }
    assert all(entry['instance_level_available'] is True for entry in manifest['entries'])
    assert all('instance_sha' in entry for entry in manifest['entries'])


def test_build_collection_manifest_requires_collection_jsonl(
    tmp_path: Path,
) -> None:
    datastore, aggregate_jsonl = _write_index_row(tmp_path, _aggregate())
    aggregate_dir = (
        datastore / 'flat' / 'indexes' / 'by_collection' / 'MMLU-Pro'
    )
    aggregate_dir.mkdir(parents=True)
    (aggregate_dir / 'aggregate.jsonl').write_text(
        aggregate_jsonl.read_text(encoding='utf-8'),
        encoding='utf-8',
    )

    with pytest.raises(
        community_evals_converter.HFEvalsError,
        match='flat/indexes/by_collection/MMLU-Pro\\.jsonl',
    ):
        community_evals_converter.build_collection_manifest(
            collection_name='MMLU-Pro',
            datastore='evaleval/EEE_datastore@abc123',
            api=FakeHfApi(),
            download_file=_fake_download(datastore),
        )


def test_build_collection_manifest_suggests_nearby_collection_stems(
    tmp_path: Path,
) -> None:
    datastore, _collection_jsonl = _write_collection_rows(
        tmp_path,
        [_aggregate()],
        collection_name='fibble_arena',
    )

    with pytest.raises(
        community_evals_converter.HFEvalsError,
        match='Nearby collection stems: fibble_arena',
    ):
        community_evals_converter.build_collection_manifest(
            collection_name='fibbl_arena',
            datastore='evaleval/EEE_datastore@abc123',
            api=FakeHfApi(
                dataset_files_by_revision={
                    (
                        'evaleval/EEE_datastore',
                        'abc123',
                    ): [
                        'flat/indexes/by_collection/fibble_arena.jsonl',
                        'flat/indexes/by_collection/MMLU-Pro.jsonl',
                    ]
                }
            ),
            download_file=_fake_download(datastore),
        )


def test_build_collection_manifest_rejects_malformed_instance_provenance(
    tmp_path: Path,
) -> None:
    datastore, collection_jsonl = _write_collection_rows(
        tmp_path,
        [_aggregate()],
        include_instance_level=True,
    )
    row = json.loads(collection_jsonl.read_text(encoding='utf-8'))
    row.pop('instance_sha')
    collection_jsonl.write_text(json.dumps(row) + '\n', encoding='utf-8')

    with pytest.raises(community_evals_converter.HFEvalsError, match='missing instance_sha'):
        community_evals_converter.build_collection_manifest(
            collection_name='MMLU-Pro',
            datastore='evaleval/EEE_datastore@abc123',
            api=FakeHfApi(),
            download_file=_fake_download(datastore),
        )


def test_build_collection_manifest_rejects_path_like_collection_name(
    tmp_path: Path,
) -> None:
    datastore, _collection_jsonl = _write_collection_rows(tmp_path, [_aggregate()])

    with pytest.raises(community_evals_converter.HFEvalsError, match='without the \\.jsonl'):
        community_evals_converter.build_collection_manifest(
            collection_name='MMLU-Pro.jsonl',
            datastore='evaleval/EEE_datastore@abc123',
            api=FakeHfApi(),
            download_file=_fake_download(datastore),
        )

    with pytest.raises(community_evals_converter.HFEvalsError, match='single by_collection'):
        community_evals_converter.build_collection_manifest(
            collection_name='MMLU-Pro/records',
            datastore='evaleval/EEE_datastore@abc123',
            api=FakeHfApi(),
            download_file=_fake_download(datastore),
        )


def test_build_collection_manifest_records_url_only_result_as_skipped(
    tmp_path: Path,
) -> None:
    record = _aggregate()
    record['evaluation_results'][0]['source_data'] = {
        'dataset_name': 'External Benchmark',
        'source_type': 'url',
        'url': ['https://example.com/not-a-hf-benchmark'],
    }
    datastore, _collection_jsonl = _write_collection_rows(
        tmp_path, [record], collection_name='external'
    )

    manifest = community_evals_converter.build_collection_manifest(
        collection_name='external',
        datastore='evaleval/EEE_datastore@abc123',
        api=FakeHfApi(),
        download_file=_fake_download(datastore),
    )

    assert manifest['entries'] == []
    assert manifest['skipped'][0]['reason'] == 'no_supported_hf_dataset_result'


def test_build_index_manifest_downloads_online_record_and_links_source(
    tmp_path: Path,
) -> None:
    datastore, index_jsonl = _write_index_row(tmp_path, _aggregate())
    api = FakeHfApi(datastore_sha='abc123')

    manifest = community_evals_converter.build_index_manifest(
        index_jsonl=index_jsonl,
        datastore='evaleval/EEE_datastore',
        benchmarks=['mmlu_pro'],
        api=api,
        download_file=_fake_download(datastore),
    )

    assert manifest['source_url_mode'] == 'online_flat_index_jsonl'
    assert manifest['datastore'] == 'evaleval/EEE_datastore@abc123'
    assert manifest['datastore_input'] == 'evaleval/EEE_datastore'
    assert api.repo_info_calls
    assert manifest['entries'][0]['target_path'] == '.eval_results/mmlu_pro.yaml'
    assert manifest['entries'][0]['yaml_entry']['value'] == 64.1
    assert manifest['entries'][0]['yaml_entry']['source']['url'].startswith(
        'https://huggingface.co/datasets/evaleval/EEE_datastore/blob/abc123/flat/objects/'
    )


def test_build_index_manifest_accepts_index_directory(tmp_path: Path) -> None:
    datastore, index_jsonl = _write_index_row(tmp_path, _aggregate())
    index_dir = tmp_path / 'flat' / 'indexes' / 'by_benchmark' / 'MMLU-Pro'
    index_dir.mkdir(parents=True)
    (index_dir / 'aggregate.jsonl').write_text(
        index_jsonl.read_text(encoding='utf-8'),
        encoding='utf-8',
    )

    manifest = community_evals_converter.build_index_manifest(
        index_jsonl=index_dir,
        datastore='evaleval/EEE_datastore@abc123',
        benchmarks=['mmlu_pro'],
        api=FakeHfApi(),
        download_file=_fake_download(datastore),
    )

    assert manifest['index_jsonl'] == (index_dir / 'aggregate.jsonl').as_posix()
    assert manifest['entries'][0]['flat_object_path'].startswith(
        'flat/objects/'
    )


def test_build_index_manifest_rejects_index_directory_without_aggregate_jsonl(
    tmp_path: Path,
) -> None:
    index_dir = tmp_path / 'flat' / 'indexes' / 'by_benchmark' / 'MMLU-Pro'
    index_dir.mkdir(parents=True)

    with pytest.raises(
        community_evals_converter.HFEvalsError, match='must contain aggregate\\.jsonl'
    ):
        community_evals_converter.build_index_manifest(
            index_jsonl=index_dir,
            datastore='evaleval/EEE_datastore@abc123',
            benchmarks=['mmlu_pro'],
            api=FakeHfApi(),
        )


def test_build_index_manifest_rejects_direct_url_row(tmp_path: Path) -> None:
    datastore, index_jsonl = _write_index_row(
        tmp_path,
        _aggregate(),
        row_overrides={
            'object_path': None,
            'url': 'https://huggingface.co/datasets/evaleval/EEE_datastore/blob/main/flat/objects/test.json',
        },
    )

    with pytest.raises(community_evals_converter.HFEvalsError, match='unsupported.*url'):
        community_evals_converter.build_index_manifest(
            index_jsonl=index_jsonl,
            datastore='evaleval/EEE_datastore@abc123',
            benchmarks=['mmlu_pro'],
            api=FakeHfApi(),
            download_file=_fake_download(datastore),
        )


def test_build_index_manifest_rejects_local_path_row(tmp_path: Path) -> None:
    datastore, index_jsonl = _write_index_row(
        tmp_path,
        _aggregate(),
        row_overrides={'object_path': None},
    )
    aggregate_path = next((datastore / 'flat' / 'objects').rglob('*.json'))
    row = json.loads(index_jsonl.read_text(encoding='utf-8'))
    row['local_path'] = aggregate_path.relative_to(tmp_path).as_posix()
    index_jsonl.write_text(json.dumps(row) + '\n', encoding='utf-8')

    with pytest.raises(community_evals_converter.HFEvalsError, match='unsupported.*local_path'):
        community_evals_converter.build_index_manifest(
            index_jsonl=index_jsonl,
            datastore='evaleval/EEE_datastore@abc123',
            benchmarks=['mmlu_pro'],
            api=FakeHfApi(),
            download_file=_fake_download(datastore),
        )


def test_build_index_manifest_rejects_url_even_with_object_path(
    tmp_path: Path,
) -> None:
    datastore, index_jsonl = _write_index_row(
        tmp_path,
        _aggregate(),
        row_overrides={
            'url': 'https://huggingface.co/datasets/evaleval/EEE_datastore/blob/main/flat/objects/test.json',
        },
    )

    with pytest.raises(community_evals_converter.HFEvalsError, match='unsupported.*url'):
        community_evals_converter.build_index_manifest(
            index_jsonl=index_jsonl,
            datastore='evaleval/EEE_datastore@abc123',
            benchmarks=['mmlu_pro'],
            api=FakeHfApi(),
            download_file=_fake_download(datastore),
        )


def test_build_index_manifest_preserves_subset_from_index_row(
    tmp_path: Path,
) -> None:
    datastore, index_jsonl = _write_index_row(
        tmp_path,
        _aggregate(),
        row_overrides={'subset': 'overall'},
    )

    manifest = community_evals_converter.build_index_manifest(
        index_jsonl=index_jsonl,
        datastore='evaleval/EEE_datastore@abc123',
        benchmarks=['mmlu_pro'],
        api=FakeHfApi(),
        download_file=_fake_download(datastore),
    )

    assert manifest['entries'][0]['subset'] == 'overall'


def test_build_index_manifest_uses_gpqa_subset_for_task_id(
    tmp_path: Path,
) -> None:
    datastore, index_jsonl = _write_index_row(
        tmp_path,
        _gpqa_aggregate(),
        row_overrides={'benchmark': 'gpqa', 'subset': 'main'},
    )

    manifest = community_evals_converter.build_index_manifest(
        index_jsonl=index_jsonl,
        datastore='evaleval/EEE_datastore@abc123',
        benchmarks=['gpqa'],
        api=FakeHfApi(),
        download_file=_fake_download(datastore),
    )

    yaml_entry = manifest['entries'][0]['yaml_entry']
    assert manifest['entries'][0]['subset'] == 'main'
    assert yaml_entry['dataset'] == {
        'id': 'Idavidrein/gpqa',
        'task_id': 'main',
    }
    assert yaml_entry['notes'] == 'GPQA chain-of-thought'


def test_build_index_manifest_rejects_invalid_subset_type(
    tmp_path: Path,
) -> None:
    datastore, index_jsonl = _write_index_row(
        tmp_path,
        _aggregate(),
        row_overrides={'subset': {'name': 'overall'}},
    )

    with pytest.raises(community_evals_converter.HFEvalsError, match='subset'):
        community_evals_converter.build_index_manifest(
            index_jsonl=index_jsonl,
            datastore='evaleval/EEE_datastore@abc123',
            benchmarks=['mmlu_pro'],
            api=FakeHfApi(),
            download_file=_fake_download(datastore),
        )


def test_build_index_manifest_accepts_persistent_fixture() -> None:
    manifest = community_evals_converter.build_index_manifest(
        index_jsonl=FIXTURE_DIR / 'aggregate.jsonl',
        datastore='evaleval/EEE_datastore@abc123',
        benchmarks=['mmlu_pro'],
        api=FakeHfApi(),
        download_file=_fake_download(FIXTURE_DIR / 'datastore'),
    )

    assert len(manifest['entries']) == 1
    entry = manifest['entries'][0]
    assert entry['flat_object_path'] == (
        'flat/objects/67/6f/676f4465-ce78-411a-9f5a-c97b3d2eac4f.json'
    )
    assert entry['yaml_entry']['value'] == 52.29
    assert entry['yaml_entry']['dataset'] == {
        'id': 'TIGER-Lab/MMLU-Pro',
        'task_id': 'mmlu_pro',
    }


def test_build_index_manifest_fails_on_hash_mismatch(tmp_path: Path) -> None:
    datastore, index_jsonl = _write_index_row(tmp_path, _aggregate())
    row = json.loads(index_jsonl.read_text(encoding='utf-8'))
    row['sha256'] = '0' * 64
    index_jsonl.write_text(json.dumps(row) + '\n', encoding='utf-8')

    with pytest.raises(community_evals_converter.HFEvalsError, match='sha256 mismatch'):
        community_evals_converter.build_index_manifest(
            index_jsonl=index_jsonl,
            datastore='evaleval/EEE_datastore@abc123',
            benchmarks=['mmlu_pro'],
            api=FakeHfApi(),
            download_file=_fake_download(datastore),
        )


def test_review_index_writes_yaml_and_review(tmp_path: Path) -> None:
    datastore, index_jsonl = _write_index_row(tmp_path, _aggregate())
    review = community_evals_converter.review_index_for_hf_evals(
        index_jsonl=index_jsonl,
        datastore='evaleval/EEE_datastore@abc123',
        benchmarks=['mmlu_pro'],
        manifest_output_path=tmp_path / 'manifest.json',
        yaml_output_dir=tmp_path / 'yamls',
        review_output_path=tmp_path / 'review.json',
        api=FakeHfApi(),
        download_file=_fake_download(datastore),
    )

    yaml_path = (
        tmp_path
        / 'yamls'
        / 'google'
        / 'gemma-2b-it'
        / '.eval_results'
        / 'mmlu_pro.yaml'
    )
    loaded_yaml = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
    loaded_review = json.loads((tmp_path / 'review.json').read_text(encoding='utf-8'))

    assert review['can_open_prs'] is True
    assert loaded_review['can_open_prs'] is True
    assert loaded_yaml[0]['dataset'] == {
        'id': 'TIGER-Lab/MMLU-Pro',
        'task_id': 'mmlu_pro',
    }


def test_review_index_writes_yaml_without_reloading_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datastore, index_jsonl = _write_index_row(tmp_path, _aggregate())

    def fail_load_manifest(_path: Path) -> dict:
        raise AssertionError('review flow should use the in-memory manifest')

    monkeypatch.setattr(community_evals_converter, 'load_manifest', fail_load_manifest)

    review = community_evals_converter.review_index_for_hf_evals(
        index_jsonl=index_jsonl,
        datastore='evaleval/EEE_datastore@abc123',
        benchmarks=['mmlu_pro'],
        manifest_output_path=tmp_path / 'manifest.json',
        yaml_output_dir=tmp_path / 'yamls',
        review_output_path=tmp_path / 'review.json',
        api=FakeHfApi(),
        download_file=_fake_download(datastore),
    )

    assert review['yaml_count'] == 1
    assert (tmp_path / 'manifest.json').exists()
    assert (tmp_path / 'review.json').exists()


def test_review_index_reports_missing_model_without_aliasing(tmp_path: Path) -> None:
    record = _aggregate(model_id='local/missing-model')
    datastore, index_jsonl = _write_index_row(tmp_path, record)

    review = community_evals_converter.review_index_for_hf_evals(
        index_jsonl=index_jsonl,
        datastore='evaleval/EEE_datastore@abc123',
        benchmarks=['mmlu_pro'],
        manifest_output_path=tmp_path / 'manifest.json',
        yaml_output_dir=tmp_path / 'yamls',
        review_output_path=tmp_path / 'review.json',
        api=FakeHfApi(missing_models={'local/missing-model'}),
        download_file=_fake_download(datastore),
    )

    assert review['can_open_prs'] is False
    assert review['yaml_count'] == 0
    assert len(review['missing_hf_models']) == 1
    missing = review['missing_hf_models'][0]
    assert missing['model_repo'] == 'local/missing-model'
    assert missing['status'] == 'missing_hf_model'
    assert 'model_repo_alias_from' not in missing


def test_review_collection_suppresses_existing_same_score_from_any_yaml_name(
    tmp_path: Path,
) -> None:
    datastore, _collection_jsonl = _write_collection_rows(
        tmp_path, [_aggregate()], collection_name='MMLU-Pro'
    )
    model_yaml = tmp_path / 'model_main.yaml'
    model_yaml.write_text(
        yaml.safe_dump(
            [
                {
                    'dataset': {
                        'id': 'TIGER-Lab/MMLU-Pro',
                        'task_id': 'mmlu_pro',
                    },
                    'value': 64.1,
                }
            ],
            sort_keys=False,
        ),
        encoding='utf-8',
    )
    api = FakeHfApi(
        repo_files_by_revision={
            ('google/gemma-2b-it', 'main'): ['.eval_results/not_the_name.yaml']
        }
    )

    review = community_evals_converter.review_collection_for_hf_evals(
        collection_name='MMLU-Pro',
        datastore='evaleval/EEE_datastore@abc123',
        manifest_output_path=tmp_path / 'manifest.json',
        yaml_output_dir=tmp_path / 'yamls',
        review_output_path=tmp_path / 'review.json',
        api=api,
        download_file=_fake_download_with_model_files(
            datastore,
            {
                (
                    'google/gemma-2b-it',
                    'main',
                    '.eval_results/not_the_name.yaml',
                ): model_yaml
            },
        ),
    )

    assert review['can_open_prs'] is False
    assert review['yaml_count'] == 0
    assert review['manifest']['entries'][0]['status'] == 'already_present'
    assert review['duplicate_audit']['findings'][0]['status'] == 'already_present'


def test_review_collection_reports_progress_phases(tmp_path: Path) -> None:
    datastore, _collection_jsonl = _write_collection_rows(
        tmp_path,
        [_aggregate()],
        collection_name='MMLU-Pro',
    )
    progress = RecordingProgress()

    review = community_evals_converter.review_collection_for_hf_evals(
        collection_name='MMLU-Pro',
        datastore='evaleval/EEE_datastore@abc123',
        manifest_output_path=tmp_path / 'manifest.json',
        yaml_output_dir=tmp_path / 'yamls',
        review_output_path=tmp_path / 'review.json',
        api=FakeHfApi(),
        download_file=_fake_download(datastore),
        progress=progress,
    )

    joined = '\n'.join(progress.descriptions)
    assert review['can_open_prs'] is True
    assert 'Downloading collection index MMLU-Pro.jsonl' in joined
    assert 'Processing 1 aggregate rows' in joined
    assert 'row 1/1: downloading flat/objects/' in joined
    assert 'row 1/1: checking google/gemma-2b-it' in joined
    assert 'Auditing 1 ready candidates' in joined


def test_rich_review_progress_uses_one_visible_task() -> None:
    console = Console(file=io.StringIO(), force_terminal=True)
    progress = Progress(console=console)
    review_progress = community_evals_converter.RichReviewProgress(progress)

    with progress:
        setup_task = review_progress.add_task('Resolving datastore revision', total=4)
        review_progress.update(setup_task, advance=4, description='Built manifest')
        row_task = review_progress.add_task('Processing 2 aggregate rows', total=2)
        review_progress.update(row_task, advance=2, description='Processed 2 rows')
        audit_task = review_progress.add_task('Auditing 1 ready candidates', total=1)
        review_progress.update(audit_task, advance=1, description='Audit complete')

    assert len(progress.tasks) == 1
    task = progress.tasks[0]
    assert task.total == 1
    assert task.completed == 1


def test_review_collection_progress_advances_api_only_rows(
    tmp_path: Path,
) -> None:
    api_only_record = _aggregate(model_id='anthropic/claude-3-opus')
    api_only_record['model_info']['developer'] = 'anthropic'
    api_only_record['model_info']['inference_platform'] = 'anthropic'
    datastore, _collection_jsonl = _write_collection_rows(
        tmp_path,
        [_aggregate(), api_only_record],
        collection_name='MMLU-Pro',
    )
    progress = RecordingProgress()

    review = community_evals_converter.review_collection_for_hf_evals(
        collection_name='MMLU-Pro',
        datastore='evaleval/EEE_datastore@abc123',
        manifest_output_path=tmp_path / 'manifest.json',
        yaml_output_dir=tmp_path / 'yamls',
        review_output_path=tmp_path / 'review.json',
        api=FakeHfApi(),
        download_file=_fake_download(datastore),
        progress=progress,
    )

    row_task = next(
        task_id
        for task_id, description in progress.task_initial_descriptions.items()
        if description == 'Processing 2 aggregate rows'
    )
    joined = '\n'.join(progress.descriptions)
    assert review['can_open_prs'] is True
    assert progress.advance_by_task[row_task] == 2
    assert 'Processed 2 aggregate rows' in joined


def test_review_collection_reuses_cached_review_without_downloads(
    tmp_path: Path,
) -> None:
    datastore, _collection_jsonl = _write_collection_rows(
        tmp_path,
        [_aggregate()],
        collection_name='MMLU-Pro',
    )
    manifest_path = tmp_path / 'manifest.json'
    yaml_dir = tmp_path / 'yamls'
    review_path = tmp_path / 'review.json'
    first_review = community_evals_converter.review_collection_for_hf_evals(
        collection_name='MMLU-Pro',
        datastore='evaleval/EEE_datastore@abc123',
        manifest_output_path=manifest_path,
        yaml_output_dir=yaml_dir,
        review_output_path=review_path,
        api=FakeHfApi(),
        download_file=_fake_download(datastore),
    )

    def fail_download(**_kwargs) -> str:
        raise AssertionError('cached review should not download anything')

    api = FakeHfApi(missing_models={'google/gemma-2b-it'})
    second_review = community_evals_converter.review_collection_for_hf_evals(
        collection_name='MMLU-Pro',
        datastore='evaleval/EEE_datastore@abc123',
        manifest_output_path=manifest_path,
        yaml_output_dir=yaml_dir,
        review_output_path=review_path,
        api=api,
        download_file=fail_download,
    )

    assert second_review['created_at'] == first_review['created_at']
    assert second_review['yaml_count'] == 1
    assert api.model_info_calls == []


def test_review_collection_force_ignores_cached_review(
    tmp_path: Path,
) -> None:
    datastore, _collection_jsonl = _write_collection_rows(
        tmp_path,
        [_aggregate()],
        collection_name='MMLU-Pro',
    )
    manifest_path = tmp_path / 'manifest.json'
    yaml_dir = tmp_path / 'yamls'
    review_path = tmp_path / 'review.json'
    community_evals_converter.review_collection_for_hf_evals(
        collection_name='MMLU-Pro',
        datastore='evaleval/EEE_datastore@abc123',
        manifest_output_path=manifest_path,
        yaml_output_dir=yaml_dir,
        review_output_path=review_path,
        api=FakeHfApi(),
        download_file=_fake_download(datastore),
    )

    api = FakeHfApi(missing_models={'google/gemma-2b-it'})
    forced_review = community_evals_converter.review_collection_for_hf_evals(
        collection_name='MMLU-Pro',
        datastore='evaleval/EEE_datastore@abc123',
        manifest_output_path=manifest_path,
        yaml_output_dir=yaml_dir,
        review_output_path=review_path,
        api=api,
        download_file=_fake_download(datastore),
        force=True,
    )

    assert forced_review['can_open_prs'] is False
    assert forced_review['yaml_count'] == 0
    assert api.model_info_calls == ['google/gemma-2b-it']


def test_review_collection_resumes_cached_manifest_without_datastore_downloads(
    tmp_path: Path,
) -> None:
    datastore, _collection_jsonl = _write_collection_rows(
        tmp_path,
        [_aggregate()],
        collection_name='MMLU-Pro',
    )
    manifest_path = tmp_path / 'manifest.json'
    community_evals_converter.build_collection_manifest(
        collection_name='MMLU-Pro',
        datastore='evaleval/EEE_datastore@abc123',
        output_path=manifest_path,
        api=FakeHfApi(),
        download_file=_fake_download(datastore),
    )

    def fail_download(**_kwargs) -> str:
        raise AssertionError('cached manifest should skip datastore downloads')

    api = FakeHfApi(missing_models={'google/gemma-2b-it'})
    review = community_evals_converter.review_collection_for_hf_evals(
        collection_name='MMLU-Pro',
        datastore='evaleval/EEE_datastore@abc123',
        manifest_output_path=manifest_path,
        yaml_output_dir=tmp_path / 'yamls',
        review_output_path=tmp_path / 'review.json',
        api=api,
        download_file=fail_download,
    )

    assert review['can_open_prs'] is True
    assert review['yaml_count'] == 1
    assert api.model_info_calls == []


def test_review_details_use_clear_headers_and_aggregate_existing_scores() -> None:
    console = Console(record=True, width=200)
    review = {
        'duplicate_audit': {
            'errors': [
                {
                    'entry_index': 0,
                    'model_repo': 'google/gemma-blocked',
                    'stage': 'read_open_pr_eval_results',
                    'path': '.eval_results/mmlu_pro.yaml',
                    'error': 'Unable to download eval results YAML',
                }
            ],
            'findings': [
                {
                    'status': 'score_conflict',
                    'model_repo': 'nexusflow/athene-v2-chat',
                    'existing_value': 73.11,
                    'candidate_value': 70.21,
                    'pr_url': 'https://huggingface.co/example/repo/discussions/1',
                    'paths': ['.eval_results/mmlu_pro.yaml'],
                },
                {'status': 'already_present'},
                {'status': 'already_present'},
            ],
        },
        'missing_hf_models': [
            {
                'model_repo': 'missing/model',
                'hf_check_error': 'HF model repo does not exist: missing/model',
                'eee_record_path': 'flat/objects/aa/bb/record.json',
                'yaml_entry': {
                    'source': {
                        'url': (
                            'https://huggingface.co/datasets/evaleval/'
                            'EEE_datastore/blob/abc123/flat/objects/aa/bb/'
                            'record.json'
                        )
                    }
                },
            }
        ],
        'manifest': {
            'datastore_repo': 'evaleval/EEE_datastore',
            'datastore_revision': 'abc123',
            'skipped': [
                {
                    'model_id': 'api/model',
                    'reason': 'api_only_or_closed_provider:gemini',
                    'eee_record_path': 'flat/objects/cc/dd/skipped.json',
                }
            ],
        },
    }

    community_evals_converter._render_review_details(console, review)

    output = console.export_text()
    assert 'Needs Attention' in output
    assert 'Issue' in output
    assert 'Where' in output
    assert 'Details' in output
    assert 'Candidate' not in output
    assert 'Context' not in output
    assert 'Score' not in output
    assert '2 models' in output
    assert 'https://huggingface.co/example/repo/discussions/1' in output
    assert (
        'https://huggingface.co/datasets/evaleval/EEE_datastore/blob/abc123/'
        'flat/objects/aa/bb/record.json'
    ) in output
    assert (
        'https://huggingface.co/datasets/evaleval/EEE_datastore/blob/abc123/'
        'flat/objects/cc/dd/skipped.json'
    ) in output


def test_review_collection_submits_clean_records_despite_open_pr_conflict(
    tmp_path: Path,
) -> None:
    conflict_record = _aggregate(model_id='google/gemma-2b-it')
    clean_record = _aggregate(model_id='google/gemma-clean')
    datastore, _collection_jsonl = _write_collection_rows(
        tmp_path,
        [conflict_record, clean_record],
        collection_name='MMLU-Pro',
    )
    pr_yaml = tmp_path / 'model_pr.yaml'
    pr_yaml.write_text(
        yaml.safe_dump(
            [
                {
                    'dataset': {
                        'id': 'TIGER-Lab/MMLU-Pro',
                        'task_id': 'mmlu_pro',
                    },
                    'value': 12.3,
                }
            ],
            sort_keys=False,
        ),
        encoding='utf-8',
    )
    api = FakeHfApi(
        repo_files_by_revision={
            ('google/gemma-2b-it', 'refs/pr/7'): [
                '.eval_results/random.yaml'
            ],
        },
        discussions={
            'google/gemma-2b-it': [
                FakeDiscussion(git_reference='refs/pr/7', num=7)
            ],
        },
    )

    review = community_evals_converter.review_collection_for_hf_evals(
        collection_name='MMLU-Pro',
        datastore='evaleval/EEE_datastore@abc123',
        manifest_output_path=tmp_path / 'manifest.json',
        yaml_output_dir=tmp_path / 'yamls',
        review_output_path=tmp_path / 'review.json',
        api=api,
        download_file=_fake_download_with_model_files(
            datastore,
            {
                (
                    'google/gemma-2b-it',
                    'refs/pr/7',
                    '.eval_results/random.yaml',
                ): pr_yaml
            },
        ),
    )

    assert review['can_open_prs'] is True
    assert review['yaml_count'] == 1
    statuses = {
        entry['model_repo']: entry['status']
        for entry in review['manifest']['entries']
    }
    assert statuses == {
        'google/gemma-2b-it': 'score_conflict',
        'google/gemma-clean': 'ready',
    }
    assert review['duplicate_audit']['findings'][0]['status'] == 'score_conflict'


def test_review_collection_blocks_only_candidate_with_audit_error(
    tmp_path: Path,
) -> None:
    blocked_record = _aggregate(model_id='google/gemma-blocked')
    clean_record = _aggregate(model_id='google/gemma-clean')
    datastore, _collection_jsonl = _write_collection_rows(
        tmp_path,
        [blocked_record, clean_record],
        collection_name='MMLU-Pro',
    )
    api = FakeHfApi(
        repo_files_by_revision={
            ('google/gemma-blocked', 'refs/pr/7'): [
                '.eval_results/mmlu_pro.yaml'
            ],
        },
        discussions={
            'google/gemma-blocked': [
                FakeDiscussion(git_reference='refs/pr/7', num=7)
            ],
        },
    )

    review = community_evals_converter.review_collection_for_hf_evals(
        collection_name='MMLU-Pro',
        datastore='evaleval/EEE_datastore@abc123',
        manifest_output_path=tmp_path / 'manifest.json',
        yaml_output_dir=tmp_path / 'yamls',
        review_output_path=tmp_path / 'review.json',
        api=api,
        download_file=_fake_download_with_model_files(datastore, {}),
    )

    statuses = {
        entry['model_repo']: entry['status']
        for entry in review['manifest']['entries']
    }
    assert review['can_open_prs'] is True
    assert statuses == {
        'google/gemma-blocked': 'audit_error',
        'google/gemma-clean': 'ready',
    }
    assert review['duplicate_audit']['error_count'] == 1
    assert review['duplicate_audit']['errors'][0]['entry_index'] == 0
    assert review['audit_blocked_entries'][0]['model_repo'] == (
        'google/gemma-blocked'
    )
    assert review['yaml_count'] == 2
    assert (
        tmp_path
        / 'yamls'
        / 'google'
        / 'gemma-blocked'
        / '.eval_results'
        / 'mmlu_pro.yaml'
    ).exists()

    submit_api = FakeHfApi()
    result = community_evals_converter.create_prs_from_manifest(
        manifest_path=tmp_path / 'manifest.json',
        limit=None,
        yes_i_reviewed=True,
        commit_message='Add EvalEval result',
        api=submit_api,
    )

    assert result['count'] == 1
    assert submit_api.commits[0]['repo_id'] == 'google/gemma-clean'


def test_create_prs_from_manifest_creates_fresh_pr_only(tmp_path: Path) -> None:
    datastore, index_jsonl = _write_index_row(tmp_path, _aggregate())
    manifest_path = tmp_path / 'manifest.json'
    community_evals_converter.build_index_manifest(
        index_jsonl=index_jsonl,
        datastore='evaleval/EEE_datastore@abc123',
        benchmarks=['mmlu_pro'],
        output_path=manifest_path,
        api=FakeHfApi(),
        download_file=_fake_download(datastore),
    )
    api = FakeHfApi(
        discussions={
            'google/gemma-2b-it': [
                FakeDiscussion(git_reference='refs/pr/123'),
            ]
        }
    )

    result = community_evals_converter.create_prs_from_manifest(
        manifest_path=manifest_path,
        limit=None,
        yes_i_reviewed=True,
        commit_message='Add EvalEval result',
        api=api,
    )

    assert result['count'] == 1
    assert api.discussion_calls == []
    assert len(api.commits) == 1
    commit = api.commits[0]
    assert commit['repo_id'] == 'google/gemma-2b-it'
    assert commit['revision'] == 'main'
    assert commit['create_pr'] is True
    assert [op.__class__.__name__ for op in commit['operations']] == [
        'CommitOperationAdd'
    ]


def test_tui_approval_requires_exact_phrase(monkeypatch) -> None:
    console = Console(record=True)
    review = {
        'manifest': {
            'entries': [
                {
                    'status': 'ready',
                    'model_repo': 'google/gemma-2b-it',
                    'target_path': '.eval_results/mmlu_pro.yaml',
                }
            ]
        }
    }

    monkeypatch.setattr(community_evals_converter.Prompt, 'ask', lambda *_args, **_kwargs: 'yes')

    assert community_evals_converter._approve_pr_submission(console, review) is False


def test_tui_approval_accepts_open_prs(monkeypatch) -> None:
    console = Console(record=True)
    review = {
        'manifest': {
            'entries': [
                {
                    'status': 'ready',
                    'model_repo': 'google/gemma-2b-it',
                    'target_path': '.eval_results/mmlu_pro.yaml',
                }
            ]
        }
    }

    monkeypatch.setattr(
        community_evals_converter.Prompt,
        'ask',
        lambda *_args, **_kwargs: community_evals_converter.APPROVAL_PHRASE,
    )

    assert community_evals_converter._approve_pr_submission(console, review) is True


def test_prompt_commit_message_requires_non_empty(monkeypatch) -> None:
    console = Console(record=True)

    monkeypatch.setattr(community_evals_converter.Prompt, 'ask', lambda *_args, **_kwargs: ' ')

    assert community_evals_converter._prompt_commit_message(console) is None


def test_prompt_commit_message_returns_typed_message(monkeypatch) -> None:
    console = Console(record=True)

    monkeypatch.setattr(
        community_evals_converter.Prompt,
        'ask',
        lambda *_args, **_kwargs: 'Add verified EvalEval result',
    )

    assert (
        community_evals_converter._prompt_commit_message(console)
        == 'Add verified EvalEval result'
    )


def test_parser_rejects_removed_open_prs_flag() -> None:
    parser = community_evals_converter.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                'aggregate.jsonl',
                '--datastore',
                'evaleval/EEE_datastore@abc123',
                '--open-prs',
            ]
        )


def test_parser_rejects_old_index_workflow_flags() -> None:
    parser = community_evals_converter.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(['MMLU-Pro', '--benchmarks', 'mmlu_pro'])

    with pytest.raises(SystemExit):
        parser.parse_args(['MMLU-Pro', '--manifest-output', 'manifest.json'])


def test_parser_defaults_to_datastore_repo() -> None:
    parser = community_evals_converter.build_parser()

    args = parser.parse_args(['MMLU-Pro'])

    assert args.collection_name == 'MMLU-Pro'
    assert args.datastore == 'evaleval/EEE_datastore'
    assert args.force is False


def test_parser_accepts_force() -> None:
    parser = community_evals_converter.build_parser()

    args = parser.parse_args(['MMLU-Pro', '--force'])

    assert args.force is True


def test_every_eval_ever_cli_no_longer_exposes_hf_evals() -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(['hf-evals'])
