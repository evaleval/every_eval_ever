from types import SimpleNamespace

import pytest

from every_eval_ever.converters.common.utils import (
    extract_file_uuid_from_detailed_results,
)

UUID = '5cd3f6ca-2fd0-4f88-8f19-9d53089641df'


@pytest.mark.parametrize('codec', ['gz', 'zst', 'bz2', 'xz', 'lz4'])
def test_extract_uuid_accepts_compressed_sample_paths(codec: str):
    log = SimpleNamespace(
        detailed_evaluation_results=SimpleNamespace(
            file_path=f'data/bench/dev/model/{UUID}_samples.jsonl.{codec}'
        )
    )
    assert extract_file_uuid_from_detailed_results(log) == UUID
