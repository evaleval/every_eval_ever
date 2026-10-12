from pathlib import Path

import pytest

from every_eval_ever.validator import check_duplicate_entries as checker


def test_explicit_unsupported_file_is_not_silently_ignored(tmp_path: Path):
    path = tmp_path / 'wrong.txt'
    path.write_text('{}', encoding='utf-8')
    with pytest.raises(SystemExit):
        checker.main([str(path)])


def test_explicit_samples_file_is_not_silently_ignored(tmp_path: Path):
    path = tmp_path / 'x_samples.jsonl'
    path.write_text('{}\n', encoding='utf-8')
    with pytest.raises(SystemExit):
        checker.main([str(path)])


def test_non_object_aggregate_is_reported_not_crashed(tmp_path: Path):
    path = tmp_path / 'x.json'
    path.write_text('[]', encoding='utf-8')
    assert checker.main([str(path)]) == 1
