from pathlib import Path

import pytest

import every_eval_ever.helpers.io as io


def test_reserved_data_output_component_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match='reserved datastore name'):
        io.generate_output_path(tmp_path, 'data', 'model')

    with pytest.raises(ValueError, match='reserved datastore name'):
        io.generate_output_path(tmp_path, 'developer', 'data')

    with pytest.raises(ValueError, match='reserved datastore name'):
        io.datastore_path_components('data', 'developer/model')

    output = io.EvaluationLogOutput(
        eval_log=None,  # type: ignore[arg-type]
        base_dir=tmp_path / 'data' / 'data',
        developer='developer',
        model_name='model',
    )
    with pytest.raises(ValueError, match='reserved datastore name'):
        io.save_evaluation_logs([output])


def test_rollback_does_not_remove_file_that_lost_creation_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    first = tmp_path / 'first.json'
    raced = tmp_path / 'raced.json'
    monkeypatch.setattr(
        io,
        '_prepare_evaluation_logs',
        lambda _outputs: [
            io._PreparedEvaluationLog(first, '{"first": true}\n'),
            io._PreparedEvaluationLog(raced, '{"second": true}\n'),
        ],
    )
    original_open = Path.open

    def racing_open(path: Path, mode='r', *args, **kwargs):
        if path == raced and mode == 'x':
            with original_open(path, 'w', encoding='utf-8') as file:
                file.write('created by another process\n')
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', racing_open)

    with pytest.raises(FileExistsError):
        io.save_evaluation_logs([])

    assert not first.exists()
    assert raced.read_text(encoding='utf-8') == 'created by another process\n'
