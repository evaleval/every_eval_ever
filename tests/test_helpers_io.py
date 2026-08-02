from pathlib import Path
from types import SimpleNamespace

import pytest

import every_eval_ever.helpers.io as io


def test_publication_replaces_colons_in_collection_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    eval_log = SimpleNamespace(model_dump=lambda: {})
    validated = SimpleNamespace(
        model_info=SimpleNamespace(id='developer/model'),
        model_dump=lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        io.EvaluationLog,
        'model_validate',
        lambda _value: validated,
    )

    prepared = io._prepare_evaluation_logs(
        [
            io.EvaluationLogOutput(
                eval_log=eval_log,  # type: ignore[arg-type]
                base_dir=tmp_path / 'data' / 'benchmark::version',
                developer='developer',
                model_name='model',
            )
        ]
    )

    assert (
        prepared[0].path.parent
        == tmp_path / 'data' / 'benchmark__version' / 'developer' / 'model'
    )


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


def test_batch_rejects_distinct_model_ids_with_colliding_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def validate(value):
        return SimpleNamespace(
            model_info=SimpleNamespace(id=value['model_id']),
            model_dump=lambda **_kwargs: value,
        )

    monkeypatch.setattr(io.EvaluationLog, 'model_validate', validate)
    outputs = [
        io.EvaluationLogOutput(
            eval_log=SimpleNamespace(
                model_dump=lambda: {'model_id': 'developer/family/model'}
            ),
            base_dir=tmp_path / 'data' / 'benchmark',
            developer='developer',
            model_name='family/model',
        ),
        io.EvaluationLogOutput(
            eval_log=SimpleNamespace(
                model_dump=lambda: {'model_id': 'developer/family_model'}
            ),
            base_dir=tmp_path / 'data' / 'benchmark',
            developer='developer',
            model_name='family_model',
        ),
    ]

    with pytest.raises(ValueError, match='same datastore directory'):
        io.save_evaluation_logs(outputs)  # type: ignore[arg-type]

    assert not (tmp_path / 'data').exists()


def test_batch_rollback_removes_new_empty_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    first = tmp_path / 'data' / 'bench' / 'dev' / 'first' / 'first.json'
    second = tmp_path / 'data' / 'bench' / 'dev' / 'second' / 'second.json'
    monkeypatch.setattr(
        io,
        '_prepare_evaluation_logs',
        lambda _outputs: [
            io._PreparedEvaluationLog(first, '{"first": true}\n'),
            io._PreparedEvaluationLog(second, '{"second": true}\n'),
        ],
    )
    original_open = Path.open

    def failing_open(path: Path, mode='r', *args, **kwargs):
        if path == second and mode == 'x':
            raise OSError('simulated write failure')
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', failing_open)

    with pytest.raises(OSError, match='simulated write failure'):
        io.save_evaluation_logs([])

    assert not (tmp_path / 'data').exists()
