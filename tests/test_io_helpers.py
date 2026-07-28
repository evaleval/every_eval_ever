from every_eval_ever.helpers.io import (
    datastore_path_components,
    generate_output_path,
)


def test_datastore_path_replaces_colons_for_windows():
    assert datastore_path_components(
        'benchmark::version',
        'developer/model:revision',
    ) == (
        'benchmark__version',
        'developer',
        'model_revision',
    )


def test_basic_output_path_replaces_colons_for_windows(tmp_path):
    path = generate_output_path(
        tmp_path,
        'developer',
        'model::revision',
    )

    assert path == tmp_path / 'developer' / 'model__revision'
