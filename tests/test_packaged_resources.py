from pathlib import Path

from every_eval_ever.helpers import schema as helper_schema
from every_eval_ever.schema import get_schema_version


def test_helper_schema_version_is_independent_of_checkout_layout(
    monkeypatch, tmp_path: Path
) -> None:
    installed_module = (
        tmp_path
        / 'site-packages'
        / 'every_eval_ever'
        / 'helpers'
        / 'schema.py'
    )
    monkeypatch.setattr(helper_schema, '__file__', str(installed_module))

    assert helper_schema._load_schema_version() == get_schema_version()
