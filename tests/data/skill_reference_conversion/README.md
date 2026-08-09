Frozen output of the `eee-dataset-conversion` skill templates — the
artifact a contributor following the skill would submit to the EEE_datastore.

Re-validated as-is by tests/test_skill_conversion.py using the CLI semantic checks,
and compared byte-for-byte against what the templates produce today.

Do not hand-edit. Regenerate with:

    uv run python -c "from tests.test_skill_conversion import regenerate_frozen_reference as r; r()"
