"""Migrate legacy RewardBench JSON files to the current EEE schema.

Key changes:
- Set ``schema_version`` to the packaged schema version
- Remove top-level "source_data" field
- Add "source_data" to each evaluation result item
- Remove "inference_platform": "unknown" from model_info (now optional)
- Add required model deployment metadata with conservative ``unknown`` values
- Ensure RewardBench's continuous scores declare their 0-1 bounds

For RewardBench v1 results (evaluation_id starts with "reward-bench/"):
    source_data = {"dataset_name": "RewardBench", "source_type": "hf_dataset", "hf_repo": "allenai/reward-bench"}

For RewardBench v2 results (evaluation_id starts with "reward-bench-2/"):
    source_data = {"dataset_name": "RewardBench 2", "source_type": "hf_dataset", "hf_repo": "allenai/reward-bench-2-results"}

Usage:
    uv run python -m every_eval_ever.adapters.rewardbench.migrate_to_v030
"""

import json
from pathlib import Path

from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.schema import get_schema_version
from every_eval_ever.validator.validation_core import (
    check_model_deployment,
    check_score_metadata,
)

DATA_DIR = Path('data/reward-bench')
TARGET_SCHEMA_VERSION = get_schema_version()
LEGACY_SCHEMA_VERSIONS = {
    '0.1.0',
    '0.2.0',
    '0.2.1',
    '0.2.2',
    '0.2.3',
}

V1_SOURCE_DATA = {
    'dataset_name': 'RewardBench',
    'source_type': 'hf_dataset',
    'hf_repo': 'allenai/reward-bench',
}

V2_SOURCE_DATA = {
    'dataset_name': 'RewardBench 2',
    'source_type': 'hf_dataset',
    'hf_repo': 'allenai/reward-bench-2-results',
}


def migrate_file(filepath: Path) -> bool:
    """Migrate one legacy JSON file to the current packaged schema.

    Returns True if the file was modified, False if it was already up to date.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    source_version = data.get('schema_version')
    if source_version == TARGET_SCHEMA_VERSION:
        return False
    if source_version not in LEGACY_SCHEMA_VERSIONS:
        raise ValueError(
            f'{filepath}: expected a legacy schema version, got '
            f'{source_version!r}'
        )

    # Determine source_data based on evaluation_id
    evaluation_id = data.get('evaluation_id', '')
    if evaluation_id.startswith('reward-bench-2/'):
        source_data = V2_SOURCE_DATA
    else:
        source_data = V1_SOURCE_DATA

    data['schema_version'] = TARGET_SCHEMA_VERSION

    # 2. Remove top-level source_data
    data.pop('source_data', None)

    # 3. Add source_data to each evaluation result
    for result in data.get('evaluation_results', []):
        if 'source_data' not in result:
            result['source_data'] = source_data
        metric_config = result.setdefault('metric_config', {})
        metric_config.setdefault('lower_is_better', False)
        metric_config.setdefault('score_type', 'continuous')
        if metric_config.get('score_type') == 'continuous':
            metric_config.setdefault('min_score', 0.0)
            metric_config.setdefault('max_score', 1.0)

    # 4. Clean up model_info: remove inference_platform if "unknown"
    model_info = data.get('model_info', {})
    if model_info.get('inference_platform') == 'unknown':
        del model_info['inference_platform']
    model_details = model_info.setdefault('additional_details', {})
    model_details.setdefault('deployment_type', 'unknown')
    model_details.setdefault('model_availability', 'unknown')

    EvaluationLog.model_validate(data)
    rule_errors = check_score_metadata(data) + check_model_deployment(data)
    if rule_errors:
        raise ValueError(
            f'{filepath}: migrated data violates current rules: {rule_errors}'
        )

    # Write back
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')

    return True


def main():
    """Migrate all RewardBench JSON files to the current schema."""
    if not DATA_DIR.exists():
        print(f'Error: {DATA_DIR} does not exist')
        return

    json_files = sorted(DATA_DIR.rglob('*.json'))
    print(f'Found {len(json_files)} JSON files in {DATA_DIR}')

    migrated = 0
    skipped = 0
    errors = 0

    for filepath in json_files:
        try:
            if migrate_file(filepath):
                migrated += 1
            else:
                skipped += 1
        except Exception as e:
            print(f'  Error migrating {filepath}: {e}')
            errors += 1

    print(f'\nMigration to schema {TARGET_SCHEMA_VERSION} complete:')
    print(f'  Migrated: {migrated}')
    print(f'  Skipped (already {TARGET_SCHEMA_VERSION}): {skipped}')
    print(f'  Errors: {errors}')


if __name__ == '__main__':
    main()
