"""Migrate legacy Live Code Bench Pro data to the current EEE schema.

Moves top-level source_data URLs into per-evaluation_result source_data fields
using SourceDataUrl, matches each URL to its evaluation by difficulty, and
backfills metadata required by the current validation rules.

Usage:
    uv run python -m every_eval_ever.adapters.livecodebenchpro.adapter
"""

import json
from pathlib import Path

from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.schema import get_schema_version
from every_eval_ever.validator.validation_core import (
    check_model_deployment,
    check_score_metadata,
)

BASE_URL = 'https://webhook.cp-bench.orzzh.com/leaderboard/llm/difficulty'
DATA_DIR = Path(__file__).resolve().parents[3] / 'data' / 'livecodebenchpro'
TARGET_SCHEMA_VERSION = get_schema_version()
LEGACY_SCHEMA_VERSIONS = {
    '0.1.0',
    '0.2.0',
    '0.2.1',
    '0.2.2',
    '0.2.3',
}

# Map evaluation_name -> difficulty for URL matching
DIFFICULTY_FOR_EVAL = {
    'Hard Problems': 'hard',
    'Medium Problems': 'medium',
    'Easy Problems': 'easy',
}


def make_source_data(difficulty: str) -> dict:
    """Build a SourceDataUrl dict for a given difficulty."""
    return {
        'dataset_name': f'{difficulty.capitalize()} Problems',
        'source_type': 'url',
        'url': [f'{BASE_URL}?difficulty={difficulty}&benchmark_mode=live'],
    }


def migrate_file(filepath: Path) -> bool:
    """Migrate one legacy JSON file to the current packaged schema."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    source_version = data.get('schema_version')
    if source_version == TARGET_SCHEMA_VERSION:
        print(f'Skipping (already {TARGET_SCHEMA_VERSION}): {filepath}')
        return False
    if source_version not in LEGACY_SCHEMA_VERSIONS:
        raise ValueError(
            f'{filepath}: expected a legacy schema version, got '
            f'{source_version!r}'
        )

    # Remove top-level source_data
    data.pop('source_data', None)

    scores = [
        result.get('score_details', {}).get('score')
        for result in data.get('evaluation_results', [])
    ]
    max_score = (
        100.0
        if any(
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and score > 1
            for score in scores
        )
        else 1.0
    )

    # Add source_data to each evaluation_result
    for result in data['evaluation_results']:
        eval_name = result.get('evaluation_name')
        if not eval_name:
            raise ValueError(
                f'{filepath}: evaluation_result missing evaluation_name'
            )

        difficulty = DIFFICULTY_FOR_EVAL.get(eval_name)
        if not difficulty:
            raise ValueError(
                f"{filepath}: unknown evaluation_name '{eval_name}'"
            )

        result.setdefault('source_data', make_source_data(difficulty))
        metric_config = result.setdefault('metric_config', {})
        metric_config.setdefault('lower_is_better', False)
        metric_config.setdefault('score_type', 'continuous')
        if metric_config.get('score_type') == 'continuous':
            metric_config.setdefault('min_score', 0.0)
            metric_config.setdefault('max_score', max_score)

    model_info = data.get('model_info', {})
    model_details = model_info.setdefault('additional_details', {})
    model_details.setdefault('deployment_type', 'unknown')
    model_details.setdefault('model_availability', 'unknown')

    data['schema_version'] = TARGET_SCHEMA_VERSION

    EvaluationLog.model_validate(data)
    rule_errors = check_score_metadata(data) + check_model_deployment(data)
    if rule_errors:
        raise ValueError(
            f'{filepath}: migrated data violates current rules: {rule_errors}'
        )

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')
    return True


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f'Data directory not found: {DATA_DIR}')

    files = list(DATA_DIR.rglob('*.json'))
    if not files:
        raise FileNotFoundError(f'No JSON files found in {DATA_DIR}')

    print(
        f'Migrating {len(files)} files in {DATA_DIR} '
        f'to schema {TARGET_SCHEMA_VERSION}...'
    )

    migrated = 0
    for filepath in files:
        if migrate_file(filepath):
            migrated += 1
            print(f'Migrated: {filepath}')

    print(f'\nDone! Migrated {migrated} files to {TARGET_SCHEMA_VERSION}.')


if __name__ == '__main__':
    main()
