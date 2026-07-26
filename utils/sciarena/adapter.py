#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from every_eval_ever.adapters.sciarena.provenance import (
    SCI_ARENA_MODEL_DEVELOPERS,
    sci_arena_developer,
    sci_arena_provenance,
)
from every_eval_ever.helpers import SCHEMA_VERSION, sanitize_filename

PROVIDER_MAP = SCI_ARENA_MODEL_DEVELOPERS

SOURCE_URL = 'https://sciarena.allen.ai/api/leaderboard'


def make_source_data() -> dict:
    return {
        'source_type': 'url',
        'dataset_name': 'SciArena leaderboard API',
        'url': [SOURCE_URL],
    }


def load_rows(input_json: Path) -> list[dict]:
    return json.loads(input_json.read_text(encoding='utf-8'))


def compute_metric_bounds(rows: list[dict]) -> dict[str, dict[str, float]]:
    rating_values = [float(row['rating']) for row in rows]
    rank_values = [float(row['rank']) for row in rows]
    cost_values = [
        float(row['cost_per_100_calls_usd'])
        for row in rows
        if row.get('cost_per_100_calls_usd') is not None
    ]

    bounds = {
        'elo': {
            'min_score': min(rating_values),
            'max_score': max(rating_values),
        },
        'rank': {
            'min_score': 1.0,
            'max_score': max(rank_values),
        },
    }

    if cost_values:
        bounds['cost_per_100_calls_usd'] = {
            'min_score': 0.0,
            'max_score': max(cost_values),
        }

    return bounds


def slugify_model_name(raw_model_id: str) -> str:
    # Keep close to source aliases while ensuring a single path segment.
    model_name = sanitize_filename(raw_model_id.strip().lower())
    return model_name.replace('\\', '-').replace('/', '-')


def normalize_model(raw_model_id: str) -> tuple[str, str]:
    developer_name = sci_arena_developer(raw_model_id)
    model_name = slugify_model_name(raw_model_id)
    return developer_name, model_name


def make_results(
    row: dict, metric_bounds: dict[str, dict[str, float]]
) -> list[dict]:
    results = []

    results.append(
        {
            'evaluation_result_id': 'overall::elo',
            'evaluation_name': 'overall_elo',
            'source_data': make_source_data(),
            'metric_config': {
                'metric_id': 'elo',
                'metric_name': 'Elo rating',
                'metric_type': 'continuous',
                'metric_kind': 'elo',
                'metric_unit': 'points',
                'lower_is_better': False,
                'score_type': 'continuous',
                **metric_bounds['elo'],
                'additional_details': {
                    'raw_metric_field': 'rating',
                },
            },
            'score_details': {
                'score': float(row['rating']),
                'details': {
                    'num_battles': str(row['num_battles']),
                    'rating_q025': str(row['rating_q025']),
                    'rating_q975': str(row['rating_q975']),
                    'variance': str(row['variance']),
                },
            },
        }
    )

    results.append(
        {
            'evaluation_result_id': 'overall::rank',
            'evaluation_name': 'overall_rank',
            'source_data': make_source_data(),
            'metric_config': {
                'metric_id': 'rank',
                'metric_name': 'Rank',
                'metric_type': 'continuous',
                'metric_kind': 'rank',
                'metric_unit': 'position',
                'lower_is_better': True,
                'score_type': 'continuous',
                **metric_bounds['rank'],
            },
            'score_details': {
                'score': float(row['rank']),
            },
        }
    )

    if row.get('cost_per_100_calls_usd') is not None:
        results.append(
            {
                'evaluation_result_id': 'overall::cost_per_100_calls_usd',
                'evaluation_name': 'overall_cost_per_100_calls_usd',
                'source_data': make_source_data(),
                'metric_config': {
                    'metric_id': 'cost_per_100_calls_usd',
                    'metric_name': 'Cost per 100 calls',
                    'metric_type': 'continuous',
                    'metric_kind': 'cost',
                    'metric_unit': 'usd',
                    'lower_is_better': True,
                    'score_type': 'continuous',
                    **metric_bounds['cost_per_100_calls_usd'],
                },
                'score_details': {
                    'score': float(row['cost_per_100_calls_usd']),
                },
            }
        )

    return results


def make_log(
    row: dict,
    metric_bounds: dict[str, dict[str, float]],
    retrieved_timestamp: str,
) -> tuple[dict, str, str]:
    raw_model_id = row['modelId']
    developer_name, model_name = normalize_model(raw_model_id)
    provenance = sci_arena_provenance(f'{developer_name}/{model_name}')

    log = {
        'schema_version': SCHEMA_VERSION,
        'evaluation_id': (
            f'sciarena/{developer_name}/{model_name}/{retrieved_timestamp}'
        ),
        'retrieved_timestamp': retrieved_timestamp,
        'source_metadata': {
            'source_name': 'SciArena leaderboard API',
            'source_type': 'documentation',
            'source_organization_name': 'Ai2',
            'source_organization_url': 'https://sciarena.allen.ai',
            'evaluator_relationship': 'third_party',
            'additional_details': {
                'api_endpoint': SOURCE_URL,
            },
        },
        'eval_library': {
            'name': 'SciArena',
            'version': 'unknown',
        },
        'model_info': {
            'name': raw_model_id,
            'id': f'{developer_name}/{model_name}',
            'developer': developer_name,
            'inference_platform': provenance.inference_platform,
            'inference_engine': {
                'name': provenance.inference_engine_name,
                'version': provenance.inference_engine_version,
            },
            'additional_details': {
                'raw_model_id': raw_model_id,
                'deployment_type': provenance.deployment_type,
                'model_availability': provenance.model_availability,
            },
        },
        'evaluation_results': make_results(row, metric_bounds),
    }
    return log, developer_name, model_name


def write_log(log: dict, out_root: Path, developer: str, model: str) -> Path:
    out_dir = out_root / 'sciarena' / developer / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{uuid.uuid4()}.json'
    out_path.write_text(json.dumps(log, indent=2) + '\n', encoding='utf-8')
    return out_path


def export_one(
    row: dict,
    out_root: Path,
    metric_bounds: dict[str, dict[str, float]],
    retrieved_timestamp: str,
) -> Path:
    log, developer, model = make_log(row, metric_bounds, retrieved_timestamp)
    return write_log(log, out_root, developer, model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-json', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.input_json)
    retrieved_timestamp = str(time.time())

    missing = [
        row['modelId'] for row in rows if row['modelId'] not in PROVIDER_MAP
    ]
    if missing:
        raise SystemExit(f'Missing provider mappings for: {missing}')

    metric_bounds = compute_metric_bounds(rows)

    prepared = [
        make_log(row, metric_bounds, retrieved_timestamp) for row in rows
    ]
    exported = 0
    for log, developer, model in prepared:
        out_path = write_log(log, args.output_dir, developer, model)
        print(out_path)
        exported += 1

    print(f'Exported {exported} model(s).')


if __name__ == '__main__':
    main()
