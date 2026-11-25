"""The Open LLM Leaderboard v2 table, read from ``open-llm-leaderboard/contents``.

The leaderboard publishes its final table as one parquet file. Each row is
one (model, precision) evaluation. ``load_contents`` turns the rows into the
shape the leaderboard's Space API served, which is what the adapter converts.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

CONTENTS_REPO = 'open-llm-leaderboard/contents'
CONTENTS_REVISION = '9c09a7cae43334062a82cb164f2ef255013dafa2'
CONTENTS_FILE = 'data/train-00000-of-00001.parquet'

# eval_key -> (score column on the 0-1 scale, evaluation name)
SCORE_COLUMNS = {
    'ifeval': ('IFEval Raw', 'IFEval'),
    'bbh': ('BBH Raw', 'BBH'),
    'math': ('MATH Lvl 5 Raw', 'MATH Level 5'),
    'gpqa': ('GPQA Raw', 'GPQA'),
    'musr': ('MUSR Raw', 'MUSR'),
    'mmlu_pro': ('MMLU-PRO Raw', 'MMLU-PRO'),
}


def _stated(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def contents_row(row: dict[str, Any]) -> dict[str, Any]:
    """One parquet row in the Space API's row shape."""
    model = {
        'name': row.get('fullname'),
        'precision': row.get('Precision'),
        'architecture': row.get('Architecture'),
    }
    metadata = {}
    params = _stated(row.get('#Params (B)'))
    if params is not None:
        metadata['params_billions'] = params
    return {
        'model': model,
        'metadata': metadata,
        'evaluations': {
            key: {'name': name, 'value': _stated(row.get(column))}
            for key, (column, name) in SCORE_COLUMNS.items()
        },
    }


def load_contents(parquet_path: str | Path) -> list[dict[str, Any]]:
    """Every row of the contents parquet, in the Space API's row shape."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    return [contents_row(row) for row in table.to_pylist()]


def download_contents() -> Path:
    """Fetch the pinned contents parquet in one Hub download."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=CONTENTS_REPO,
            filename=CONTENTS_FILE,
            repo_type='dataset',
            revision=CONTENTS_REVISION,
        )
    )
