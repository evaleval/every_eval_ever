import json
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Any, Dict, Set, List

# Define Schema Path
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "eval.schema.json"


def _load_schema(schema_path: Path) -> Dict[str, Any]:
    if not schema_path.exists():
        return {}
    try:
        return json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get_json_columns(schema: Dict[str, Any]) -> Set[str]:
    """Identify top-level fields that are complex types (objects/arrays)."""
    if not schema:
        return set()

    def is_complex(spec: Dict[str, Any]) -> bool:
        t = spec.get("type")
        if isinstance(t, str) and t in ("object", "array"):
            return True
        if isinstance(t, list) and ("object" in t or "array" in t):
            return True
        for key in ("oneOf", "anyOf", "allOf"):
            if key in spec:
                return True
        return False

    props = schema.get("properties", {})
    return {name for name, spec in props.items() if is_complex(spec)}


SCHEMA = _load_schema(SCHEMA_PATH)
SCHEMA_PROPERTIES: Set[str] = set(SCHEMA.get("properties", {}).keys())
JSON_FIELDS_CONFIG = _get_json_columns(SCHEMA)


def json_to_row(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    leaderboard = data.get("source_metadata", {}).get("source_name")
    model_id = data.get("model_info", {}).get("id")
    developer = data.get("model_info", {}).get("developer")
    uuid = json_path.stem

    if not leaderboard:
        leaderboard = "unknown_leaderboard"
    if not developer:
        developer = "unknown_developer"
    if not model_id:
        model_id = "unknown_model"

    row: dict[str, Any] = {
        "_leaderboard": leaderboard,
        "_developer": developer,
        "_model": model_id,
        "_uuid": uuid,
    }

    for key, value in data.items():
        if key in JSON_FIELDS_CONFIG:
            row[key] = json.dumps(value)
        else:
            row[key] = value

    return row


def add_to_parquet(
    json_input: str | list[str], parquet_file: str, *, rebuild: bool = False
) -> bool:
    if isinstance(json_input, list):
        json_files = [Path(f) for f in json_input]
    else:
        input_path = Path(json_input)
        if input_path.is_file():
            json_files = [input_path]
        elif input_path.is_dir():
            json_files = list(input_path.rglob("*.json"))
        else:
            raise ValueError(f"Invalid input: {json_input}")

    if not json_files:
        print("No JSON files found.")
        return False

    parquet_path = Path(parquet_file)
    existing_keys = set()
    existing_df = None

    if parquet_path.exists() and not rebuild:
        try:
            existing_df = pd.read_parquet(parquet_file)
            existing_keys = set(
                existing_df[["_leaderboard", "_developer", "_model", "_uuid"]].apply(
                    tuple, axis=1
                )
            )
        except Exception as e:
            print(f"Warning: Could not read existing parquet: {e}")

    new_rows = []
    skipped = 0

    for json_file in json_files:
        try:
            row = json_to_row(json_file)
            key = (row["_leaderboard"], row["_developer"], row["_model"], row["_uuid"])

            if key not in existing_keys:
                new_rows.append(row)
                existing_keys.add(key)
            else:
                skipped += 1
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    if not new_rows:
        print(f"No new rows to add. Skipped {skipped} duplicates.")
        return False

    new_df = pd.DataFrame(new_rows)

    if existing_df is not None:
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    table = pa.Table.from_pandas(final_df)

    existing_meta = table.schema.metadata or {}
    my_meta = {"json_columns": json.dumps(list(JSON_FIELDS_CONFIG))}
    merged_meta = {
        **existing_meta,
        **{k.encode(): v.encode() for k, v in my_meta.items()},
    }

    table = table.replace_schema_metadata(merged_meta)

    pq.write_table(table, parquet_file)

    print(f"Added {len(new_rows)} rows. Total: {len(final_df)}. Skipped: {skipped}.")
    return True


def parquet_to_folder(parquet_file: str, output_dir: str):
    """
    Reconstructs JSON files using metadata stored INSIDE the parquet file.
    This makes the extraction robust to Schema changes.
    """
    path_pq = Path(parquet_file)
    if not path_pq.exists():
        raise FileNotFoundError(f"{parquet_file} does not exist")

    table = pq.read_table(parquet_file)
    df = table.to_pandas()

    metadata = table.schema.metadata or {}
    json_cols_bytes = metadata.get(b"json_columns")

    json_cols = set()
    if json_cols_bytes:
        json_cols = set(json.loads(json_cols_bytes.decode("utf-8")))
    else:
        print("Warning: No 'json_columns' metadata found in Parquet.")
        json_cols = JSON_FIELDS_CONFIG

    out_base = Path(output_dir)
    count = 0

    for row in df.itertuples(index=False):
        leaderboard = getattr(row, "_leaderboard", "unknown")
        developer = getattr(row, "_developer", "unknown")
        model_id = getattr(row, "_model", "unknown")
        uuid = getattr(row, "_uuid", "unknown")

        data_dict = {}
        for field in row._fields:
            if field.startswith("_"):
                continue

            val = getattr(row, field)

            if pd.isna(val):
                continue

            if field in json_cols and isinstance(val, str):
                try:
                    data_dict[field] = json.loads(val)
                except json.JSONDecodeError:
                    data_dict[field] = val
            else:
                data_dict[field] = val

        safe_model = str(model_id).replace("/", "_")
        target_path = (
            out_base / str(leaderboard) / str(developer) / safe_model / f"{uuid}.json"
        )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(data_dict, indent=2), encoding="utf-8")
        count += 1

    print(f"Successfully reconstructed {count} JSON files in {output_dir}")
