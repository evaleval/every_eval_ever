import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any

_UUID_FILE_RE = re.compile(
    r'(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})(?:_samples)?(?:\.jsonl?)?$',
    re.IGNORECASE,
)


def convert_timestamp_to_unix_format(timestamp: str) -> str:
    dt = datetime.fromisoformat(timestamp)
    return str(dt.timestamp())


def get_current_unix_timestamp() -> str:
    return str(datetime.now().timestamp())


def sha256_file(path, chunk_size=8192):
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def sha256_string(text: str, chunk_size=8192):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def extract_file_uuid_from_detailed_results(log: Any) -> str | None:
    detailed = getattr(log, 'detailed_evaluation_results', None)
    if not detailed:
        return None

    file_path = getattr(detailed, 'file_path', None)
    if not file_path:
        return None

    filename = Path(str(file_path)).name
    uuid_match = _UUID_FILE_RE.search(filename)
    if uuid_match:
        return uuid_match.group('uuid')

    return None
