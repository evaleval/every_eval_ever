"""Remove the retired BenchPress collection from the datastore."""

from __future__ import annotations

import os
from typing import Any

from huggingface_hub import CommitOperationDelete, HfApi

from every_eval_ever.cron import store
from every_eval_ever.cron.submit import DEFAULT_DATASTORE_REPO

COLLECTION_PATH = 'data/benchpress'


def _exists(api: Any, repo_id: str) -> bool:
    root = api.list_repo_tree(repo_id=repo_id, repo_type='dataset')
    if not any(item.path == 'data' for item in root):
        return False
    data = api.list_repo_tree(
        repo_id=repo_id, repo_type='dataset', path_in_repo='data'
    )
    return any(item.path == COLLECTION_PATH for item in data)


def retire(api: Any, repo_id: str) -> bool:
    """Delete the collection if present, and verify that it is absent."""
    for attempt in range(1, store.COMMIT_ATTEMPTS + 1):
        if not _exists(api, repo_id):
            return False
        try:
            api.create_commit(
                repo_id=repo_id,
                repo_type='dataset',
                operations=[
                    CommitOperationDelete(
                        path_in_repo=COLLECTION_PATH, is_folder=True
                    )
                ],
                commit_message='Remove retired BenchPress collection',
            )
        except Exception as exc:
            if not _exists(api, repo_id):
                return True
            if attempt < store.COMMIT_ATTEMPTS and store.is_commit_conflict(exc):
                store.wait_before_retry(attempt)
                continue
            raise
        if _exists(api, repo_id):
            raise RuntimeError(f'{COLLECTION_PATH} still exists after deletion')
        return True
    raise RuntimeError(f'could not remove {COLLECTION_PATH}')


def main() -> None:
    if not (os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')):
        raise RuntimeError('a Hugging Face token is required for cleanup')
    repo_id = os.environ.get('EEE_DATASTORE_REPO_ID') or DEFAULT_DATASTORE_REPO
    removed = retire(HfApi(), repo_id)
    print(f'{COLLECTION_PATH}: {"removed" if removed else "already absent"}')


if __name__ == '__main__':
    main()
