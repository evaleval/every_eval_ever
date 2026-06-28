from __future__ import annotations

from pathlib import Path

from tools import serve_docs

ROOT = Path(__file__).resolve().parents[1]
DOCS_BASEURL = '/projects/every-eval-ever/docs'


def test_output_dir_for_base_url_mounts_under_site_root(
    tmp_path: Path,
) -> None:
    assert serve_docs.output_dir_for_base_url(
        tmp_path, DOCS_BASEURL
    ) == tmp_path / 'projects/every-eval-ever/docs'
    assert serve_docs.output_dir_for_base_url(tmp_path, '/') == tmp_path


def test_docs_dev_server_watches_exact_docs_sources() -> None:
    watched = {path.relative_to(ROOT).as_posix() for path in serve_docs.watched_paths()}

    assert 'tools/build_docs.py' in watched
    assert 'docs/nav.yml' in watched
    assert 'docs/pages/home.md' in watched
    assert 'docs/templates/page.html' in watched
    assert 'docs/assets/docs.css' in watched
    assert 'docs/assets/docs.js' in watched
    assert 'docs/assets/eee-logo.png' in watched
    assert 'pyproject.toml' in watched
    assert 'uv.lock' in watched


def test_source_snapshot_detects_file_changes(tmp_path: Path) -> None:
    source = tmp_path / 'source.md'
    source.write_text('first\n', encoding='utf-8')
    first = serve_docs.source_snapshot((source,))

    source.write_text('second\n', encoding='utf-8')
    second = serve_docs.source_snapshot((source,))

    assert first != second


def test_docker_docs_stack_uses_pandoc_live_server_only() -> None:
    dockerfile = (ROOT / 'docs/Dockerfile.docs').read_text(encoding='utf-8')
    compose = (ROOT / 'docs/compose.docs.yml').read_text(encoding='utf-8')
    dockerignore = (ROOT / 'docs/Dockerfile.docs.dockerignore').read_text(
        encoding='utf-8'
    )
    combined = '\n'.join((dockerfile, compose))

    assert 'pandoc' in dockerfile
    assert 'uv sync --dev --frozen' in dockerfile
    assert 'UV_LINK_MODE=copy' in dockerfile
    assert 'UV_LINK_MODE: copy' in compose
    assert 'context: ..' in compose
    assert 'dockerfile: docs/Dockerfile.docs' in compose
    assert 'Dockerfile.docs.dockerignore' not in compose
    assert 'tools/serve_docs.py' in combined
    assert '4173:4173' in compose
    assert '..:/workspace' in compose
    assert 'docs-venv:/workspace/.venv' in compose
    assert 'docs-uv-cache:/tmp/uv-cache' in compose
    assert '.venv' in dockerignore
    assert '_site' in dockerignore
    assert '.git' in dockerignore

    for forbidden in (
        'jekyll',
        'Just-the-Docs',
        'bundle exec',
        'ruby/setup-ruby',
        'Gemfile',
    ):
        assert forbidden not in combined
