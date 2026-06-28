from __future__ import annotations

import json
import os
import stat
import textwrap
from pathlib import Path

import pytest

from tools import build_docs

DOCS_BASEURL = '/projects/every-eval-ever/docs'


def _write_fake_pandoc(path: Path, body: str | None = None) -> None:
    if body is None:
        body = r'''
file=""
for arg do
  file="$arg"
done
awk '
function slug(text, id) {
  id = tolower(text)
  gsub(/[^a-z0-9]+/, "-", id)
  gsub(/^-|-$/, "", id)
  return id
}
/^# / {
  text = substr($0, 3)
  print "<h1 id=\"" slug(text) "\">" text "</h1>"
  next
}
/^## / {
  text = substr($0, 4)
  print "<h2 id=\"" slug(text) "\">" text "</h2>"
  next
}
/href="/ {
  print
  next
}
/^$/ {
  next
}
{
  gsub(/&/, "\\&amp;")
  gsub(/</, "\\&lt;")
  gsub(/>/, "\\&gt;")
  print "<p>" $0 "</p>"
}
' "$file"
'''
    path.write_text(f'#!/bin/sh\n{body}\n', encoding='utf-8')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_build_produces_static_artifact_with_fake_pandoc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    _write_fake_pandoc(fake_bin / 'pandoc')
    monkeypatch.setenv('PATH', f'{fake_bin}{os.pathsep}{os.environ["PATH"]}')

    output = tmp_path / 'site'
    build_docs.build_docs(
        DOCS_BASEURL,
        'https://evalevalai.com',
        output,
    )

    expected_pages = (
        'index.html',
        'getting-started/index.html',
        'data-model/index.html',
        'validation/index.html',
        'converters/index.html',
        'hf-community-evals/index.html',
        'contributing/index.html',
    )
    for page in expected_pages:
        assert (output / page).is_file(), page

    assert (output / 'assets/docs.css').is_file()
    assert (output / 'assets/docs.js').is_file()
    assert (output / 'assets/eee-logo.png').is_file()
    search_index = json.loads(
        (output / 'assets/search-index.json').read_text(encoding='utf-8')
    )
    assert [entry['title'] for entry in search_index] == [
        'Home',
        'Getting Started',
        'Data Model',
        'Validation',
        'Converters',
        'HF Community Evals',
        'Contributing',
    ]
    assert all(
        entry['route'].startswith(DOCS_BASEURL) for entry in search_index
    )

    home = (output / 'index.html').read_text(encoding='utf-8')
    assert f'href="{DOCS_BASEURL}/getting-started/"' in home
    assert f'href="{DOCS_BASEURL}/assets/docs.css?v=' in home
    assert f'src="{DOCS_BASEURL}/assets/docs.js?v=' in home
    assert f'src="{DOCS_BASEURL}/assets/eee-logo.png?v=' in home
    assert f'data-search-index="{DOCS_BASEURL}/assets/search-index.json?v=' in home
    assert '<footer class="footer">' in home
    assert 'aria-label="Project links"' in home
    assert 'href="https://github.com/evaleval/every_eval_ever"' in home
    assert 'href="https://huggingface.co/datasets/evaleval/EEE_datastore"' in home
    assert '<aside class="toc" aria-label="On this page">' not in home
    assert 'On This Page' not in home
    assert 'mobile-drawer' not in home
    assert 'href="/every_eval_ever' not in home


def test_builder_rejects_missing_pandoc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('PATH', '')

    with pytest.raises(build_docs.DocsBuildError, match='Pandoc is required'):
        build_docs.require_pandoc()


def test_source_manifest_lists_exact_pandoc_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('PATH', '')

    manifest = build_docs.source_manifest()

    assert manifest['external_dependencies'] == ['pandoc']
    assert manifest['source_files'] == [
        'tools/build_docs.py',
        'docs/nav.yml',
        'docs/pages/home.md',
        'docs/pages/getting-started.md',
        'docs/pages/data-model.md',
        'docs/pages/validation.md',
        'docs/pages/converters.md',
        'docs/pages/hf-community-evals.md',
        'docs/pages/contributing.md',
        'docs/templates/page.html',
        'docs/assets/docs.css',
        'docs/assets/docs.js',
        'docs/assets/eee-logo.png',
    ]
    assert [route['route'] for route in manifest['routes']] == [
        '/',
        '/getting-started/',
        '/data-model/',
        '/validation/',
        '/converters/',
        '/hf-community-evals/',
        '/contributing/',
    ]


def test_builder_rejects_generated_legacy_internal_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    _write_fake_pandoc(
        fake_bin / 'pandoc',
        'printf %s \'<h1 id="bad">Bad</h1><a href="/every_eval_ever/">Legacy</a>\'',
    )
    monkeypatch.setenv('PATH', f'{fake_bin}{os.pathsep}{os.environ["PATH"]}')

    with pytest.raises(build_docs.DocsBuildError, match='/every_eval_ever'):
        build_docs.build_docs(
            DOCS_BASEURL,
            'https://evalevalai.com',
            tmp_path / 'site',
        )


def test_nav_validation_rejects_duplicates_and_missing_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = tmp_path / 'docs'
    pages = docs / 'pages'
    pages.mkdir(parents=True)
    (pages / 'one.md').write_text('# One\n', encoding='utf-8')
    (docs / 'nav.yml').write_text(
        textwrap.dedent(
            '''
            site:
              title: Test
              description: Test docs
            items:
              - title: One
                route: /
                source: pages/one.md
                description: One
              - title: Duplicate
                route: /
                source: pages/missing.md
                description: Duplicate
            '''
        ),
        encoding='utf-8',
    )
    monkeypatch.setattr(build_docs, 'DOCS_DIR', docs)
    monkeypatch.setattr(build_docs, 'NAV_PATH', docs / 'nav.yml')

    with pytest.raises(build_docs.DocsBuildError) as excinfo:
        build_docs.load_nav()

    message = str(excinfo.value)
    assert 'duplicate docs route' in message


def test_internal_link_resolution_is_strict() -> None:
    routes = {'/', '/validation/'}
    anchors = {'/': {'every-eval-ever-docs'}, '/validation/': {'commands'}}

    errors: list[str] = []
    assert (
        build_docs.resolve_internal_url(
            '/validation/#commands',
            current_route='/',
            route_set=routes,
            anchors=anchors,
            base_url=DOCS_BASEURL,
            attr='href',
            errors=errors,
        )
        == f'{DOCS_BASEURL}/validation/#commands'
    )
    assert errors == []

    build_docs.resolve_internal_url(
        '/missing/',
        current_route='/',
        route_set=routes,
        anchors=anchors,
        base_url=DOCS_BASEURL,
        attr='href',
        errors=errors,
    )
    build_docs.resolve_internal_url(
        '/validation/#missing',
        current_route='/',
        route_set=routes,
        anchors=anchors,
        base_url=DOCS_BASEURL,
        attr='href',
        errors=errors,
    )
    build_docs.resolve_internal_url(
        '/every_eval_ever/',
        current_route='/',
        route_set=routes,
        anchors=anchors,
        base_url=DOCS_BASEURL,
        attr='href',
        errors=errors,
    )
    build_docs.resolve_internal_url(
        '/assets/unknown.css',
        current_route='/',
        route_set=routes,
        anchors=anchors,
        base_url=DOCS_BASEURL,
        attr='href',
        errors=errors,
    )

    assert any('unknown docs route' in error for error in errors)
    assert any('missing anchor' in error for error in errors)
    assert any('legacy internal path' in error for error in errors)
    assert any('unknown asset' in error for error in errors)
