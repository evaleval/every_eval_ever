from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS_BASEURL = '/projects/every-eval-ever/docs'
DOCS_URL = f'https://evalevalai.com{DOCS_BASEURL}/'


def test_jekyll_docs_stack_was_removed() -> None:
    removed_paths = (
        '_config.yml',
        'Gemfile',
        'Gemfile.lock',
        'Dockerfile.jekyll',
        'compose.jekyll.yml',
        'docs/_sass',
        'tests/test_docker_jekyll_workflow.py',
    )

    for relative_path in removed_paths:
        assert not (ROOT / relative_path).exists(), relative_path


def test_docs_source_contract_is_pandoc_static_site() -> None:
    nav = yaml.safe_load((ROOT / 'docs/nav.yml').read_text(encoding='utf-8'))
    routes = [item['route'] for item in nav['items']]
    sources = [item['source'] for item in nav['items']]

    assert routes == [
        '/',
        '/getting-started/',
        '/data-model/',
        '/validation/',
        '/converters/',
        '/hf-community-evals/',
        '/contributing/',
    ]
    assert len(routes) == len(set(routes))

    for source in sources:
        assert (ROOT / 'docs' / source).is_file()

    assert (ROOT / 'docs/templates/page.html').is_file()
    assert (ROOT / 'docs/assets/docs.css').is_file()
    assert (ROOT / 'docs/assets/docs.js').is_file()
    assert (ROOT / 'docs/assets/eee-logo.png').is_file()
    assert (ROOT / 'tools/build_docs.py').is_file()
    assert (ROOT / 'tools/serve_docs.py').is_file()
    assert (ROOT / 'docs/Dockerfile.docs').is_file()
    assert (ROOT / 'docs/compose.docs.yml').is_file()
    assert (ROOT / 'docs/Dockerfile.docs.dockerignore').is_file()
    assert not (ROOT / '.dockerignore').exists()


def test_docs_homepage_routes_without_custom_markup() -> None:
    home = (ROOT / 'docs/pages/home.md').read_text(encoding='utf-8')

    assert '## Choose A Path' in home
    assert 'Checking files before or after a PR opens: use [Validation](/validation/)' in home
    assert '[Validation](/validation/)' in home
    assert 'class=' not in home
    assert '<div' not in home
    assert '<a ' not in home

    for markdown_link in (
        '[Getting Started](/getting-started/)',
        '[Data Model](/data-model/)',
        '[Validation](/validation/)',
        '[Converters](/converters/)',
        '[HF Community Evals](/hf-community-evals/)',
        '[Contributing](/contributing/)',
    ):
        assert markdown_link in home

    for label in (
        'Getting Started',
        'Data Model',
        'Validation',
        'Converters',
        'HF Community Evals',
        'Contributing',
        'Basic Flow',
        'Useful Links',
    ):
        assert label in home


def test_validation_page_covers_local_cli_and_pr_bot() -> None:
    validation = (ROOT / 'docs/pages/validation.md').read_text(
        encoding='utf-8'
    )

    assert '## Local CLI' in validation
    assert 'uv run every_eval_ever validate' in validation
    assert '## Duplicate Check' in validation
    assert 'uv run every_eval_ever check-duplicates' in validation
    assert '## PR Bot' in validation
    assert '/eee validate changed' in validation
    assert '/eee schema status' in validation
    assert '/validator/' not in validation


def test_docs_assets_are_local_and_minimal() -> None:
    css = (ROOT / 'docs/assets/docs.css').read_text(encoding='utf-8')
    js = (ROOT / 'docs/assets/docs.js').read_text(encoding='utf-8')
    template = (ROOT / 'docs/templates/page.html').read_text(
        encoding='utf-8'
    )
    combined = '\n'.join((css, js, template))

    for forbidden in (
        '@import',
        'cdn.',
        'analytics',
        'plausible',
        'gtag',
        'mobile-drawer',
        'mobile-backdrop',
        'data-mobile-toggle',
        'linear-gradient',
        'box-shadow',
        'color-mix',
        'site-header',
        'toc-list',
        'On This Page',
    ):
        assert forbidden not in combined

    for plain_html_token in (
        '--page: #f7f7f7',
        '@media (prefers-color-scheme: dark)',
        '--paper: #ffffff',
        '--text: #111111',
        '--line: #d8d8d8',
        '--evaleval-blue: #3b82f6',
        'font-family: "JetBrains Mono"',
        '--link: var(--evaleval-blue)',
        'a:active',
        '.nav-link.is-active:visited',
        'grid-template-columns: minmax(12rem, 16rem) minmax(0, 48rem)',
        '.brand-logo',
        '.page-header',
    ):
        assert plain_html_token in css

    assert 'columns: 2' in css
    assert 'data-search-index="{{search_index}}"' in template
    assert 'https://fonts.googleapis.com/css2?family=JetBrains+Mono' in template
    assert 'https://fonts.gstatic.com' in template
    assert 'src="{{logo}}"' in template
    assert 'class="brand-logo"' in template
    assert '{{repository_url}}' in template
    assert '{{datastore_url}}' in template
    assert '<footer class="footer">' in template
    assert 'fetch(searchIndexUrl' in js
    assert 'data-mobile' not in js


def test_pages_workflow_uses_pandoc_build_and_keeps_dispatch() -> None:
    workflow = (ROOT / '.github/workflows/pages.yml').read_text(
        encoding='utf-8'
    )

    assert 'astral-sh/setup-uv' in workflow
    assert 'sudo apt-get install -y pandoc' in workflow
    assert 'bundle exec jekyll build' not in workflow
    assert 'ruby/setup-ruby' not in workflow
    assert (
        'uv run tools/build_docs.py --base-url '
        '/projects/every-eval-ever/docs --site-url https://evalevalai.com '
        '--output _site'
    ) in workflow
    assert 'tools/serve_docs.py' in workflow
    assert 'tests/test_pandoc_docs_workflow.py' in workflow
    assert 'tests/test_docs_dev_server.py' in workflow
    assert 'test -f _site/assets/eee-logo.png' in workflow
    assert 'test -f _site/assets/search-index.json' in workflow
    assert 'src="/projects/every-eval-ever/docs/assets/eee-logo.png' in workflow
    assert "grep -R 'href=\"/projects/every-eval-ever/docs'" in workflow
    assert "grep -R 'href=\"/every_eval_ever'" in workflow
    assert 'legacy_site/index.html' in workflow
    assert 'legacy_site/404.html' in workflow
    assert 'const legacyPrefix = "/every_eval_ever";' in workflow
    assert 'actions/deploy-pages@v5' in workflow
    assert 'https://api.github.com/repos/evaleval/evaleval.github.io/dispatches' in workflow
    assert '"event_type":"every-eval-ever-docs"' in workflow
    assert '"source_sha":"${{ github.sha }}"' in workflow


def test_readme_points_to_public_docs_route() -> None:
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')

    assert DOCS_URL in readme
    assert 'https://evalevalai.com/every_eval_ever/' not in readme
