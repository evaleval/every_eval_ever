#!/usr/bin/env python3
"""Build the Every Eval Ever docs as a Pandoc static site."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / 'docs'
NAV_PATH = DOCS_DIR / 'nav.yml'
TEMPLATE_PATH = DOCS_DIR / 'templates' / 'page.html'
ASSET_NAMES = ('docs.css', 'docs.js', 'eee-logo.png')
SEARCH_INDEX = 'search-index.json'
VALID_ASSET_NAMES = {*ASSET_NAMES, SEARCH_INDEX}
LEGACY_INTERNAL_PATH = '/every_eval_ever'
HTML_EXTENSIONS = (
    '+yaml_metadata_block'
    '+fenced_divs'
    '+pipe_tables'
)


class DocsBuildError(RuntimeError):
    """Raised when the docs cannot be built safely."""


@dataclass(frozen=True)
class NavItem:
    title: str
    route: str
    source: Path
    description: str


@dataclass
class RenderedPage:
    item: NavItem
    body_html: str
    full_html: str = ''
    headings: list[dict[str, str]] | None = None
    text: str = ''


class HtmlInfoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self.heading_ids: set[str] = set()
        self.element_ids: set[str] = set()
        self.headings: list[dict[str, str]] = []
        self.text_parts: list[str] = []
        self._heading_tag: str | None = None
        self._heading_id: str | None = None
        self._heading_text: list[str] = []
        self._skip_text_depth = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attrs_dict = dict(attrs)
        if tag in {'script', 'style'}:
            self._skip_text_depth += 1
        if tag in {'a', 'link', 'script', 'img'}:
            for attr_name in ('href', 'src'):
                value = attrs_dict.get(attr_name)
                if value:
                    self.links.append((attr_name, value))
        element_id = attrs_dict.get('id')
        if element_id:
            self.element_ids.add(element_id)
        if re.fullmatch(r'h[1-6]', tag):
            heading_id = element_id
            self._heading_tag = tag
            self._heading_id = heading_id
            self._heading_text = []
            if heading_id:
                self.heading_ids.add(heading_id)

    def handle_endtag(self, tag: str) -> None:
        if tag in {'script', 'style'} and self._skip_text_depth:
            self._skip_text_depth -= 1
        if tag == self._heading_tag:
            title = ' '.join(''.join(self._heading_text).split())
            if self._heading_id and title:
                self.headings.append(
                    {
                        'id': self._heading_id,
                        'level': self._heading_tag[1],
                        'text': title,
                    }
                )
            self._heading_tag = None
            self._heading_id = None
            self._heading_text = []

    def handle_data(self, data: str) -> None:
        if not self._skip_text_depth:
            text = data.strip()
            if text:
                self.text_parts.append(text)
        if self._heading_tag:
            self._heading_text.append(data)


class HtmlLinkRewriter(HTMLParser):
    def __init__(self, rewrite_value: 'LinkRewriteFunc') -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []
        self._rewrite_value = rewrite_value

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.parts.append(self._format_tag(tag, attrs, close=False))

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.parts.append(self._format_tag(tag, attrs, close=True))

    def handle_endtag(self, tag: str) -> None:
        self.parts.append(f'</{tag}>')

    def handle_data(self, data: str) -> None:
        self.parts.append(html.escape(data, quote=False))

    def handle_entityref(self, name: str) -> None:
        self.parts.append(f'&{name};')

    def handle_charref(self, name: str) -> None:
        self.parts.append(f'&#{name};')

    def handle_comment(self, data: str) -> None:
        self.parts.append(f'<!--{data}-->')

    def get_html(self) -> str:
        return ''.join(self.parts)

    def _format_tag(
        self, tag: str, attrs: list[tuple[str, str | None]], *, close: bool
    ) -> str:
        rendered_attrs = []
        for name, value in attrs:
            if value is None:
                rendered_attrs.append(name)
                continue
            if name in {'href', 'src'}:
                value = self._rewrite_value(name, value)
            escaped = html.escape(value, quote=True)
            rendered_attrs.append(f'{name}="{escaped}"')

        attr_text = f' {" ".join(rendered_attrs)}' if rendered_attrs else ''
        suffix = ' /' if close else ''
        return f'<{tag}{attr_text}{suffix}>'


LinkRewriteFunc = Callable[[str, str], str]


def normalize_base_url(raw: str) -> str:
    value = raw.strip()
    if not value or value == '/':
        return ''
    if not value.startswith('/'):
        raise DocsBuildError('--base-url must be an absolute path.')
    return value.rstrip('/')


def normalize_site_url(raw: str) -> str:
    value = raw.strip().rstrip('/')
    if not value.startswith(('http://', 'https://')):
        raise DocsBuildError('--site-url must start with http:// or https://.')
    return value


def site_path(base_url: str, route: str) -> str:
    if route == '/':
        return f'{base_url}/' if base_url else '/'
    return f'{base_url}{route}' if base_url else route


def site_asset_path(base_url: str, asset_name: str) -> str:
    return f'{base_url}/assets/{asset_name}' if base_url else f'/assets/{asset_name}'


def versioned_asset_path(
    base_url: str, asset_name: str, asset_versions: dict[str, str]
) -> str:
    version = asset_versions.get(asset_name)
    path = site_asset_path(base_url, asset_name)
    return f'{path}?v={version}' if version else path


def load_nav() -> tuple[dict[str, str], list[NavItem]]:
    if not NAV_PATH.is_file():
        raise DocsBuildError(f'Missing docs navigation file: {NAV_PATH}')

    data = yaml.safe_load(NAV_PATH.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise DocsBuildError('docs/nav.yml must contain a mapping.')

    site = data.get('site')
    if not isinstance(site, dict):
        raise DocsBuildError('docs/nav.yml must contain a site mapping.')

    items_data = data.get('items')
    if not isinstance(items_data, list) or not items_data:
        raise DocsBuildError('docs/nav.yml must contain a non-empty items list.')

    seen_routes: set[str] = set()
    items: list[NavItem] = []
    errors: list[str] = []

    for index, item_data in enumerate(items_data, start=1):
        if not isinstance(item_data, dict):
            errors.append(f'nav item {index} must be a mapping.')
            continue

        title = _required_string(item_data, 'title', index, errors)
        route = _required_string(item_data, 'route', index, errors)
        source_text = _required_string(item_data, 'source', index, errors)
        description = _required_string(
            item_data, 'description', index, errors
        )
        if not title or not route or not source_text or not description:
            continue

        if route != '/' and not (
            route.startswith('/') and route.endswith('/')
        ):
            errors.append(
                f'nav item {index} route must be / or an absolute directory path.'
            )
            continue
        if route in seen_routes:
            errors.append(f'duplicate docs route in nav: {route}')
            continue
        seen_routes.add(route)

        source = (DOCS_DIR / source_text).resolve()
        try:
            source.relative_to(DOCS_DIR.resolve())
        except ValueError:
            errors.append(f'nav source escapes docs/: {source_text}')
            continue
        if not source.is_file():
            errors.append(f'nav source is missing for {route}: {source_text}')
            continue

        items.append(
            NavItem(
                title=title,
                route=route,
                source=source,
                description=description,
            )
        )

    if errors:
        raise DocsBuildError('\n'.join(errors))
    return {str(key): str(value) for key, value in site.items()}, items


def _required_string(
    data: dict[str, object], key: str, index: int, errors: list[str]
) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f'nav item {index} must define {key}.')
        return ''
    return value.strip()


def require_pandoc() -> str:
    pandoc = shutil.which('pandoc')
    if pandoc is None:
        raise DocsBuildError(
            'Pandoc is required to build docs, but `pandoc` is not on PATH.'
        )
    return pandoc


def source_manifest() -> dict[str, object]:
    site, items = load_nav()

    if not TEMPLATE_PATH.is_file():
        raise DocsBuildError(f'Missing page template: {TEMPLATE_PATH}')

    assets = []
    for asset_name in ASSET_NAMES:
        source = DOCS_DIR / 'assets' / asset_name
        if not source.is_file():
            raise DocsBuildError(f'Missing docs asset: {source}')
        assets.append(_root_relative(source))

    pages = [_root_relative(item.source) for item in items]
    source_files = [
        _root_relative(Path(__file__).resolve()),
        _root_relative(NAV_PATH),
        *pages,
        _root_relative(TEMPLATE_PATH),
        *assets,
    ]

    return {
        'external_dependencies': ['pandoc'],
        'site': site,
        'routes': [
            {
                'title': item.title,
                'route': item.route,
                'source': _root_relative(item.source),
            }
            for item in items
        ],
        'source_files': source_files,
        'generated_files': [
            '_site/index.html',
            '_site/<route>/index.html',
            '_site/assets/docs.css',
            '_site/assets/docs.js',
            '_site/assets/eee-logo.png',
            '_site/assets/search-index.json',
        ],
    }


def _root_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def render_markdown(pandoc: str, source: Path) -> str:
    command = [
        pandoc,
        '--from',
        f'gfm{HTML_EXTENSIONS}',
        '--to',
        'html5',
        '--wrap=none',
        str(source),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise DocsBuildError(f'Pandoc failed for {source}:\n{message}')
    return result.stdout.strip()


def parse_html_info(html_text: str) -> HtmlInfoParser:
    parser = HtmlInfoParser()
    parser.feed(html_text)
    parser.close()
    return parser


def render_nav(items: list[NavItem], active_route: str, base_url: str) -> str:
    lines = ['<ol class="nav-list">']
    for item in items:
        active = item.route == active_route
        attrs = ' class="nav-link is-active" aria-current="page"'
        if not active:
            attrs = ' class="nav-link"'
        lines.append(
            '<li>'
            f'<a{attrs} href="{html.escape(site_path(base_url, item.route))}">'
            f'{html.escape(item.title)}</a>'
            '</li>'
        )
    lines.append('</ol>')
    return '\n'.join(lines)


def rewrite_page_links(
    body_html: str,
    *,
    current_route: str,
    route_set: set[str],
    anchors: dict[str, set[str]],
    base_url: str,
) -> str:
    errors: list[str] = []

    def rewrite_value(attr: str, value: str) -> str:
        return resolve_internal_url(
            value,
            current_route=current_route,
            route_set=route_set,
            anchors=anchors,
            base_url=base_url,
            attr=attr,
            errors=errors,
        )

    rewriter = HtmlLinkRewriter(rewrite_value)
    rewriter.feed(body_html)
    rewriter.close()
    if errors:
        raise DocsBuildError(
            f'Broken link(s) in {current_route}:\n' + '\n'.join(errors)
        )
    return rewriter.get_html()


def resolve_internal_url(
    value: str,
    *,
    current_route: str,
    route_set: set[str],
    anchors: dict[str, set[str]],
    base_url: str,
    attr: str,
    errors: list[str],
) -> str:
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc or value.startswith(('mailto:', 'tel:')):
        return value

    if parsed.path.startswith(LEGACY_INTERNAL_PATH):
        errors.append(f'{attr} uses legacy internal path: {value}')
        return value

    target_route: str | None = None
    target_path = parsed.path
    if target_path == '':
        target_route = current_route
    elif base_url and target_path.startswith(f'{base_url}/'):
        target_route = target_path[len(base_url) :] or '/'
    elif target_path.startswith('/'):
        target_route = target_path
    else:
        target_route = _resolve_relative_route(current_route, target_path)

    if target_route.startswith('/assets/'):
        asset_name = target_route.removeprefix('/assets/')
        if asset_name not in VALID_ASSET_NAMES:
            errors.append(f'{attr} points to unknown asset: {value}')
            return value
        rewritten = site_asset_path(base_url, asset_name)
        return urlunparse(parsed._replace(path=rewritten))

    normalized_route = _normalize_route_path(target_route)
    if normalized_route not in route_set:
        errors.append(f'{attr} points to unknown docs route: {value}')
        return value

    if parsed.fragment:
        target_anchors = anchors.get(normalized_route, set())
        if parsed.fragment not in target_anchors:
            errors.append(f'{attr} points to missing anchor: {value}')
            return value

    rewritten_path = site_path(base_url, normalized_route)
    return urlunparse(parsed._replace(path=rewritten_path))


def _resolve_relative_route(current_route: str, target_path: str) -> str:
    base_dir = current_route if current_route.endswith('/') else current_route
    resolved = posixpath.normpath(posixpath.join(base_dir, target_path))
    if not resolved.startswith('/'):
        resolved = f'/{resolved}'
    if target_path.endswith('/') and not resolved.endswith('/'):
        resolved = f'{resolved}/'
    return resolved


def _normalize_route_path(path: str) -> str:
    if not path or path == '.':
        return '/'
    normalized = posixpath.normpath(path)
    if not normalized.startswith('/'):
        normalized = f'/{normalized}'
    if '.' not in posixpath.basename(normalized) and not normalized.endswith('/'):
        normalized = f'{normalized}/'
    return normalized


def validate_full_html(
    pages: list[RenderedPage],
    *,
    base_url: str,
    route_set: set[str],
    anchors: dict[str, set[str]],
) -> None:
    errors: list[str] = []
    for page in pages:
        if re.search(r'\b(?:href|src)="/every_eval_ever', page.full_html):
            errors.append(
                f'{page.item.route} contains legacy {LEGACY_INTERNAL_PATH} link.'
            )
        info = parse_html_info(page.full_html)
        page_anchors = {
            **anchors,
            page.item.route: anchors.get(page.item.route, set())
            | info.element_ids,
        }
        for attr, value in info.links:
            link_errors: list[str] = []
            resolve_internal_url(
                value,
                current_route=page.item.route,
                route_set=route_set,
                anchors=page_anchors,
                base_url=base_url,
                attr=attr,
                errors=link_errors,
            )
            errors.extend(f'{page.item.route}: {error}' for error in link_errors)

    if errors:
        raise DocsBuildError('\n'.join(errors))


def load_template() -> str:
    if not TEMPLATE_PATH.is_file():
        raise DocsBuildError(f'Missing page template: {TEMPLATE_PATH}')
    return TEMPLATE_PATH.read_text(encoding='utf-8')


def render_template(
    template: str,
    *,
    site: dict[str, str],
    page: RenderedPage,
    nav_html: str,
    base_url: str,
    site_url: str,
    asset_versions: dict[str, str],
) -> str:
    title = required_site_value(site, 'title')
    description = required_site_value(site, 'description')
    repository_url = required_site_value(site, 'repository')
    datastore_url = required_site_value(site, 'datastore')
    canonical_url = f'{site_url}{site_path(base_url, page.item.route)}'
    replacements = {
        '{{base_url}}': html.escape(base_url),
        '{{canonical_url}}': html.escape(canonical_url),
        '{{description}}': html.escape(description),
        '{{docs_js}}': html.escape(
            versioned_asset_path(base_url, 'docs.js', asset_versions)
        ),
        '{{docs_css}}': html.escape(
            versioned_asset_path(base_url, 'docs.css', asset_versions)
        ),
        '{{logo}}': html.escape(
            versioned_asset_path(base_url, 'eee-logo.png', asset_versions)
        ),
        '{{nav}}': nav_html,
        '{{page_description}}': html.escape(page.item.description),
        '{{page_title}}': html.escape(page.item.title),
        '{{repository_url}}': html.escape(repository_url, quote=True),
        '{{search_index}}': html.escape(
            versioned_asset_path(base_url, SEARCH_INDEX, asset_versions)
        ),
        '{{site_title}}': html.escape(title),
        '{{datastore_url}}': html.escape(datastore_url, quote=True),
        '{{title}}': html.escape(f'{page.item.title} | {title}'),
        '{{content}}': page.body_html,
    }

    rendered = template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    missing = sorted(set(re.findall(r'{{[^}]+}}', rendered)))
    if missing:
        raise DocsBuildError(
            'Template has unresolved placeholders: ' + ', '.join(missing)
        )
    return rendered


def required_site_value(site: dict[str, str], key: str) -> str:
    value = site.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DocsBuildError(f'docs/nav.yml site.{key} must be set.')
    return value.strip()


def output_path_for_route(output_dir: Path, route: str) -> Path:
    if route == '/':
        return output_dir / 'index.html'
    return output_dir / route.strip('/') / 'index.html'


def write_outputs(
    *,
    output_dir: Path,
    pages: list[RenderedPage],
    search_index: list[dict[str, object]],
) -> None:
    stage = Path(tempfile.mkdtemp(prefix='eee-docs-build-'))

    for page in pages:
        html_path = output_path_for_route(stage, page.item.route)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(page.full_html, encoding='utf-8')

    asset_output = stage / 'assets'
    asset_output.mkdir(parents=True, exist_ok=True)
    for asset_name in ASSET_NAMES:
        source = DOCS_DIR / 'assets' / asset_name
        if not source.is_file():
            raise DocsBuildError(f'Missing docs asset: {source}')
        shutil.copy2(source, asset_output / asset_name)

    (asset_output / SEARCH_INDEX).write_text(
        json.dumps(search_index, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        trash_dir = trash_root()
        trash_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
        trash_target = trash_dir / f'{output_dir.name}.{stamp}.{uuid.uuid4().hex}'
        shutil.move(str(output_dir), str(trash_target))
    shutil.move(str(stage), str(output_dir))


def trash_root() -> Path:
    macos_trash = Path.home() / '.Trash'
    if macos_trash.exists():
        return macos_trash
    return Path(tempfile.gettempdir()) / 'Trash'


def build_search_index(
    pages: list[RenderedPage], *, base_url: str
) -> list[dict[str, object]]:
    index = []
    for page in pages:
        summary = ' '.join(page.text.split())
        if len(summary) > 280:
            summary = f'{summary[:277].rstrip()}...'
        index.append(
            {
                'title': page.item.title,
                'route': site_path(base_url, page.item.route),
                'summary': summary,
                'headings': [
                    heading['text'] for heading in (page.headings or [])
                ],
            }
        )
    return index


def build_asset_versions(
    search_index: list[dict[str, object]]
) -> dict[str, str]:
    versions = {
        asset_name: hash_file(DOCS_DIR / 'assets' / asset_name)
        for asset_name in ASSET_NAMES
    }
    encoded_index = (
        json.dumps(search_index, sort_keys=True, separators=(',', ':'))
        .encode('utf-8')
    )
    versions[SEARCH_INDEX] = hashlib.sha256(encoded_index).hexdigest()[:12]
    return versions


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def build_docs(base_url: str, site_url: str, output_dir: Path) -> None:
    base_url = normalize_base_url(base_url)
    site_url = normalize_site_url(site_url)
    site, items = load_nav()
    pandoc = require_pandoc()
    template = load_template()

    pages = [
        RenderedPage(item=item, body_html=render_markdown(pandoc, item.source))
        for item in items
    ]
    route_set = {item.route for item in items}
    anchors: dict[str, set[str]] = {}
    for page in pages:
        info = parse_html_info(page.body_html)
        page.headings = info.headings
        page.text = ' '.join(info.text_parts)
        anchors[page.item.route] = info.element_ids

    search_index = build_search_index(pages, base_url=base_url)
    asset_versions = build_asset_versions(search_index)

    for page in pages:
        page.body_html = rewrite_page_links(
            page.body_html,
            current_route=page.item.route,
            route_set=route_set,
            anchors=anchors,
            base_url=base_url,
        )
        page.full_html = render_template(
            template,
            site=site,
            page=page,
            nav_html=render_nav(items, page.item.route, base_url),
            base_url=base_url,
            site_url=site_url,
            asset_versions=asset_versions,
        )

    validate_full_html(
        pages, base_url=base_url, route_set=route_set, anchors=anchors
    )
    write_outputs(
        output_dir=output_dir,
        pages=pages,
        search_index=search_index,
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build the Every Eval Ever docs static site with Pandoc.'
    )
    parser.add_argument(
        '--print-source-manifest',
        action='store_true',
        help='Print the exact Pandoc docs source file manifest as JSON.',
    )
    parser.add_argument(
        '--base-url',
        help='Absolute URL path where the docs artifact will be mounted.',
    )
    parser.add_argument(
        '--site-url',
        help='Origin for canonical URLs, such as https://evalevalai.com.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Directory where the static site artifact is written.',
    )
    args = parser.parse_args(argv)
    if not args.print_source_manifest:
        missing = [
            option
            for option, value in (
                ('--base-url', args.base_url),
                ('--site-url', args.site_url),
                ('--output', args.output),
            )
            if value is None
        ]
        if missing:
            parser.error(
                'the following arguments are required unless '
                f'--print-source-manifest is used: {", ".join(missing)}'
            )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.print_source_manifest:
            print(json.dumps(source_manifest(), indent=2, sort_keys=True))
        else:
            build_docs(args.base_url, args.site_url, args.output)
    except DocsBuildError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
