"""Keep runnable documentation commands inside the uv project environment."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

COMMAND_PATTERNS = {
    'python': re.compile(r'\bpython3?\s+(?:-m\b|-c\b|[^\s`]+\.py\b)'),
    'pip install': re.compile(r'\bpip3?\s+install\b'),
    'pytest': re.compile(r'\bpytest\s+(?:-[^\s`]+|[^\s`]+)'),
    'ruff check': re.compile(r'\bruff\s+check\b'),
    'datamodel-codegen': re.compile(r'\bdatamodel-codegen\b'),
}


def documentation_paths() -> list[Path]:
    """Return committed documentation surfaces that contain runnable commands."""
    paths = {
        REPO_ROOT / 'AGENTS.md',
        REPO_ROOT / 'CONTRIBUTING.md',
        REPO_ROOT / 'README.md',
    }
    for root in (
        REPO_ROOT / '.agents',
        REPO_ROOT / '.github',
        REPO_ROOT / 'every_eval_ever',
        REPO_ROOT / 'tests' / 'data',
    ):
        paths.update(root.rglob('*.md'))
    return sorted(path for path in paths if path.is_file())


def test_documented_commands_use_uv() -> None:
    """Reject bare Python and project-tool commands in documentation."""
    violations: list[str] = []
    for path in documentation_paths():
        for line_number, line in enumerate(
            path.read_text(encoding='utf-8').splitlines(), start=1
        ):
            lowered = line.lower()
            for command, pattern in COMMAND_PATTERNS.items():
                for match in pattern.finditer(lowered):
                    prefix = lowered[: match.start()]
                    uses_uv = (
                        'uv run' in prefix
                        if command != 'pip install'
                        else 'uv ' in prefix
                    )
                    if not uses_uv:
                        relative_path = path.relative_to(REPO_ROOT)
                        violations.append(
                            f'{relative_path}:{line_number}: bare {command}: '
                            f'{line.strip()}'
                        )

    assert not violations, (
        'Run documented project commands through uv:\n' + '\n'.join(violations)
    )
