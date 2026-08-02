"""
Adapter to convert HAL (Holistic Agent Leaderboard) data into the EEE schema.

Data source:
    https://hal.cs.princeton.edu/
    Nine benchmarks: AssistantBench, CORE-Bench Hard, GAIA, Online Mind2Web,
    Scicode, ScienceAgentBench, SWE-bench Verified Mini, TAU-bench Airline, USACO

Each benchmark page embeds the leaderboard as an HTML table. This adapter:
  1. Fetches each benchmark page
  2. Parses the HTML table rows
  3. Produces one EEE JSON file per (benchmark, agent, model) entry

Output structure:
    data/hal-{benchmark_slug}/{developer}/{model_slug}/{uuid}.json

Usage:
    cd every_eval_ever
    uv run python -m utils.hal.adapter
    uv run python -m utils.hal.adapter --benchmark gaia
    uv run python -m utils.hal.adapter --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import asdict, dataclass, field
from html import unescape
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.helpers import SCHEMA_VERSION
from every_eval_ever.helpers.io import (
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordExclusion,
    SourceRecordFailure,
    default_failure_report_path,
    require_identity,
    save_evaluation_logs,
    save_failure_report,
)

HAL_BASE_URL = 'https://hal.cs.princeton.edu'

# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------


@dataclass
class ToolDef:
    name: str
    description: Optional[str] = None


@dataclass
class BenchmarkDef:
    slug: str  # URL path segment on hal.cs.princeton.edu
    name: str  # Human-readable name
    output_name: str  # Directory name under data/
    category: str  # Category label
    dataset_url: str  # Canonical dataset/benchmark URL
    description: str  # Short description of what is measured
    metric_description: str  # Description of the primary accuracy metric
    # Tools available to agents in this benchmark
    tools: list[ToolDef] = field(default_factory=list)
    extra_metrics: list[dict] = field(default_factory=list)  # GAIA levels etc.
    # Optional extra key/value pairs added to source_data.additional_details
    source_data_details: dict[str, str] = field(default_factory=dict)


BENCHMARKS: list[BenchmarkDef] = [
    BenchmarkDef(
        slug='assistantbench',
        name='AssistantBench',
        output_name='hal-assistantbench',
        category='Web Assistance',
        dataset_url='https://assistantbench.github.io',
        description=(
            'HAL evaluates AI agents on a 33-task split of AssistantBench, '
            'a benchmark of realistic, time-consuming, and automatically verifiable '
            'web assistance tasks based on real human needs. '
            'The full benchmark contains 214 tasks.'
        ),
        metric_description="Accuracy on HAL's 33-task AssistantBench split (0.0–1.0)",
        tools=[
            ToolDef('browser', 'Navigate and interact with live web pages'),
            ToolDef('web_search', 'Search the web for information'),
        ],
        source_data_details={
            'tasks_evaluated': '33',
            'full_benchmark_size': '214',
            'note': 'HAL evaluates on a 33-task subset; full AssistantBench has 214 tasks',
        },
    ),
    BenchmarkDef(
        slug='corebench_hard',
        name='CORE-Bench Hard',
        output_name='hal-corebench-hard',
        category='Scientific Programming',
        dataset_url='https://github.com/siegelz/core-bench',
        description=(
            'CORE-Bench Hard tests agents on hard computational reproducibility tasks '
            'drawn from published scientific papers.'
        ),
        metric_description='Fraction of CORE-Bench Hard tasks solved (0.0–1.0)',
        tools=[
            ToolDef('bash', 'Execute shell commands'),
            ToolDef('python', 'Execute Python code'),
            ToolDef('read_file', 'Read files from the filesystem'),
            ToolDef('write_file', 'Write files to the filesystem'),
        ],
    ),
    BenchmarkDef(
        slug='gaia',
        name='GAIA',
        output_name='hal-gaia',
        category='Web Assistance',
        dataset_url='https://huggingface.co/datasets/gaia-benchmark/GAIA',
        description=(
            'GAIA (General AI Assistants) measures whether AI agents can answer '
            'real-world questions requiring multi-step reasoning and tool use. '
            'Questions are divided into 3 difficulty levels.'
        ),
        metric_description='Overall accuracy on GAIA validation set (0.0–1.0)',
        tools=[
            ToolDef('web_search', 'Search the web for information'),
            ToolDef('browser', 'Navigate and interact with live web pages'),
            ToolDef('python', 'Execute Python code for computation'),
            ToolDef('read_file', 'Read and process files'),
        ],
        extra_metrics=[
            {
                'key': 'level1',
                'name': 'GAIA Level 1',
                'description': 'Accuracy on Level 1 questions (simplest)',
            },
            {
                'key': 'level2',
                'name': 'GAIA Level 2',
                'description': 'Accuracy on Level 2 questions (moderate)',
            },
            {
                'key': 'level3',
                'name': 'GAIA Level 3',
                'description': 'Accuracy on Level 3 questions (hardest)',
            },
        ],
    ),
    BenchmarkDef(
        slug='online_mind2web',
        name='Online Mind2Web',
        output_name='hal-online-mind2web',
        category='Web Assistance',
        dataset_url='https://osu-nlp-group.github.io/Mind2Web/',
        description=(
            'Online Mind2Web evaluates web agents on live website tasks '
            'requiring multi-step navigation and interaction.'
        ),
        metric_description='Task success rate on Online Mind2Web (0.0–1.0)',
        tools=[
            ToolDef('browser', 'Navigate and interact with live web pages'),
            ToolDef('click', 'Click on web page elements'),
            ToolDef('type', 'Type text into web page inputs'),
            ToolDef('scroll', 'Scroll web pages'),
        ],
    ),
    BenchmarkDef(
        slug='scicode',
        name='Scicode',
        output_name='hal-scicode',
        category='Scientific Programming',
        dataset_url='https://scicode-bench.github.io',
        description=(
            'Scicode tests agents on scientific coding problems spanning '
            'mathematics, physics, chemistry, biology, and material science.'
        ),
        metric_description='Fraction of Scicode problems solved (0.0–1.0)',
        tools=[
            ToolDef('python', 'Execute Python code for scientific computation'),
            ToolDef('bash', 'Execute shell commands'),
        ],
    ),
    BenchmarkDef(
        slug='scienceagentbench',
        name='ScienceAgentBench',
        output_name='hal-scienceagentbench',
        category='Scientific Programming',
        dataset_url='https://osu-nlp-group.github.io/ScienceAgentBench/',
        description=(
            'ScienceAgentBench evaluates language agents on end-to-end data-driven '
            'scientific discovery tasks drawn from peer-reviewed publications.'
        ),
        metric_description='Success rate on ScienceAgentBench tasks (0.0–1.0)',
        tools=[
            ToolDef('python', 'Execute Python code for data analysis'),
            ToolDef('bash', 'Execute shell commands'),
            ToolDef('read_file', 'Read datasets and files'),
            ToolDef('write_file', 'Write output files and results'),
        ],
    ),
    BenchmarkDef(
        slug='swebench_verified_mini',
        name='SWE-bench Verified Mini',
        output_name='hal-swebench-verified-mini',
        category='Software Engineering',
        dataset_url='https://www.swebench.com',
        description=(
            'SWE-bench Verified Mini is a 50-instance subset of SWE-bench Verified, '
            'requiring agents to resolve real GitHub issues.'
        ),
        metric_description='Fraction of 50 verified GitHub issues resolved (0.0–1.0)',
        tools=[
            ToolDef('bash', 'Execute shell commands'),
            ToolDef('edit_file', 'Edit files in the repository'),
            ToolDef('read_file', 'Read files from the repository'),
        ],
    ),
    BenchmarkDef(
        slug='taubench_airline',
        name='TAU-bench Airline',
        output_name='hal-taubench-airline',
        category='Customer Service',
        dataset_url='https://github.com/sierra-research/tau-bench',
        description=(
            'TAU-bench Airline tests tool-augmented language models on realistic '
            'airline customer service tasks with complex policies.'
        ),
        metric_description='Task success rate on TAU-bench Airline (0.0–1.0)',
        tools=[
            ToolDef(
                'function_calling',
                'Call predefined airline service API functions',
            ),
        ],
    ),
    BenchmarkDef(
        slug='usaco',
        name='USACO',
        output_name='hal-usaco',
        category='Programming',
        dataset_url='https://usaco.guide',
        description=(
            'USACO evaluates agents on competitive programming problems from the '
            'USA Computing Olympiad across multiple difficulty levels.'
        ),
        metric_description='Fraction of USACO problems solved (0.0–1.0)',
        tools=[
            ToolDef('bash', 'Execute shell commands and compile/run code'),
            ToolDef('python', 'Execute Python code'),
        ],
    ),
]

BENCHMARK_BY_SLUG: dict[str, BenchmarkDef] = {b.slug: b for b in BENCHMARKS}

# ---------------------------------------------------------------------------
# Model name → developer mapping
# ---------------------------------------------------------------------------

# HAL model names (as displayed on the leaderboard) mapped to EEE developer slugs.
# These override the automatic pattern-matching in helpers/developer.py.
MODEL_DEVELOPER_MAP: dict[str, str] = {
    # OpenAI
    'o1': 'openai',
    'o3': 'openai',
    'o4': 'openai',
    'gpt': 'openai',
    # Anthropic
    'claude': 'anthropic',
    # Google
    'gemini': 'google',
    'gemma': 'google',
    # Meta
    'llama': 'meta',
    # DeepSeek
    'deepseek': 'deepseek',
    # Mistral
    'mistral': 'mistralai',
    'mixtral': 'mistralai',
    # Qwen / Alibaba — use "qwen" to match all other adapters in this repo
    'qwen': 'qwen',
    # Microsoft
    'phi': 'microsoft',
}

# Maps cleaned model name → canonical EEE model ID
# This handles date-stripped versions of model names.
MODEL_ID_OVERRIDES: dict[str, str] = {
    'claude-3-7-sonnet': 'anthropic/claude-3-7-sonnet-20250219',
    'claude-3-5-sonnet': 'anthropic/claude-3-5-sonnet-20241022',
    'claude-3-5-haiku': 'anthropic/claude-3-5-haiku-20241022',
    'claude-3-opus': 'anthropic/claude-3-opus-20240229',
    'claude opus 4.5': 'anthropic/claude-opus-4-5',
    'claude sonnet 4.5': 'anthropic/claude-sonnet-4-5',
    'claude opus 4.1': 'anthropic/claude-opus-4-1',
    'claude sonnet 4.1': 'anthropic/claude-sonnet-4-1',
    'claude haiku 4.1': 'anthropic/claude-haiku-4-1',
    'claude-3.7 sonnet': 'anthropic/claude-3-7-sonnet-20250219',
    'gemini 2.0 flash': 'google/gemini-2.0-flash',
    'gemini 2.5 pro': 'google/gemini-2.5-pro',
    'gemini 2.5 flash': 'google/gemini-2.5-flash',
    'gemini 1.5 pro': 'google/gemini-1.5-pro',
    'deepseek r1': 'deepseek/deepseek-r1',
    'deepseek v3': 'deepseek/deepseek-v3',
    'gpt-5': 'openai/gpt-5',
    'gpt-4.1': 'openai/gpt-4.1',
    'gpt-4o': 'openai/gpt-4o',
    'o1': 'openai/o1',
    'o3': 'openai/o3',
    'o4-mini': 'openai/o4-mini',
    'llama-4-maverick': 'meta-llama/llama-4-maverick',
    'llama-4-scout': 'meta-llama/llama-4-scout',
    'qwen3-235b': 'qwen/qwen3-235b',
    'qwen3-32b': 'qwen/qwen3-32b',
}


def _normalize_model_name(raw: str) -> str:
    """
    Strip parenthetical date/effort suffixes and normalize whitespace.

    Examples:
        "o3 Medium (April 2025)"           → "o3"
        "Claude Sonnet 4.5 (September 2025)"→ "claude sonnet 4.5"
        "Claude Sonnet 4.5 High (September 2025)" → "claude sonnet 4.5 high"
        "Claude Opus 4.5"                  → "claude opus 4.5"
        "DeepSeek R1 (January 2025)"       → "deepseek r1"
    """
    # Remove trailing (Month Year) date annotation
    cleaned = re.sub(r'\s*\(\w+\s+\d{4}\)\s*$', '', raw).strip()
    return cleaned.lower()


def _effort_level(raw: str) -> Optional[str]:
    """Extract effort level like Low/Medium/High from model name."""
    m = re.search(r'\b(low|medium|high)\b', raw, re.IGNORECASE)
    return m.group(1).lower() if m else None


def get_developer_for_model(raw_model: str) -> str:
    """Resolve developer name from a HAL model name string."""
    lower = raw_model.lower()
    for prefix, dev in MODEL_DEVELOPER_MAP.items():
        if lower.startswith(prefix) or f' {prefix}' in lower:
            return dev
    return 'unknown'


def get_model_id(raw_model: str) -> str:
    """
    Produce a canonical EEE model ID from a HAL model name.

    Strategy:
      1. Check explicit overrides (case-insensitive)
      2. Strip date suffix, slugify, prepend developer
    """
    cleaned = _normalize_model_name(raw_model)

    # Check overrides — longest key first to avoid shorter keys shadowing more
    # specific ones (e.g. "gpt-4.1" must not match "gpt-4.1-mini").
    # A prefix match is only accepted when the remaining suffix is solely a
    # known inference-effort token (low / medium / high).
    _EFFORT_SUFFIX = re.compile(r'^(low|medium|high)$', re.IGNORECASE)
    for key, model_id in sorted(
        MODEL_ID_OVERRIDES.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        if cleaned == key:
            return model_id
        if cleaned.startswith(f'{key} '):
            suffix = cleaned[len(key) :].strip()
            if _EFFORT_SUFFIX.fullmatch(suffix):
                return model_id

    # Auto-derive: developer/slugified-name
    developer = get_developer_for_model(raw_model)
    # Remove effort level from the slug
    slug = re.sub(r'\b(low|medium|high)\b', '', cleaned, flags=re.IGNORECASE)
    slug = re.sub(r'\s+', '-', slug.strip()).strip('-')
    slug = re.sub(r'-+', '-', slug)
    if developer != 'unknown':
        return f'{developer}/{slug}'
    raise ValueError(f'Cannot determine HAL model developer from {raw_model!r}')


def slugify(text: str) -> str:
    """Convert text to a URL/path safe slug."""
    text = text.strip().lower()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    text = re.sub(r'-+', '-', text).strip('-')
    return text


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------


def _fetch_page(url: str) -> str:
    """Fetch a URL and return the HTML body as a string."""
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0 (EEE-adapter)'})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode('utf-8', errors='replace')


def _clean_cell(html_fragment: str) -> str:
    """Strip HTML tags and collapse whitespace from a table cell fragment."""
    text = re.sub(r'<[^>]+>', ' ', html_fragment)
    text = unescape(text)
    return re.sub(r'\s+', ' ', text).strip()


def _parse_percent(value: str) -> Optional[float]:
    """Parse '38.81%' or '38.81% (-0.3/+0.4)' → 0.3881 (fraction 0-1)."""
    m = re.search(r'(?<![\d.,])(\d[\d,]*(?:\.\d+)?)(?=\s*%)', value)
    if m:
        try:
            return round(float(m.group(1).replace(',', '')) / 100.0, 6)
        except ValueError:
            return None
    return None


def _parse_cost(value: str) -> Optional[float]:
    """Parse '$15.15' or '$178.20 (-9.17/+9.17)' → 15.15."""
    m = re.search(r'\$(\d[\d,]*(?:\.\d+)?)', value)
    if m:
        try:
            return float(m.group(1).replace(',', ''))
        except ValueError:
            return None
    return None


def _parse_ci(value: str) -> Optional[str]:
    """Extract confidence interval string like '(-0.3/+0.4)'."""
    m = re.search(r'\(([^)]+)\)', value)
    return m.group(1) if m else None


@dataclass
class LeaderboardRow:
    rank: int
    agent_name: str
    model_raw: str
    verified: bool
    is_pareto: bool
    accuracy: float  # 0.0–1.0
    accuracy_ci: Optional[str]
    cost_usd: Optional[float]
    cost_ci: Optional[str]
    runs: Optional[int]
    extra_scores: dict[str, Optional[float]] = field(default_factory=dict)
    notes: Optional[str] = None  # e.g. "(95.5% w/ manual validation)"


def _parse_agent_cell(raw: str) -> tuple[str, bool, bool, Optional[str]]:
    """
    Parse the agent/scaffold cell which can contain:
      "Browser-Use Pareto optimal"
      "Claude Code Submitted by Nicholas Carlini Download main.py Pareto optimal"

    Returns: (agent_name, is_pareto, has_submitter_link, submitter_info)
    """
    is_pareto = 'Pareto optimal' in raw
    # Remove marker text
    name = raw.replace('Pareto optimal', '').strip()
    # Remove "Submitted by ... Download main.py" patterns
    submitter = None
    m = re.search(
        r'Submitted by (.+?)(?:Download\s+\w+\.?\w*)?$', name, re.IGNORECASE
    )
    if m:
        submitter = m.group(1).strip()
        name = re.sub(r'Submitted by .+', '', name).strip()
    # Clean trailing "Download *.py" style text
    name = re.sub(r'\s+Download\s+\S+', '', name).strip()
    name = re.sub(r'\s+', ' ', name).strip()
    return name, is_pareto, bool(submitter), submitter


def _parse_accuracy_cell(
    raw: str,
) -> tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Parse accuracy cell, which may include notes like '(95.5% w/ manual validation)'.
    Returns: (accuracy_fraction, confidence_interval, notes_string)
    """
    # Notes: secondary score in parens like "(95.5% w/ manual validation)"
    notes = None
    m_notes = re.search(r'\(([^)]*%[^)]*)\)', raw)
    if m_notes:
        notes = m_notes.group(1)

    accuracy = _parse_percent(raw)
    ci_m = re.search(r'%\s*\(([+-][\d.]+/[+-][\d.]+)\)', raw)
    ci = ci_m.group(1) if ci_m else None
    return accuracy, ci, notes


def parse_table_result(
    html: str,
    benchmark: BenchmarkDef,
) -> SourceConversionResult[LeaderboardRow]:
    """Parse data rows and retain malformed/non-data row provenance."""
    m = re.search(r'<tbody[^>]*>(.*?)</tbody>', html, re.DOTALL)
    if not m:
        raise ValueError(f'No <tbody> found on {benchmark.slug} page')

    tbody = m.group(1)
    raw_rows = re.findall(r'<tr[^>]*>(.*?)</tr>', tbody, re.DOTALL)
    results: list[LeaderboardRow] = []
    failures: list[SourceRecordFailure] = []
    exclusions: list[SourceRecordExclusion] = []

    is_gaia = bool(benchmark.extra_metrics)
    has_extra = len(benchmark.extra_metrics)

    for row_index, raw_row in enumerate(raw_rows):
        row_ref = f'{benchmark.slug} HTML table row {row_index + 1}'
        raw_cells = re.findall(r'<td[^>]*>(.*?)</td>', raw_row, re.DOTALL)
        cells = [_clean_cell(c) for c in raw_cells]

        if len(cells) < 6:
            exclusions.append(
                SourceRecordExclusion(
                    source_ref=row_ref,
                    reason='not a leaderboard data row (fewer than 6 cells)',
                    source_record={'cells': cells},
                )
            )
            continue

        rank_text = re.sub(r'\D', '', cells[0])
        if not rank_text:
            exclusions.append(
                SourceRecordExclusion(
                    source_ref=row_ref,
                    reason='not a ranked leaderboard row',
                    source_record={'cells': cells},
                )
            )
            continue
        rank = int(rank_text)

        agent_name, is_pareto, _, submitter = _parse_agent_cell(cells[1])
        model_raw = cells[2]
        verified = '✓' in cells[3]

        accuracy_raw = cells[4]
        accuracy, accuracy_ci, notes = _parse_accuracy_cell(accuracy_raw)
        if accuracy is None:
            failures.append(
                SourceRecordFailure(
                    source_ref=f'{row_ref} accuracy',
                    reason=f'could not parse percentage from {accuracy_raw!r}',
                    source_record={'cells': cells},
                )
            )
            continue

        # For GAIA: cells[5], [6], [7] are level scores; cost is cells[8]
        extra_scores: dict[str, Optional[float]] = {}
        cost_idx = 5
        if is_gaia and has_extra > 0:
            for i, em in enumerate(benchmark.extra_metrics):
                cell_idx = 5 + i
                if cell_idx < len(cells):
                    raw_extra = cells[cell_idx]
                    parsed_extra = _parse_percent(raw_extra)
                    extra_scores[em['key']] = parsed_extra
                    if raw_extra and parsed_extra is None:
                        failures.append(
                            SourceRecordFailure(
                                source_ref=(f'{row_ref} metric {em["key"]!r}'),
                                reason=(
                                    'could not parse percentage from '
                                    f'{raw_extra!r}'
                                ),
                                source_record={
                                    'cells': cells,
                                    'metric': em['key'],
                                    'value': raw_extra,
                                },
                            )
                        )
            cost_idx = 5 + has_extra

        cost_raw = cells[cost_idx] if cost_idx < len(cells) else ''
        cost_usd = _parse_cost(cost_raw)
        if (
            cost_raw
            and cost_raw.lower() not in {'n/a', 'na', '-', '—'}
            and cost_usd is None
        ):
            failures.append(
                SourceRecordFailure(
                    source_ref=f'{row_ref} cost',
                    reason=f'could not parse cost from {cost_raw!r}',
                    source_record={'cells': cells, 'value': cost_raw},
                )
            )
        cost_ci = None
        ci_m = re.search(r'\(([+-][\d.]+/[+-][\d.]+)\)', cost_raw)
        if ci_m:
            cost_ci = ci_m.group(1)

        runs_idx = cost_idx + 1
        runs = None
        if runs_idx < len(cells):
            runs_raw = cells[runs_idx].strip()
            try:
                if runs_raw:
                    runs = int(runs_raw)
            except ValueError as exc:
                failures.append(
                    SourceRecordFailure(
                        source_ref=f'{row_ref} runs',
                        reason=f'invalid run count: {exc}',
                        source_record={'cells': cells, 'value': runs_raw},
                    )
                )

        results.append(
            LeaderboardRow(
                rank=rank,
                agent_name=agent_name,
                model_raw=model_raw,
                verified=verified,
                is_pareto=is_pareto,
                accuracy=accuracy,
                accuracy_ci=accuracy_ci,
                cost_usd=cost_usd,
                cost_ci=cost_ci,
                runs=runs,
                extra_scores=extra_scores,
                notes=notes,
            )
        )

    return SourceConversionResult(
        source_name=f'HAL {benchmark.name} HTML table',
        total_records=len(raw_rows),
        records=results,
        failures=failures,
        exclusions=exclusions,
    )


def parse_table(html: str, benchmark: BenchmarkDef) -> list[LeaderboardRow]:
    """Strict parser for callers that require every data row to parse."""
    result = parse_table_result(html, benchmark)
    result.raise_if_incomplete()
    return result.records


# ---------------------------------------------------------------------------
# EEE record building
# ---------------------------------------------------------------------------


def build_source_data(benchmark: BenchmarkDef) -> dict:
    result: dict = {
        'source_type': 'url',
        'dataset_name': benchmark.name,
        'url': [
            benchmark.dataset_url,
            f'{HAL_BASE_URL}/{benchmark.slug}',
        ],
    }
    if benchmark.source_data_details:
        result['additional_details'] = benchmark.source_data_details
    return result


def build_evaluation_result(
    evaluation_name: str,
    score: float,
    description: str,
    benchmark: BenchmarkDef,
    row: LeaderboardRow,
    details: Optional[dict] = None,
) -> dict:
    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f'{evaluation_name} score must be between 0 and 1, got {score!r}'
        )
    score_details: dict = {'score': round(score, 6)}
    if details:
        score_details['details'] = {
            k: str(v) for k, v in details.items() if v is not None
        }

    available_tools = [
        {
            'name': t.name,
            **({'description': t.description} if t.description else {}),
        }
        for t in benchmark.tools
    ]

    generation_details = {
        'agent_scaffold': row.agent_name,
        'hal_rank': str(row.rank),
        'verified': str(row.verified),
        'is_pareto': str(row.is_pareto),
        **(
            {'total_cost_usd': str(row.cost_usd)}
            if row.cost_usd is not None
            else {}
        ),
        **({'cost_confidence_interval': row.cost_ci} if row.cost_ci else {}),
        **(
            {'accuracy_confidence_interval': row.accuracy_ci}
            if row.accuracy_ci
            else {}
        ),
        **({'notes': row.notes} if row.notes else {}),
    }
    if row.runs is not None:
        generation_details['runs'] = str(row.runs)

    return {
        'evaluation_name': evaluation_name,
        'source_data': build_source_data(benchmark),
        'metric_config': {
            'evaluation_description': description,
            'lower_is_better': False,
            'score_type': 'continuous',
            'min_score': 0.0,
            'max_score': 1.0,
        },
        'score_details': score_details,
        'generation_config': {
            'generation_args': {
                'agentic_eval_config': {
                    'available_tools': available_tools,
                },
            },
            'additional_details': generation_details,
        },
    }


def build_eee_record(
    benchmark: BenchmarkDef,
    row: LeaderboardRow,
    retrieved_timestamp: str,
) -> tuple[dict, str, str]:
    """
    Build one EEE JSON record for a single leaderboard row.

    Returns: (record_dict, developer_slug, model_slug)
    """
    model_id = get_model_id(row.model_raw)
    if '/' not in model_id:
        raise ValueError(f'HAL model id must be developer/model: {model_id!r}')
    developer, model_slug_clean = model_id.split('/', 1)
    developer = require_identity(developer, 'HAL model developer')
    model_slug_clean = require_identity(model_slug_clean, 'HAL model name')

    effort = _effort_level(row.model_raw)
    agent_slug = slugify(row.agent_name)
    # Use the raw HAL model name (slugified) in the evaluation_id so that
    # variants that share the same normalized model_info.id — e.g. effort
    # levels ("High") or dated releases ("April 2025") — produce distinct IDs.
    raw_model_slug = slugify(row.model_raw)

    eval_id = (
        f'{benchmark.output_name}/{agent_slug}'
        f'/{raw_model_slug}/{retrieved_timestamp}'
    )

    # Primary evaluation result
    eval_results = [
        build_evaluation_result(
            evaluation_name=benchmark.name,
            score=row.accuracy,
            description=benchmark.metric_description,
            benchmark=benchmark,
            row=row,
            details={
                'accuracy_raw': f'{row.accuracy * 100:.2f}%',
            },
        )
    ]

    # Extra metrics (e.g. GAIA levels)
    for em in benchmark.extra_metrics:
        score = row.extra_scores.get(em['key'])
        if score is not None:
            eval_results.append(
                build_evaluation_result(
                    evaluation_name=f'{benchmark.name} - {em["name"].split(" - ")[-1] if " - " in em["name"] else em["name"]}',
                    score=score,
                    description=em['description'] + ' (0.0–1.0)',
                    benchmark=benchmark,
                    row=row,
                )
            )

    # Model info additional details
    model_additional: dict[str, str] = {
        'hal_model_name': row.model_raw,
        'agent_scaffold': row.agent_name,
        'benchmark': benchmark.name,
        'deployment_type': 'unknown',
        'model_availability': 'unknown',
    }
    if effort:
        model_additional['inference_effort'] = effort
    if row.cost_usd is not None:
        model_additional['total_cost_usd'] = str(row.cost_usd)

    record = {
        'schema_version': SCHEMA_VERSION,
        'evaluation_id': eval_id,
        'retrieved_timestamp': retrieved_timestamp,
        'source_metadata': {
            'source_name': f'HAL Leaderboard — {benchmark.name}',
            'source_type': 'documentation',
            'source_organization_name': 'Princeton SAgE Team',
            'source_organization_url': HAL_BASE_URL,
            'evaluator_relationship': 'third_party',
            'additional_details': {
                'paper': 'https://arxiv.org/pdf/2510.11977',
                'benchmark_category': benchmark.category,
                'benchmark_slug': benchmark.slug,
            },
        },
        'eval_library': {
            'name': 'HAL',
            'version': 'unknown',
        },
        'model_info': {
            'name': row.model_raw,
            'id': model_id,
            'developer': developer,
            'additional_details': model_additional,
        },
        'evaluation_results': eval_results,
    }

    return record, developer, slugify(model_slug_clean)


# ---------------------------------------------------------------------------
# Collection and publication
# ---------------------------------------------------------------------------


def collect_benchmark(
    benchmark: BenchmarkDef,
    out_root: Path,
    retrieved_timestamp: str,
) -> SourceConversionResult[EvaluationLogOutput]:
    """Fetch and convert one benchmark without discarding usable rows."""
    url = f'{HAL_BASE_URL}/{benchmark.slug}'
    print(f'\n{"=" * 60}')
    print(f'Fetching {benchmark.name} from {url} …')

    try:
        html = _fetch_page(url)
    except (OSError, TimeoutError, URLError) as exc:
        return SourceConversionResult(
            source_name=f'HAL {benchmark.name}',
            total_records=1,
            records=[],
            failures=[
                SourceRecordFailure(
                    source_ref=url,
                    reason=f'failed to fetch leaderboard page: {exc}',
                )
            ],
        )

    try:
        parsed = parse_table_result(html, benchmark)
    except ValueError as exc:
        return SourceConversionResult(
            source_name=f'HAL {benchmark.name}',
            total_records=1,
            records=[],
            failures=[
                SourceRecordFailure(
                    source_ref=url,
                    reason=f'failed to parse leaderboard table: {exc}',
                )
            ],
        )

    print(f'  Found {len(parsed.records)} leaderboard entries')

    benchmark_out_root = out_root / benchmark.output_name
    outputs = []
    failures = list(parsed.failures)
    for row in parsed.records:
        try:
            record, developer, model_slug = build_eee_record(
                benchmark,
                row,
                retrieved_timestamp,
            )
            outputs.append(
                EvaluationLogOutput(
                    eval_log=EvaluationLog.model_validate(record),
                    base_dir=benchmark_out_root,
                    developer=developer,
                    model_name=model_slug,
                )
            )
        except Exception as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=(
                        f'{benchmark.slug} leaderboard rank {row.rank}'
                    ),
                    reason=str(exc),
                    source_record=asdict(row),
                )
            )

    if not parsed.records and not failures:
        failures.append(
            SourceRecordFailure(
                source_ref=url,
                reason='leaderboard contained no data rows',
            )
        )

    return SourceConversionResult(
        source_name=f'HAL {benchmark.name}',
        total_records=parsed.total_records,
        records=outputs,
        failures=failures,
        exclusions=parsed.exclusions,
    )


def process_benchmark(
    benchmark: BenchmarkDef,
    out_root: Path,
    retrieved_timestamp: str,
) -> tuple[int, int]:
    """Publish one benchmark, preserving successes and reporting failures."""
    result = collect_benchmark(benchmark, out_root, retrieved_timestamp)
    paths = save_evaluation_logs(result.records)
    if result.failures or result.exclusions:
        save_failure_report(
            result,
            default_failure_report_path(out_root / benchmark.output_name),
        )
    result.raise_if_incomplete()
    return len(paths), len(result.failures)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Fetch HAL leaderboard data and convert to EEE schema.'
    )
    parser.add_argument(
        '--benchmark',
        choices=list(BENCHMARK_BY_SLUG.keys()) + ['all'],
        default='all',
        help='Which benchmark(s) to fetch (default: all)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data'),
        help='Root output directory (default: data/)',
    )
    args = parser.parse_args()

    benchmarks = (
        list(BENCHMARK_BY_SLUG.values())
        if args.benchmark == 'all'
        else [BENCHMARK_BY_SLUG[args.benchmark]]
    )

    retrieved_timestamp = str(time.time())
    results = [
        collect_benchmark(
            benchmark,
            args.output_dir,
            retrieved_timestamp,
        )
        for benchmark in benchmarks
    ]
    combined = SourceConversionResult(
        source_name='HAL',
        total_records=sum(result.total_records for result in results),
        records=[record for result in results for record in result.records],
        failures=[failure for result in results for failure in result.failures],
        exclusions=[
            exclusion for result in results for exclusion in result.exclusions
        ],
    )

    paths = save_evaluation_logs(combined.records)
    for path in paths:
        print(f'  Saved: {path.relative_to(args.output_dir)}')
    if combined.failures or combined.exclusions:
        report_path = save_failure_report(
            combined,
            default_failure_report_path(args.output_dir / 'hal'),
        )
        print(f'Provenance report: {report_path}')

    print(f'\n{"=" * 60}')
    print(
        f'Done. Saved {len(paths)} files, '
        f'{len(combined.failures)} errors → {args.output_dir}/'
    )
    combined.raise_if_incomplete()


if __name__ == '__main__':
    main()
