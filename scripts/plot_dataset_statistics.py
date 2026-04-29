"""Generate PDF plots and a narrative summary from dataset statistics JSON."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any

PLOT_FILES = {
    'coverage': 'coverage_counts.pdf',
    'quality': 'normalization_quality.pdf',
    'top_coverage': 'top_evaluation_coverage.pdf',
    'mean': 'normalized_score_mean_by_eval.pdf',
    'variability': 'normalized_score_variability.pdf',
    'range': 'score_range_by_eval.pdf',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate dataset-statistics PDF plots and summary.'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('audit/dataset_statistics.json'),
        help='Path to dataset_statistics.json.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('audit/dataset_statistics_plots'),
        help='Directory for generated PDF plots.',
    )
    parser.add_argument(
        '--summary-output',
        type=Path,
        default=Path('audit/dataset_statistics_summary.md'),
        help='Path for generated Markdown summary.',
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=25,
        help='Number of evaluation rows to show in ranked plots.',
    )
    return parser.parse_args()


def load_statistics(path: Path) -> dict[str, Any]:
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def import_plotting() -> tuple[Any, Any | None]:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            'matplotlib is required to generate plots. Install matplotlib '
            'or seaborn in the active environment and rerun this script.'
        ) from exc

    try:
        import seaborn as sns
    except ModuleNotFoundError:
        sns = None

    if sns is not None:
        sns.set_theme(style='whitegrid', context='talk')
    else:
        plt.style.use('ggplot')
    return plt, sns


def label(row: dict[str, Any]) -> str:
    benchmark = str(row['benchmark'])
    evaluation = str(row['evaluation_name'])
    if benchmark == evaluation:
        return benchmark
    return f'{benchmark}: {evaluation}'


def short_label(value: str, width: int = 46) -> str:
    return textwrap.shorten(value, width=width, placeholder='...')


def columns(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> dict[str, list[Any]]:
    return {key: [row[key] for row in rows] for key in keys}


def save(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, format='pdf', bbox_inches='tight')


def plot_coverage_counts(
    stats: dict[str, Any], output_dir: Path, plt: Any, sns: Any | None
) -> Path:
    counts = stats['descriptive']['counts']
    order = [
        'result_rows',
        'unique_models',
        'unique_developers',
        'unique_evaluations',
        'unique_benchmarks',
    ]
    rows = [
        {'metric': key.replace('_', ' ').title(), 'count': counts[key]}
        for key in order
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    if sns is not None:
        sns.barplot(
            data=columns(rows, ('metric', 'count')),
            x='metric',
            y='count',
            hue='metric',
            ax=ax,
            legend=False,
        )
    else:
        ax.bar([row['metric'] for row in rows], [row['count'] for row in rows])
    ax.set_yscale('log')
    ax.set_xlabel('')
    ax.set_ylabel('Count, log scale')
    ax.set_title('Dataset Coverage')
    ax.tick_params(axis='x', rotation=25)
    for index, row in enumerate(rows):
        ax.text(
            index, row['count'], f'{row["count"]:,}', ha='center', va='bottom'
        )

    path = output_dir / PLOT_FILES['coverage']
    save(fig, path)
    plt.close(fig)
    return path


def plot_normalization_quality(
    stats: dict[str, Any], output_dir: Path, plt: Any, sns: Any | None
) -> Path:
    valid = stats['observational']['valid_normalized_rows']
    exclusions = stats['observational']['exclusions']
    rows = [{'category': 'valid normalized rows', 'count': valid}] + [
        {'category': key.replace('_', ' '), 'count': value}
        for key, value in exclusions.items()
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    if sns is not None:
        sns.barplot(
            data=columns(rows, ('category', 'count')),
            x='category',
            y='count',
            hue='category',
            ax=ax,
            legend=False,
        )
    else:
        ax.bar(
            [row['category'] for row in rows], [row['count'] for row in rows]
        )
    ax.set_yscale('symlog', linthresh=1)
    ax.set_xlabel('')
    ax.set_ylabel('Rows, symmetric log scale')
    ax.set_title('Normalization Quality')
    ax.tick_params(axis='x', rotation=25)
    for index, row in enumerate(rows):
        ax.text(
            index,
            max(row['count'], 1),
            f'{row["count"]:,}',
            ha='center',
            va='bottom',
        )

    path = output_dir / PLOT_FILES['quality']
    save(fig, path)
    plt.close(fig)
    return path


def top_rows(
    rows: list[dict[str, Any]], key: str, limit: int
) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (-float(row[key]), label(row)))[:limit]


def plot_top_evaluation_coverage(
    rows: list[dict[str, Any]],
    output_dir: Path,
    plt: Any,
    sns: Any | None,
    top_n: int,
) -> Path:
    selected = list(reversed(top_rows(rows, 'count', top_n)))
    labels = [short_label(label(row), 58) for row in selected]
    counts = [row['count'] for row in selected]

    fig, ax = plt.subplots(figsize=(11, max(6, top_n * 0.35)))
    if sns is not None:
        sns.barplot(x=counts, y=labels, hue=labels, ax=ax, legend=False)
    else:
        ax.barh(labels, counts)
    ax.set_xlabel('Normalized result rows')
    ax.set_ylabel('')
    ax.set_title(f'Top {len(selected)} Evaluations By Coverage')

    path = output_dir / PLOT_FILES['top_coverage']
    save(fig, path)
    plt.close(fig)
    return path


def plot_normalized_score_means(
    rows: list[dict[str, Any]],
    output_dir: Path,
    plt: Any,
    sns: Any | None,
    top_n: int,
) -> Path:
    selected = sorted(rows, key=lambda row: (float(row['mean']), label(row)))[
        :top_n
    ]
    labels = [short_label(label(row), 58) for row in selected]
    means = [row['mean'] for row in selected]

    fig, ax = plt.subplots(figsize=(11, max(6, top_n * 0.35)))
    if sns is not None:
        sns.barplot(x=means, y=labels, hue=labels, ax=ax, legend=False)
    else:
        ax.barh(labels, means)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Mean normalized score')
    ax.set_ylabel('')
    ax.set_title(f'Lowest {len(selected)} Mean Normalized Scores')

    path = output_dir / PLOT_FILES['mean']
    save(fig, path)
    plt.close(fig)
    return path


def plot_score_variability(
    rows: list[dict[str, Any]], output_dir: Path, plt: Any, sns: Any | None
) -> Path:
    plot_rows = [
        {
            'mean': row['mean'],
            'stddev': row['stddev'] or 0.0,
            'count': row['count'],
            'label': label(row),
        }
        for row in rows
    ]
    max_count = max(row['count'] for row in plot_rows)
    sizes = [
        45 + 455 * math.sqrt(row['count'] / max_count) for row in plot_rows
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    if sns is not None:
        sns.scatterplot(
            data=columns(plot_rows, ('mean', 'stddev', 'count')),
            x='mean',
            y='stddev',
            size='count',
            sizes=(45, 500),
            alpha=0.75,
            legend=False,
            ax=ax,
        )
    else:
        ax.scatter(
            [row['mean'] for row in plot_rows],
            [row['stddev'] for row in plot_rows],
            s=sizes,
            alpha=0.75,
        )
    ax.set_xlim(0, 1)
    ax.set_xlabel('Mean normalized score')
    ax.set_ylabel('Standard deviation')
    ax.set_title('Normalized Score Level vs. Variability')

    notable = sorted(
        plot_rows,
        key=lambda row: (row['stddev'], abs(row['mean'] - 0.5)),
        reverse=True,
    )[:8]
    for row in notable:
        ax.annotate(
            short_label(row['label'], 24),
            (row['mean'], row['stddev']),
            xytext=(5, 4),
            textcoords='offset points',
            fontsize=8,
        )

    path = output_dir / PLOT_FILES['variability']
    save(fig, path)
    plt.close(fig)
    return path


def plot_score_ranges(
    rows: list[dict[str, Any]],
    output_dir: Path,
    plt: Any,
    top_n: int,
) -> Path:
    selected = sorted(
        rows,
        key=lambda row: (float(row['max']) - float(row['min']), label(row)),
    )[-top_n:]
    labels = [short_label(label(row), 58) for row in selected]
    mins = [row['min'] for row in selected]
    maxes = [row['max'] for row in selected]
    means = [row['mean'] for row in selected]
    ypos = list(range(len(selected)))

    fig, ax = plt.subplots(figsize=(11, max(6, top_n * 0.35)))
    for y, low, high in zip(ypos, mins, maxes, strict=True):
        ax.hlines(y, low, high, color='#5b6770', linewidth=2.0, alpha=0.9)
    ax.scatter(means, ypos, color='#0072b2', s=32, zorder=3, label='mean')
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Normalized score range')
    ax.set_ylabel('')
    ax.set_title(f'Widest {len(selected)} Normalized Score Ranges')
    ax.legend(loc='lower right')

    path = output_dir / PLOT_FILES['range']
    save(fig, path)
    plt.close(fig)
    return path


def pct(part: int, total: int) -> float:
    return 100.0 * part / total if total else 0.0


def write_summary(
    stats: dict[str, Any],
    rows: list[dict[str, Any]],
    plot_paths: dict[str, Path],
    output_path: Path,
) -> None:
    counts = stats['descriptive']['counts']
    quality = stats['descriptive']['quality']
    valid = stats['observational']['valid_normalized_rows']
    exclusions = stats['observational']['exclusions']
    out_of_range = exclusions.get('out_of_range', 0)
    most_covered = top_rows(rows, 'count', 6)
    highest_variance = sorted(
        rows, key=lambda row: float(row['stddev'] or 0.0), reverse=True
    )[:4]
    hardest = sorted(rows, key=lambda row: float(row['mean']))[:4]
    easiest = sorted(rows, key=lambda row: float(row['mean']), reverse=True)[:4]

    def names(items: list[dict[str, Any]]) -> str:
        return ', '.join(label(item) for item in items)

    relative_plots = {
        name: path.relative_to(output_path.parent)
        if path.is_relative_to(output_path.parent)
        else path
        for name, path in plot_paths.items()
    }
    text = f"""# Dataset Statistics Summary

This report summarizes the latest Every Eval Ever datastore snapshot represented by `dataset_statistics.json`. The corpus contains {counts['result_rows']:,} result rows across {counts['unique_benchmarks']:,} benchmarks, {counts['unique_evaluations']:,} evaluation names, {counts['unique_developers']:,} developers, and {counts['unique_models']:,} models. The coverage plot (`{relative_plots['coverage']}`) shows that the datastore is broad in model count but still highly concentrated in a small number of repeated evaluation families.

Normalization quality is strong for this snapshot. Of {quality['total_result_rows']:,} result rows, {valid:,} rows can be converted onto the shared zero-to-one scale, or {pct(valid, quality['total_result_rows']):.1f}% of the dataset. The only observed normalization exclusion is {out_of_range:,} out-of-range rows; missing scores, missing bounds, zero-width bounds, and incompatible score types are all zero. This makes the normalized summaries a useful cross-benchmark view, while still leaving the raw score summaries available for scale-specific inspection.

Coverage is uneven by design. The most-covered normalized summaries are {names(most_covered)}. These heavily represented evaluations dominate aggregate descriptive patterns, so the top-coverage chart (`{relative_plots['top_coverage']}`) should be read alongside any mean-score chart. A benchmark with thousands of rows provides a much steadier estimate than a niche evaluation with dozens or hundreds of rows, even if both appear as one row in the summary table.

Mean normalized scores vary sharply across tasks. The lowest means include {names(hardest)}, while the highest means include {names(easiest)}. These values should not be interpreted as a leaderboard: they summarize all available submitted model results within each benchmark/evaluation pair, not matched model cohorts. They are best used to spot which evaluations are generally difficult, saturated, or mixed across the collected model population.

The variability plots add the most diagnostic texture. High-standard-deviation evaluations such as {names(highest_variance)} indicate tasks where model results span a wide range, often because the benchmark separates weak and strong systems clearly or because the source data combines distinct regimes. The range plot (`{relative_plots['range']}`) highlights the same issue from min-to-max spread, while the mean-versus-standard-deviation scatter (`{relative_plots['variability']}`) separates broad, high-confidence coverage from sparse or volatile summaries.

The PDF figures are meant to be inspected together rather than as standalone claims. The count and quality charts answer whether the data is large and clean enough to trust; the mean, variability, and range charts answer where the benchmark landscape is concentrated, sparse, easy, hard, or discriminative. That division keeps coverage questions separate from score interpretation.

Overall, the datastore is large, mostly normalization-ready, and informative for benchmark-level descriptive analysis. The main caveat is comparability: normalized scores put different metrics on a common scale, but they do not control for which models appear in each benchmark. Use these figures as a map of datastore coverage and score distribution, then rely on paired or coverage-aware analyses for direct model comparisons.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding='utf-8')


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise SystemExit('--top-n must be at least 1')

    stats = load_statistics(args.input)
    rows = stats['descriptive']['normalized_score_summaries']
    if not rows:
        raise SystemExit(
            'No normalized_score_summaries found in statistics JSON.'
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    plt, sns = import_plotting()

    plot_paths = {
        'coverage': plot_coverage_counts(stats, output_dir, plt, sns),
        'quality': plot_normalization_quality(stats, output_dir, plt, sns),
        'top_coverage': plot_top_evaluation_coverage(
            rows, output_dir, plt, sns, args.top_n
        ),
        'mean': plot_normalized_score_means(
            rows, output_dir, plt, sns, args.top_n
        ),
        'variability': plot_score_variability(rows, output_dir, plt, sns),
        'range': plot_score_ranges(rows, output_dir, plt, args.top_n),
    }
    write_summary(stats, rows, plot_paths, args.summary_output)

    print(f'Wrote {len(plot_paths)} PDF plots to {output_dir}')
    print(f'Wrote Markdown summary to {args.summary_output}')


if __name__ == '__main__':
    main()
