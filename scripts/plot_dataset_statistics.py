"""Generate PDF plots from dataset statistics JSON."""

from __future__ import annotations

import argparse
import json
import math
import statistics
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
    'models_per_dataset': 'models_per_dataset_histogram.pdf',
    'engine_spread': 'inference_engine_spread.pdf',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate dataset-statistics PDF plots.'
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
    selected = top_rows(rows, 'count', top_n)
    labels = [short_label(label(row), 58) for row in selected]
    counts = [row['count'] for row in selected]

    fig, ax = plt.subplots(figsize=(11, max(6, top_n * 0.35)))
    if sns is not None:
        sns.barplot(x=counts, y=labels, hue=labels, ax=ax, legend=False)
    else:
        ax.barh(labels, counts)
    if ax.get_ylim()[0] < ax.get_ylim()[1]:
        ax.invert_yaxis()
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


def plot_models_per_dataset_histogram(
    stats: dict[str, Any], output_dir: Path, plt: Any, sns: Any | None
) -> Path:
    rows = stats['descriptive'].get('models_per_benchmark', [])
    values = [row['unique_models'] for row in rows if row['unique_models'] > 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    if values:
        bins = min(30, max(8, len(set(values))))
        if sns is not None:
            sns.histplot(values, bins=bins, ax=ax, color='#0072b2')
        else:
            ax.hist(values, bins=bins, color='#0072b2', edgecolor='white')
        median_value = statistics.median(values)
        ax.axvline(
            median_value,
            color='#d55e00',
            linestyle='--',
            linewidth=2,
            label=f'median={median_value:g}',
        )
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            'No model-per-dataset summary available',
            ha='center',
            va='center',
            transform=ax.transAxes,
        )
    ax.set_xlabel('Unique models per dataset')
    ax.set_ylabel('Datasets')
    ax.set_title('Distribution Of Unique Models Per Dataset')

    path = output_dir / PLOT_FILES['models_per_dataset']
    save(fig, path)
    plt.close(fig)
    return path


def plot_inference_engine_spread(
    stats: dict[str, Any],
    output_dir: Path,
    plt: Any,
    sns: Any | None,
    top_n: int,
) -> Path:
    rows = stats['descriptive'].get('inference_engines', [])
    selected = rows[:top_n]
    remaining = rows[top_n:]
    if remaining:
        selected = selected + [
            {
                'value': 'other',
                'count': sum(int(row['count']) for row in remaining),
            }
        ]
    labels = [short_label(str(row['value']), 48) for row in selected]
    counts = [row['count'] for row in selected]

    fig, ax = plt.subplots(figsize=(10, max(5, len(selected) * 0.45)))
    if selected:
        if sns is not None:
            sns.barplot(x=counts, y=labels, hue=labels, ax=ax, legend=False)
        else:
            ax.barh(labels, counts)
        ax.set_xscale('log')
        ax.set_xlim(left=1)
        if ax.get_ylim()[0] < ax.get_ylim()[1]:
            ax.invert_yaxis()
    else:
        ax.text(
            0.5,
            0.5,
            'No inference-engine summary available',
            ha='center',
            va='center',
            transform=ax.transAxes,
        )
    ax.set_xlabel('Result rows (log scale)')
    ax.set_ylabel('')
    ax.set_title('Recorded Inference Engine/Platform Spread')

    path = output_dir / PLOT_FILES['engine_spread']
    save(fig, path)
    plt.close(fig)
    return path


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
        'models_per_dataset': plot_models_per_dataset_histogram(
            stats, output_dir, plt, sns
        ),
        'engine_spread': plot_inference_engine_spread(
            stats, output_dir, plt, sns, args.top_n
        ),
    }
    print(f'Wrote {len(plot_paths)} PDF plots to {output_dir}')


if __name__ == '__main__':
    main()
