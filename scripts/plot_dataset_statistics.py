"""Generate PDF plots and a narrative summary from dataset statistics JSON."""

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
    selected = list(reversed(selected))
    labels = [short_label(str(row['value']), 48) for row in selected]
    counts = [row['count'] for row in selected]

    fig, ax = plt.subplots(figsize=(10, max(5, len(selected) * 0.45)))
    if selected:
        if sns is not None:
            sns.barplot(x=counts, y=labels, hue=labels, ax=ax, legend=False)
        else:
            ax.barh(labels, counts)
    else:
        ax.text(
            0.5,
            0.5,
            'No inference-engine summary available',
            ha='center',
            va='center',
            transform=ax.transAxes,
        )
    ax.set_xlabel('Result rows')
    ax.set_ylabel('')
    ax.set_title('Recorded Inference Engine/Platform Spread')

    path = output_dir / PLOT_FILES['engine_spread']
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
    descriptive = stats['descriptive']
    counts = stats['descriptive']['counts']
    quality = stats['descriptive']['quality']
    valid = stats['observational']['valid_normalized_rows']
    exclusions = stats['observational']['exclusions']
    out_of_range = exclusions.get('out_of_range', 0)
    models_per_benchmark = descriptive.get('models_per_benchmark', [])
    inference_engines = descriptive.get('inference_engines', [])
    most_covered = top_rows(rows, 'count', 6)
    highest_variance = sorted(
        rows, key=lambda row: float(row['stddev'] or 0.0), reverse=True
    )[:4]
    hardest = sorted(rows, key=lambda row: float(row['mean']))[:4]
    easiest = sorted(rows, key=lambda row: float(row['mean']), reverse=True)[:4]
    model_counts = [
        int(row['unique_models'])
        for row in models_per_benchmark
        if int(row['unique_models']) > 0
    ]
    median_models = statistics.median(model_counts) if model_counts else 0
    max_models = max(model_counts) if model_counts else 0
    top_model_datasets = models_per_benchmark[:6]
    known_engine_rows = sum(
        int(row['count'])
        for row in inference_engines
        if str(row['value']).strip().lower() != 'unknown'
    )
    unknown_engine_rows = sum(
        int(row['count'])
        for row in inference_engines
        if str(row['value']).strip().lower() == 'unknown'
    )
    top_engines = inference_engines[:6]

    def names(items: list[dict[str, Any]]) -> str:
        return ', '.join(label(item) for item in items)

    def benchmark_model_names(items: list[dict[str, Any]]) -> str:
        return ', '.join(
            f'{item["benchmark"]} ({int(item["unique_models"]):,})'
            for item in items
        )

    def engine_names(items: list[dict[str, Any]]) -> str:
        return ', '.join(
            f'{item["value"]} ({int(item["count"]):,})' for item in items
        )

    relative_plots = {
        name: path.relative_to(output_path.parent)
        if path.is_relative_to(output_path.parent)
        else path
        for name, path in plot_paths.items()
    }
    text = f"""# Dataset Statistics Summary

This report summarizes the latest Every Eval Ever datastore snapshot represented by `dataset_statistics.json`. In the statistics file, “dataset” is represented by the `benchmark` field, which comes from `evaluation_results[].source_data.dataset_name`. That naming is worth keeping in mind when reading the figures: a benchmark is the dataset or leaderboard family that supplied the result rows, while an evaluation name is the finer slice or metric label inside that benchmark. The corpus contains {counts['result_rows']:,} result rows across {counts['unique_benchmarks']:,} datasets, {counts['unique_evaluations']:,} evaluation names, {counts['unique_developers']:,} developers, and {counts['unique_models']:,} models. The coverage plot (`{relative_plots['coverage']}`) gives the first scale check: the datastore is broad in model count, but its row-level mass is still concentrated in a smaller number of repeated evaluation families.

Normalization quality is strong for this snapshot. Of {quality['total_result_rows']:,} result rows, {valid:,} rows can be converted onto the shared zero-to-one scale, or {pct(valid, quality['total_result_rows']):.1f}% of the dataset. The only observed normalization exclusion is {out_of_range:,} out-of-range rows; missing scores, missing bounds, zero-width bounds, and incompatible score types are all zero. This means the normalized score summaries are a reasonable map of cross-benchmark score distributions. It does not make all metrics semantically identical, but it does put the numeric ranges on a common axis so that difficulty, saturation, and spread are easier to compare. The normalization quality plot (`{relative_plots['quality']}`) is therefore a guardrail figure: it says whether the rest of the normalized-score visuals are based on most of the corpus or on a narrow filtered subset.

Coverage is uneven by design. The most-covered normalized summaries are {names(most_covered)}. These heavily represented evaluations dominate aggregate descriptive patterns, so the top-coverage chart (`{relative_plots['top_coverage']}`) should be read alongside any mean-score chart. A benchmark with thousands of rows provides a much steadier estimate than a niche evaluation with dozens or hundreds of rows, even if both appear as one row in the summary table. High row coverage can mean a benchmark has broad model participation, multiple reported submetrics, repeated submissions, or some combination of the three. The plot is intentionally row-count oriented, because the descriptive JSON is primarily row-oriented; it should not be read as a direct measure of benchmark popularity without checking model coverage separately.

The new model-per-dataset histogram (`{relative_plots['models_per_dataset']}`) adds that missing model-coverage view. Across datasets, the median number of unique models is {median_models:g}, and the largest dataset-level model count is {max_models:,}. The highest-coverage datasets by unique model count are {benchmark_model_names(top_model_datasets)}. This distribution is important because a dataset with many models tells us more about the breadth of the ecosystem than a dataset with many rows from a smaller model set. A heavy right tail in this histogram means a few datasets act as common comparison hubs, while many others remain specialized or sparsely covered. That is not necessarily bad; specialized datasets are often where the datastore gets its texture. But it does mean corpus-wide summaries should avoid treating every benchmark as equally well sampled.

The inference-engine spread plot (`{relative_plots['engine_spread']}`) describes how result rows are distributed across recorded running engines or inference platforms, depending on which runtime metadata is present in the datastore export. The leading runtime labels are {engine_names(top_engines)}. In this snapshot, {known_engine_rows:,} rows have a named runtime field and {unknown_engine_rows:,} rows fall under `unknown`. The `unknown` bucket is expected whenever source records report model identity but not the serving/runtime layer. Runtime spread should therefore be read as an observability diagnostic, not just as a usage ranking. A large `unknown` bucket says that many results are still useful for model and benchmark analysis, but they cannot support claims about vLLM, Ollama, hosted APIs, or other runtime-specific execution paths. Where runtime names are present, the chart gives a quick view of which execution backends are represented strongly enough for follow-up slicing.

Mean normalized scores vary sharply across tasks. The lowest means include {names(hardest)}, while the highest means include {names(easiest)}. These values should not be interpreted as a leaderboard: they summarize all available submitted model results within each benchmark/evaluation pair, not matched model cohorts. They are best used to spot which evaluations are generally difficult, saturated, or mixed across the collected model population. A low mean can indicate a hard benchmark, a benchmark with many older or weaker systems, or a metric whose upper range is rarely reached. A high mean can indicate an easier task, a saturated benchmark, a curated set of strong submissions, or a metric where the lower-performing tail is missing. The summary plots do not decide among those explanations, but they point to where a closer paired analysis would be valuable.

The variability plots add the most diagnostic texture. High-standard-deviation evaluations such as {names(highest_variance)} indicate tasks where model results span a wide range, often because the benchmark separates weak and strong systems clearly or because the source data combines distinct regimes. The range plot (`{relative_plots['range']}`) highlights the same issue from min-to-max spread, while the mean-versus-standard-deviation scatter (`{relative_plots['variability']}`) separates broad, high-confidence coverage from sparse or volatile summaries. Evaluations with both substantial coverage and high spread are especially useful for model comparison because they appear to discriminate among systems rather than clustering everyone near the same score. Evaluations with low spread can still matter, but they may be better suited for pass/fail checks, regression testing, or detecting severe failures than for fine-grained ranking.

The PDF figures are meant to be inspected together rather than as standalone claims. The count and quality charts answer whether the data is large and clean enough to trust. The top-coverage and model-per-dataset charts separate result-row volume from unique-model breadth. The engine chart shows whether runtime metadata is available and how concentrated it is. The mean, variability, and range charts then answer where the benchmark landscape is concentrated, sparse, easy, hard, or discriminative. Keeping those questions separate avoids a common mistake: treating a high row count as evidence of broad participation, or treating a normalized mean as a direct model-quality claim.

Overall, the datastore is large, mostly normalization-ready, and informative for benchmark-level descriptive analysis. The main caveat is comparability: normalized scores put different metrics on a common scale, but they do not control for which models appear in each benchmark. Use these figures as a map of datastore coverage, runtime observability, and score distribution, then rely on paired or coverage-aware analyses for direct model comparisons. The descriptive plots are best thought of as a scouting layer: they reveal where the datastore is rich, where metadata is thin, and where more careful model-by-model analysis is likely to pay off.
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
        'models_per_dataset': plot_models_per_dataset_histogram(
            stats, output_dir, plt, sns
        ),
        'engine_spread': plot_inference_engine_spread(
            stats, output_dir, plt, sns, args.top_n
        ),
    }
    write_summary(stats, rows, plot_paths, args.summary_output)

    print(f'Wrote {len(plot_paths)} PDF plots to {output_dir}')
    print(f'Wrote Markdown summary to {args.summary_output}')


if __name__ == '__main__':
    main()
