"""Preserved Markdown summary writer removed from plot_dataset_statistics.py."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any


def label(row: dict[str, Any]) -> str:
    benchmark = str(row['benchmark'])
    evaluation = str(row['evaluation_name'])
    if benchmark == evaluation:
        return benchmark
    return f'{benchmark}: {evaluation}'


def top_rows(
    rows: list[dict[str, Any]], key: str, limit: int
) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (-float(row[key]), label(row)))[:limit]


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
