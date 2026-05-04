"""Display and summarize Every Eval Ever dataset statistics."""

import warnings

warnings.filterwarnings('ignore')

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import duckdb
from huggingface_hub import HfFileSystem
from tqdm import tqdm

SEP = '=' * 60
SUB = '-' * 60

REPO_ID = 'evaleval/EEE_datastore'
FOLDER_PATH = 'viewer_parquets'

HUGGING_FACE_DATASTORE = f'datasets/{REPO_ID}/{FOLDER_PATH}/**/*.parquet'


def execute_query(con, sql, df=False):
    if not df:
        return con.execute(sql).fetchall()
    if df:
        return con.execute(sql).df()


def section(title: str) -> None:
    print(f'\n{SEP}')
    print(f'  {title.upper()}')
    print(SUB)


def get_parquet_columns(con, url: str) -> set[str]:
    columns = con.execute(
        'DESCRIBE SELECT * FROM read_parquet(?, filename=true)', [url]
    ).fetchall()
    return {column_name for column_name, *_ in columns}


def build_instance_select_sql(available_columns: set[str]) -> str:
    def scalar_column(name: str, sql_type: str) -> str:
        if name in available_columns:
            return f'CAST({name} AS {sql_type}) AS {name}'
        return f'CAST(NULL AS {sql_type}) AS {name}'

    def json_column(source_name: str, alias: str) -> str:
        if source_name in available_columns:
            return f'CAST(to_json({source_name}) AS JSON) AS {alias}'
        return f'CAST(NULL AS JSON) AS {alias}'

    select_columns = [
        scalar_column('schema_version', 'VARCHAR'),
        scalar_column('evaluation_id', 'VARCHAR'),
        scalar_column('model_id', 'VARCHAR'),
        scalar_column('evaluation_name', 'VARCHAR'),
        scalar_column('evaluation_result_id', 'VARCHAR'),
        scalar_column('sample_id', 'VARCHAR'),
        scalar_column('sample_hash', 'VARCHAR'),
        scalar_column('interaction_type', 'VARCHAR'),
        json_column('input', 'input'),
        json_column('output', 'output'),
        json_column('messages', 'messages'),
        json_column('answer_attribution', 'answer_attribution'),
        json_column('evaluation', 'evaluation'),
        json_column('token_usage', 'token_usage'),
        json_column('performance', 'performance'),
        scalar_column('error', 'VARCHAR'),
        json_column('metadata', 'metadata'),
        scalar_column('filename', 'VARCHAR'),
    ]

    return ',\n'.join(select_columns)


def read_data(datastore) -> Tuple[List[str], List[str]]:
    hffs = HfFileSystem()
    files = hffs.glob(datastore)

    schema_urls = [f'hf://{f}' for f in files if f.endswith('dataset.parquet')]
    instance_urls = [
        f'hf://{f}' for f in files if f.endswith('dataset_samples.parquet')
    ]

    return schema_urls, instance_urls


def analyze_data(con, schema_table, instance_table, csv_path) -> None:
    print(f'\n{SEP}')
    print(f'Tables: {schema_table}, {instance_table}')
    print(SEP)

    # schema vs instance
    n_schema_files = execute_query(
        con, f'SELECT COUNT(DISTINCT filename) FROM {schema_table};'
    )[0][0]
    n_schema_rows = execute_query(con, f'SELECT COUNT(*) FROM {schema_table};')[
        0
    ][0]

    n_instance_files = execute_query(
        con, f'SELECT COUNT(DISTINCT filename) FROM {instance_table};'
    )[0][0]
    n_instance_rows = execute_query(
        con, f'SELECT COUNT(*) FROM {instance_table};'
    )[0][0]

    section('overview')
    print(f'  {"total files":<32} {n_schema_files + n_instance_files:>10,}')
    print(f'  {"total rows":<32} {n_schema_rows + n_instance_rows:>10,}')
    print(f'  {"schema files":<32} {n_schema_files:>10,}')
    print(f'  {"instance files":<32} {n_instance_files:>10,}')
    print(f'  {"schema rows":<32} {n_schema_rows:>10,}')
    print(f'  {"instance rows":<32} {n_instance_rows:>10,}')

    # schema vs instance columns
    schema_cols = execute_query(
        con,
        f"""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = '{schema_table}';
        """,
    )
    section(f'columns  ({len(schema_cols)} total)')
    for (scc,) in schema_cols:
        print(f'  - {scc}')

    instance_cols = execute_query(
        con,
        f"""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = '{instance_table}';
        """,
    )
    section(f'columns  ({len(instance_cols)} total)')
    for (inc,) in instance_cols:
        print(f'  - {inc}')

    # schema source metadata
    src_counts = execute_query(
        con,
        f"""
        SELECT 
            source_metadata.source_type AS src, 
            COUNT(*) AS n
        FROM {schema_table},
        LATERAL UNNEST(evaluation_results) AS t(er)
        WHERE source_metadata.source_type IS NOT NULL
        GROUP BY 1 ORDER BY 2 DESC;
        """,
        df=True,
    )
    src_counts.to_csv(
        f'{csv_path}/source_types.csv',
        encoding='utf-8',
        index=False,
        header=True,
    )
    section(f'source type table saved to {csv_path}')

    # parameter range
    param_range = execute_query(
        con,
        f"""
        SELECT
            MIN(CAST(model_info.additional_details.params_billions AS FLOAT)),
            MAX(CAST(model_info.additional_details.params_billions AS FLOAT))
        FROM {schema_table}
        WHERE model_info.additional_details.params_billions IS NOT NULL;
        """,
    )
    section('model parameter range  (billions)')
    if param_range and param_range[0][0] is not None:
        min_p, max_p = param_range[0]
        print(f'  {"min":<32} {min_p:>10.2f}B')
        print(f'  {"max":<32} {max_p:>10.2f}B')
    else:
        print('  no params_billions data found')

    benchmark_popularity = execute_query(
        con,
        f"""
        SELECT
            er.source_data.dataset_name AS benchmark,
            COUNT(DISTINCT model_info.id) AS n_models,
        FROM {schema_table},
        LATERAL UNNEST(evaluation_results) AS t(er)
        WHERE er.source_data.dataset_name IS NOT NULL
        GROUP BY 1
        ORDER BY 2;
        """,
        df=True,
    )
    section('unique benchmarks in dataset')
    for bench in benchmark_popularity:
        print(f' - {bench}')

    benchmark_popularity.to_csv(
        f'{csv_path}/benchmark_popularity.csv',
        encoding='utf-8',
        index=False,
        header=True,
    )

    count_inference_platform = execute_query(
        con,
        f"""
        SELECT
            COALESCE(model_info.inference_platform, 'unreported') AS platform_inference,
            COUNT(*) AS n_result_rows,
            COUNT(DISTINCT model_info.id) AS n_models,
        FROM {schema_table},
        LATERAL UNNEST(evaluation_results) AS t(er)
        GROUP BY 1
        ORDER BY 2 DESC;
        """,
        df=True,
    )
    count_inference_platform.to_csv(
        f'{csv_path}/inference_platform.csv',
        encoding='utf-8',
        index=False,
        header=True,
    )

    section(f'unique inference platform tables saved to {csv_path}')

    unique_orgs = execute_query(
        con,
        f"""
        SELECT 
            TRIM(REGEXP_REPLACE(source_metadata.source_organization_name, '[^a-zA-Z0-9, ]', '')) AS organization,
            COUNT(*) AS n_result_rows,
            COUNT(DISTINCT evaluation_id) AS n_evaluation_runs,
            COUNT(DISTINCT model_info.id) AS n_models,
            COUNT(DISTINCT eval_library.name) AS n_harnesses
        FROM {schema_table},
        LATERAL UNNEST(evaluation_results) AS t(er)
        WHERE source_metadata.source_organization_name IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC;
        """,
        df=True,
    )

    unique_orgs.to_csv(
        f'{csv_path}/org_name.csv',
        encoding='utf-8',
        index=False,
        header=True,
    )
    section(f'unique organisations in dataset saved to {csv_path}')

    table_counts = execute_query(
        con,
        f"""
        SELECT 'Result Rows' AS statistic, COUNT(*) AS value 
        FROM {schema_table}, LATERAL UNNEST(evaluation_results) AS t(er)
        UNION ALL
        SELECT 'Schema-Level Data', COUNT(DISTINCT evaluation_id) FROM {schema_table}
        UNION ALL
        SELECT 'Unique Models', COUNT(DISTINCT model_info.id) FROM {schema_table}
        UNION ALL
        SELECT 'Unique Benchmarks', COUNT(DISTINCT er.source_data.dataset_name) 
        FROM {schema_table}, LATERAL UNNEST(evaluation_results) AS t(er)
        WHERE er.source_data.dataset_name IS NOT NULL
        UNION ALL
        SELECT 'Unique Evaluation Harnesses', COUNT(DISTINCT eval_library.name) FROM {schema_table}
        UNION ALL
        SELECT 'Unique Source Organizations', COUNT(DISTINCT TRIM(source_metadata.source_organization_name)) 
        FROM {schema_table}
        WHERE source_metadata.source_organization_name IS NOT NULL
        UNION ALL
        SELECT 'Instance-Level Data', COUNT(DISTINCT evaluation_id) FROM {instance_table};
        """,
        df=True,
    )

    table_counts.to_csv(
        f'{csv_path}/table_counts.csv',
        encoding='utf-8',
        index=False,
        header=True,
    )
    section(f'unique counts across tables saved to {csv_path}')

    benchmark_model_harness = execute_query(
        con,
        f"""
        SELECT
            eval_library.name,
            COUNT(*) AS n_result_rows,
            COUNT(DISTINCT er.source_data.dataset_name) AS n_benchmarks,
            COUNT(DISTINCT model_info.id) AS n_models,
        FROM {schema_table},
        LATERAL UNNEST(evaluation_results) AS t(er)
        GROUP BY 1
        ORDER BY 2 DESC;
        """,
        df=True,
    )

    benchmark_model_harness.to_csv(
        f'{csv_path}/benchmark_model_harness.csv',
        encoding='utf-8',
        index=False,
        header=True,
    )
    section(f'model and benchmark counts per harness saved to {csv_path}')

    print(f'\n{SEP}\n')


def create_visualisations(con, schema_table, instance_table, csv_path) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ModuleNotFoundError:
        raise ImportError('seaborn or matplotlib not installed')

    os.makedirs(create_visualisations.outdir, exist_ok=True)
    output_path = Path(create_visualisations.outdir)
    top_n = create_visualisations.top_n

    # score distribution across the most used benchmarks
    benchmark_top_scores = execute_query(
        con,
        f"""
        SELECT 
            er.source_data.dataset_name AS benchmark,
            er.score_details.score::DOUBLE AS score
        FROM {schema_table},
        LATERAL UNNEST(evaluation_results) AS t(er)
        WHERE er.source_data.dataset_name IN ('GPQA', 'IFEval')
        ORDER BY benchmark, score DESC;
        """,
        df=True,
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    benchmarks = ['GPQA', 'IFEval']
    colors = sns.color_palette('pastel', 2, as_cmap=True)
    for ax, benchmark, color in zip(axes, benchmarks, colors):
        data = benchmark_top_scores[
            benchmark_top_scores['benchmark'] == benchmark
        ]['score']
        sns.kdeplot(
            data=data, fill=True, alpha=0.5, color=color, ax=ax, linewidth=1.5
        )
        ax.axvline(
            data.median(),
            color=color,
            linestyle='--',
            linewidth=1.2,
            label=f'Median: {data.median():.3f}',
        )
        ax.axvline(
            data.mean(),
            color='gray',
            linestyle=':',
            linewidth=1.2,
            label=f'Mean: {data.mean():.3f}',
        )
        ax.set_title(
            f'{benchmark} Score Distribution (n={len(data):,})', fontsize=12
        )
        ax.set_xlabel('Score', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        sns.despine(ax=ax)

    # plt.suptitle('Score Distributions: GPQA & IFEval', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(
        output_path / 'score_distribution_gpqa_ifeval.pdf',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

    # score distribution across the most used benchmarks
    benchmark_top_scores = execute_query(
        con,
        f"""
        SELECT 
            er.source_data.dataset_name AS benchmark,
            er.score_details.score::DOUBLE AS score
        FROM {schema_table},
        LATERAL UNNEST(evaluation_results) AS t(er)
        WHERE er.source_data.dataset_name IN ('BBH', 'Artificial Analysis LLM API')
        ORDER BY benchmark, score DESC;
        """,
        df=True,
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    benchmarks = ['BBH', 'Artificial Analysis LLM API']
    colors = sns.color_palette('pastel', 2, as_cmap=True)
    for ax, benchmark, color in zip(axes, benchmarks, colors):
        data = benchmark_top_scores[
            benchmark_top_scores['benchmark'] == benchmark
        ]['score']
        sns.kdeplot(
            data=data, fill=True, alpha=0.5, color=color, ax=ax, linewidth=1.5
        )
        ax.axvline(
            data.median(),
            color=color,
            linestyle='--',
            linewidth=1.2,
            label=f'Median: {data.median():.3f}',
        )
        ax.axvline(
            data.mean(),
            color='gray',
            linestyle=':',
            linewidth=1.2,
            label=f'Mean: {data.mean():.3f}',
        )
        ax.set_title(
            f'{benchmark} Score Distribution (n={len(data):,})', fontsize=12
        )
        ax.set_xlabel('Score', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(
        output_path / 'score_distribution_bbh_art_llm_api.pdf',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

    # first party vs third party evaluator breakdown — who is doing the evaluating
    evaluation_relationships = execute_query(
        con,
        f"""
        SELECT
            source_metadata.evaluator_relationship,
            COUNT(*) AS eval_runs
        FROM {schema_table},
        LATERAL UNNEST(evaluation_results) AS t(er)
        GROUP BY 1
        """,
        df=True,
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    counts = evaluation_relationships.set_index('evaluator_relationship')[
        'eval_runs'
    ]
    ax.pie(
        counts,
        labels=counts.index,
        autopct='%1.1f%%',
        colors=sns.color_palette('viridis', len(counts)),
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
    )
    ax.set_title(
        'Evaluator Relationship Breakdown\n(First Party vs Third Party)',
        fontsize=12,
        pad=20,
    )
    plt.tight_layout()
    plt.savefig(
        output_path / 'evaluator_relationship_pie.pdf',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

    org_runs = execute_query(
        con,
        f"""
        SELECT 
            CASE 
                WHEN LENGTH(TRIM(source_metadata.source_organization_name)) > 40 
                THEN LEFT(TRIM(source_metadata.source_organization_name), 40) || '...'
                ELSE TRIM(source_metadata.source_organization_name)
            END AS org,
            COUNT(*) AS n_runs
        FROM {schema_table},
        LATERAL UNNEST(evaluation_results) AS t(er)
        WHERE source_metadata.source_organization_name IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC;
        """,
        df=True,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=org_runs, x='n_runs', y='org', palette='viridis', ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=4, fontsize=9)
    ax.set_xlabel('Number of Evaluation Runs', fontsize=11)
    ax.set_ylabel('Organization', fontsize=11)
    ax.set_title(
        f'Evaluation Runs by Source Organization (n={len(org_runs)} orgs)',
        fontsize=12,
    )
    sns.despine()
    plt.tight_layout()
    plt.savefig(
        output_path / 'source_organizations.pdf', dpi=300, bbox_inches='tight'
    )
    plt.close()

    benchmark_evaluation_counts = execute_query(
        con,
        f"""
        SELECT 
            er.source_data.dataset_name AS benchmark,
            COUNT(*) AS n_result_rows,
        FROM {schema_table},
        LATERAL UNNEST(evaluation_results) AS t(er)
        WHERE er.source_data.dataset_name IS NOT NULL
        GROUP BY benchmark
        ORDER BY n_result_rows DESC
        LIMIT {top_n};
        """,
        df=True,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=benchmark_evaluation_counts,
        x='n_result_rows',
        y='benchmark',
        palette='pastel',
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=4, fontsize=9)

    ax.set_xlabel('Number of Evaluations', fontsize=11)
    ax.set_ylabel('Benchmarks', fontsize=11)
    ax.set_title(f'Top {top_n} Benchmarks Evaluations Count', fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(
        output_path / f'top{top_n}_benchmarks_evaluation_count.pdf',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

    model_evaluation_counts = execute_query(
        con,
        f"""
        SELECT 
            model_info.id AS model,
            COUNT(*) AS n_result_rows,
            COUNT(DISTINCT evaluation_id) AS n_evaluation_runs,
            COUNT(DISTINCT eval_library.name) AS n_harnesses,
            COUNT(DISTINCT TRIM(source_metadata.source_organization_name)) AS n_orgs
        FROM {schema_table},
        LATERAL UNNEST(evaluation_results) AS t(er)
        WHERE model_info.id IS NOT NULL
        GROUP BY model
        ORDER BY n_result_rows DESC
        LIMIT {top_n};
        """,
        df=True,
    )
    model_evaluation_counts.to_csv(
        f'{csv_path}/model_per_benchmark.csv',
        encoding='utf-8',
        index=False,
        header=True,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=model_evaluation_counts,
        x='n_result_rows',
        y='model',
        palette='pastel',
        color='#0072b2',
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=4, fontsize=9)

    ax.set_xlabel('Number of Benchmarks', fontsize=11)
    ax.set_ylabel('Model', fontsize=11)
    ax.set_title(f'Top {top_n} Models Evaluation Count', fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(
        output_path / f'top{top_n}_models_evaluation_count.pdf',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

    # harness coverage grouped bar
    harness_df = execute_query(
        con,
        f"""
        SELECT
            eval_library.name AS framework,
            COUNT(DISTINCT evaluation_id) AS n_evaluation_runs,
            COUNT(model_info.id) AS n_models,
            COUNT(DISTINCT er.source_data.dataset_name) AS n_benchmarks,
            COUNT(*) AS n_runs,
        FROM {schema_table}, 
        LATERAL UNNEST(evaluation_results) AS t(er)
        GROUP BY 1 ORDER BY n_runs DESC;
    """,
        df=True,
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(harness_df))
    width = 0.25
    colors = sns.color_palette('pastel', 3)
    ax.bar(
        [i - width for i in x],
        harness_df['n_runs'],
        width,
        label='Total Runs',
        color=colors[0],
    )
    ax.bar(
        [i for i in x],
        harness_df['n_models'],
        width,
        label='Models',
        color=colors[1],
    )
    ax.bar(
        [i + width for i in x],
        harness_df['n_evaluation_runs'],
        width,
        label='Evaluations',
        color=colors[2],
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        harness_df['framework'], rotation=35, ha='right', fontsize=9
    )
    ax.set_yscale('log')
    ax.set_ylabel('Count (log scale)', fontsize=11)
    ax.set_title(
        'Harness Coverage: Total Runs, Models, Evaluations', fontsize=12
    )
    ax.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig(
        output_path / 'harness_coverage.pdf', dpi=300, bbox_inches='tight'
    )
    plt.close()

    print(f'All visualisations saved to {output_path.resolve()}')


def main():
    print(f'\n{SEP}')
    print('EVERY EVAL EVER STATS')
    print(f'DATASTORE: {HUGGING_FACE_DATASTORE}')
    print(SUB)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--table', default='eee', help='Table name for database'
    )
    parser.add_argument(
        '--viz-dir',
        default='data/outputs/viz',
        help='Output directory for generated visualization files',
    )
    parser.add_argument(
        '--csv-path',
        default='data/outputs/tables',
        help='Output directory for generated tables',
    )
    parser.add_argument(
        '--top-n',
        default=25,
        type=int,
        help='Top-N groups to include in visualization charts',
    )

    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', args.table):
        parser.error('--table must be a valid SQL identifier')

    table_name = args.table
    csv_path = args.csv_path

    os.makedirs(csv_path, exist_ok=True)
    csv_path = Path(csv_path)

    create_visualisations.outdir = args.viz_dir
    create_visualisations.top_n = args.top_n

    schema_table = f'{table_name}_schema'
    instance_table = f'{table_name}_instances'

    with duckdb.connect(':memory:') as con:
        schema_loaded = False
        instance_loaded = False

        try:
            con.execute('LOAD httpfs;')
        except duckdb.Error:
            con.execute('INSTALL httpfs;')
            con.execute('LOAD httpfs;')

        schema_urls, instance_urls = read_data(HUGGING_FACE_DATASTORE)
        if not schema_urls and not instance_urls:
            print('No parquet files found')
            sys.exit(1)

        if schema_urls:
            con.execute(
                f"""
                CREATE OR REPLACE TABLE {schema_table} AS
                SELECT * FROM read_parquet(?, union_by_name=true, filename=true)
            """,
                [schema_urls],
            )
            schema_loaded = True
            print('Schema Table Successfully Loaded')

        if instance_urls:
            con.execute(
                f"""
                CREATE OR REPLACE TABLE {instance_table} (
                    schema_version VARCHAR NOT NULL,
                    evaluation_id VARCHAR NOT NULL,
                    model_id VARCHAR NOT NULL,
                    evaluation_name VARCHAR NOT NULL,
                    evaluation_result_id VARCHAR,
                    sample_id VARCHAR NOT NULL,
                    sample_hash VARCHAR,
                    interaction_type VARCHAR NOT NULL,
                    input JSON NOT NULL,
                    output JSON,
                    messages JSON,
                    answer_attribution JSON NOT NULL,
                    evaluation JSON NOT NULL,
                    token_usage JSON,
                    performance JSON,
                    error VARCHAR,
                    metadata JSON,
                    filename VARCHAR NOT NULL
                )
                """
            )
            # we load instance parquet files one by one because duckdb
            # can fail when unioning files with mixed nested optional column types (an issue on duckdbs side)
            # (struct in one file, NULL-type in another for the same column).
            # so we check each file's columns first and fill missing ones with NULL before insert.
            # will look implement similar convention for schema level data to future proof it
            for url in tqdm(
                instance_urls,
                desc='Loading instance parquet files',
                unit='file',
            ):
                available_columns = get_parquet_columns(con, url)
                instance_select_sql = build_instance_select_sql(
                    available_columns
                )
                con.execute(
                    f"""
                    INSERT INTO {instance_table}
                    SELECT
                        {instance_select_sql}
                    FROM read_parquet(?, filename=true)
                    """,
                    [url],
                )
            instance_loaded = True
            print('Instance Table Successfully Loaded')

        if not schema_loaded or not instance_loaded:
            print('Skipping combined analysis: one table was not loaded.')
            sys.exit(1)

        analyze_data(con, schema_table, instance_table, csv_path)
        create_visualisations(con, schema_table, instance_table, csv_path)


if __name__ == '__main__':
    main()
