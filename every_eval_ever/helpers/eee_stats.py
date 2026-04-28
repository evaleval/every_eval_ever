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


def analyze_data(con, schema_table, instance_table) -> None:
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

    # schema eval library names
    lib_count = execute_query(
        con,
        f"""
        SELECT eval_library.name AS lib, COUNT(*) AS n
        FROM {schema_table}
        WHERE eval_library IS NOT NULL
        GROUP BY 1 ORDER BY 2 DESC;
    """,
    )
    section('eval library usage  (schema level)')
    for lib, n in lib_count:
        print(f'  {str(lib):<42} {n:>8,}')

    # schema source metadata
    src_counts = execute_query(
        con,
        f"""
        SELECT source_metadata.source_type AS src, COUNT(*) AS n
        FROM {schema_table}
        WHERE source_metadata.source_type IS NOT NULL
        GROUP BY 1 ORDER BY 2 DESC;
    """,
    )
    section('source type breakdown  (schema level)')
    for src, n in src_counts:
        print(f'  {str(src):<42} {n:>8,}')

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

    unique_benchmark = execute_query(
        con,
        f"""
        SELECT 
            COUNT(DISTINCT er.source_data.dataset_name) AS n_unique_benchmarks
        FROM {schema_table},
        LATERAL UNNEST(evaluation_results) AS t(er)
        WHERE er.source_data.dataset_name IS NOT NULL;
        """,
    )
    section('unique benchmarks in dataset')
    for bench in unique_benchmark:
        print(f' - {bench}')

    unique_inference = execute_query(
        con,
        f"""
        SELECT DISTINCT
            model_info.inference_platform
        FROM {schema_table}
        WHERE model_info.inference_platform NOT LIKE '%unknown%'
        """,
    )
    section('unique inference platforms in dataset')
    platforms = [platform for (platform,) in unique_inference]
    print('  ' + ', '.join(str(p) for p in platforms))

    unique_orgs = execute_query(
        con,
        f"""
        SELECT DISTINCT 
            source_metadata.source_organization_name
        FROM {schema_table}
        """,
    )
    section('unique organisations in dataset')
    orgs = [org for (org,) in unique_orgs]
    print('  ' + ', '.join(str(o) for o in orgs))

    print(f'\n{SEP}\n')


def create_visualisations(con, schema_table, instance_table) -> None:
    # for a benchmark, how do scores for that benchmark vary across
    # eval_harnesses for the same evaluation / in general (get mean, range, get min, max)
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
    colors = sns.color_palette('viridis', 2)
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
        ax.legend(fontsize=9)
        sns.despine(ax=ax)

    plt.suptitle('Score Distributions: GPQA & IFEval', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(
        output_path / 'score_distribution_gpqa_ifeval.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

    # inference platform count
    inference_count = execute_query(
        con,
        f"""
        SELECT
            model_info.inference_platform,
            COUNT(*) as n_count
        FROM {schema_table}
        WHERE model_info.inference_platform NOT LIKE '%unknown%'
        GROUP BY 1
        ORDER BY 2 DESC;
        """,
        df=True,
    )

    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=inference_count,
        x='n_count',
        y='inference_platform',
        palette='viridis',
    )
    ax.set_title('Distribution of Inference Platforms', fontsize=15, pad=20)
    ax.set_xlabel('Number of Evaluations', fontsize=12)
    ax.set_ylabel('Platform', fontsize=12)
    for i in ax.containers:
        ax.bar_label(i, padding=3)
    plt.suptitle('Inference Platform value count', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(
        output_path / 'inference_value_count.png', dpi=300, bbox_inches='tight'
    )
    plt.close()

    # first party vs third party evaluator breakdown — who is doing the evaluating
    evaluation_relationships = execute_query(
        con,
        f"""
        SELECT
            source_metadata.evaluator_relationship,
            COUNT(evaluation_id)
        FROM {schema_table}
        GROUP BY 1
        """,
        df=True,
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    counts = evaluation_relationships.set_index('evaluator_relationship')[
        'count(evaluation_id)'
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
        output_path / 'evaluator_relationship_pie.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

    unique_org_runs = execute_query(
        con,
        f"""
        SELECT 
            CASE 
                WHEN LENGTH(TRIM(source_metadata.source_organization_name)) > 40 
                THEN LEFT(TRIM(source_metadata.source_organization_name), 40) || '...'
                ELSE TRIM(source_metadata.source_organization_name)
            END AS org,
            COUNT(*) AS n_runs
        FROM {schema_table}
        WHERE source_metadata.source_organization_name IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC;
        """,
        df=True,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        data=unique_org_runs, x='n_runs', y='org', palette='viridis', ax=ax
    )
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=4, fontsize=9)
    ax.set_xlabel('Number of Evaluation Runs', fontsize=11)
    ax.set_ylabel('Organization', fontsize=11)
    ax.set_title(
        f'Evaluation Runs by Source Organization (n={len(unique_org_runs)} orgs)',
        fontsize=12,
    )
    sns.despine()
    plt.tight_layout()
    plt.savefig(
        output_path / 'source_organizations.png', dpi=300, bbox_inches='tight'
    )
    plt.close()

    benchmark_model_counts = execute_query(
        con,
        f"""
        SELECT 
            er.source_data.dataset_name AS benchmark,
            COUNT(DISTINCT model_info.id) AS n_models
        FROM {schema_table},
        LATERAL UNNEST(evaluation_results) AS t(er)
        WHERE er.source_data.dataset_name IS NOT NULL
        GROUP BY benchmark
        ORDER BY n_models DESC
        LIMIT {top_n};
        """,
        df=True,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=benchmark_model_counts,
        x='n_models',
        y='benchmark',
        palette='viridis',
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=4, fontsize=9)

    ax.set_xlabel('Number of Models', fontsize=11)
    ax.set_ylabel('Benchmark', fontsize=11)
    ax.set_title(
        'Top 10 Benchmarks by Number of Models', fontsize=12
    )
    sns.despine()
    plt.tight_layout()
    plt.savefig(
        output_path / 'top10_benchmarks_by_models.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

    model_benchmark_counts = execute_query(
        con,
        f"""
        SELECT 
            model_info.id AS model,
            COUNT(DISTINCT er.source_data.dataset_name) AS n_benchmarks
        FROM {schema_table},
        LATERAL UNNEST(evaluation_results) AS t(er)
        WHERE er.source_data.dataset_name IS NOT NULL
        GROUP BY model
        ORDER BY n_benchmarks DESC
        LIMIT {top_n};
        """,
        df=True,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=model_benchmark_counts,
        x='n_benchmarks',
        y='model',
        palette='viridis',
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=4, fontsize=9)

    ax.set_xlabel('Number of Benchmarks', fontsize=11)
    ax.set_ylabel('Model', fontsize=11)
    ax.set_title(
        'Top 10 Models by Number of Benchmarks Evaluated On', fontsize=12
    )
    sns.despine()
    plt.tight_layout()
    plt.savefig(
        output_path / 'top10_models_by_benchmarks.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()

    print(f"All visualisations saved to {output_path.resolve()}")


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
        '--top-n',
        default=10,
        type=int,
        help='Top-N groups to include in visualization charts',
    )

    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', args.table):
        parser.error('--table must be a valid SQL identifier')

    table_name = args.table

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

        analyze_data(con, schema_table, instance_table)
        create_visualisations(con, schema_table, instance_table)


if __name__ == '__main__':
    main()
