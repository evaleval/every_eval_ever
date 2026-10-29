from argparse import Namespace

import pytest

from every_eval_ever import cli


def test_compression_parser_defaults():
    args = cli.build_parser().parse_args(
        ['convert', 'lm_eval', '--log-path', 'input.json']
    )
    assert cli._resolved_compression(args, 'aggregate') == 'none'
    assert cli._resolved_compression(args, 'samples') == 'none'


def test_compression_parser_overrides():
    args = cli.build_parser().parse_args(
        [
            'convert',
            'inspect',
            '--log-path',
            'input.eval',
            '--compress',
            'gz',
            '--compress-aggregate',
            'none',
            '--compress-samples',
            'xz',
        ]
    )
    assert cli._resolved_compression(args, 'aggregate') == 'none'
    assert cli._resolved_compression(args, 'samples') == 'xz'


def test_compression_resolution_accepts_legacy_namespace():
    args = Namespace()
    assert cli._resolved_compression(args, 'aggregate') == 'none'
    assert cli._resolved_compression(args, 'samples') == 'none'
    assert cli._publication_compression_kwargs(args) == {}


def test_publication_kwargs_omit_default_compression():
    args = Namespace(
        compress='gz',
        compress_aggregate='none',
        compress_samples='xz',
    )
    assert cli._publication_compression_kwargs(args) == {
        'samples_compression': 'xz',
    }


def test_compression_resolution_validates_legacy_namespace():
    args = Namespace(compress='zip')
    with pytest.raises(ValueError, match='unsupported compression'):
        cli._resolved_compression(args, 'aggregate')


def test_transcode_parser():
    args = cli.build_parser().parse_args(
        ['transcode', 'data', '--to', 'gz', '--dry-run']
    )
    assert args.command == 'transcode'
    assert args.paths == ['data']
    assert args.transcode_compression == 'gz'
    assert args.dry_run is True
