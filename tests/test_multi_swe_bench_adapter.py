import json

import yaml

from utils.multi_swe_bench.adapter import convert_submissions


def _submission(root, name, total_instances):
    submission = root / name
    (submission / "results").mkdir(parents=True)
    (submission / "metadata.yaml").write_text(
        yaml.safe_dump({"name": name}),
        encoding="utf-8",
    )
    (submission / "results" / "results.json").write_text(
        json.dumps(
            {
                "total_instances": total_instances,
                "resolved": ["one"],
            }
        ),
        encoding="utf-8",
    )
    return submission


def test_bad_submission_does_not_discard_valid_submission(tmp_path):
    good = _submission(tmp_path, "20260101_gpt-5_agent", 2)
    bad = _submission(tmp_path, "20260101_gpt-5_broken", 0)

    result = convert_submissions(
        [(good, "python"), (bad, "python")],
        retrieved_timestamp="1234",
        output_dir=str(tmp_path / "data" / "multi-swe-bench"),
    )

    assert len(result.records) == 1
    assert len(result.failures) == 1
    assert result.failures[0].source_record["submission_dir"] == str(bad)
