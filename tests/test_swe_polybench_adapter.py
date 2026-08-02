import json
from collections import Counter

import yaml

from every_eval_ever.adapters.swe_polybench.adapter import (
    process_submission_result,
)


def test_unknown_instance_is_reported_without_discarding_known_language(
    tmp_path,
):
    submission = tmp_path / "20260101_agent_gpt-5"
    logs = submission / "logs"
    logs.mkdir(parents=True)
    (submission / "metadata.yaml").write_text(
        yaml.safe_dump({"name": "submission"}),
        encoding="utf-8",
    )
    (logs / "known_result.json").write_text(
        json.dumps(
            {
                "instance_id": "known",
                "resolved": True,
            }
        ),
        encoding="utf-8",
    )
    (logs / "unknown_result.json").write_text(
        json.dumps(
            {
                "instance_id": "unknown",
                "resolved": True,
            }
        ),
        encoding="utf-8",
    )

    result = process_submission_result(
        submission,
        "PB",
        {"known": "python"},
        Counter({"python": 2}),
        "1234",
        yaml,
    )

    assert len(result.records) == 1
    assert result.records[0][1] == "python"
    assert len(result.failures) == 1
    assert result.failures[0].source_record["instance_id"] == "unknown"
