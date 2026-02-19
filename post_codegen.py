"""
Post-codegen patches for eval_types.py and instance_level_types.py.

Run after datamodel-codegen to re-apply model validators that codegen cannot generate.

Usage:
    uv run datamodel-codegen --input eval.schema.json --output eval_types.py ...
    uv run datamodel-codegen --input instance_level_eval.schema.json --output instance_level_types.py ...
    uv run python post_codegen.py
"""

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Patch definitions
# Each patch targets a specific file + class and appends a validator method.
# ---------------------------------------------------------------------------

PATCHES = [
    {
        "file": "instance_level_types.py",
        "import_add": "model_validator",
        "class_name": "InstanceLevelEvaluationLog",
        "validator": '''
    # --- validators (added by post_codegen.py) ---

    @model_validator(mode="after")
    def validate_interaction_type_consistency(self):
        if self.interaction_type == InteractionType.single_turn:
            if self.output is None:
                raise ValueError("single_turn interaction_type requires output")
            if self.messages is not None:
                raise ValueError(
                    "single_turn interaction_type must not have messages"
                )
        else:
            if self.messages is None:
                raise ValueError(
                    f"{self.interaction_type.value} interaction_type requires messages"
                )
            if self.output is not None:
                raise ValueError(
                    f"{self.interaction_type.value} interaction_type must not have output"
                )
        return self
''',
    },
    {
        "file": "eval_types.py",
        "import_add": "model_validator",
        "class_name": "MetricConfig",
        "validator": '''
    # --- validators (added by post_codegen.py) ---

    @model_validator(mode="after")
    def validate_score_type_requirements(self):
        if self.score_type == ScoreType.levels:
            if self.level_names is None:
                raise ValueError("score_type 'levels' requires level_names")
            if self.has_unknown_level is None:
                raise ValueError("score_type 'levels' requires has_unknown_level")
        elif self.score_type == ScoreType.continuous:
            if self.min_score is None:
                raise ValueError("score_type 'continuous' requires min_score")
            if self.max_score is None:
                raise ValueError("score_type 'continuous' requires max_score")
        return self
''',
    },
]


def add_import(content: str, symbol: str) -> str:
    """Add a symbol to the pydantic import line if not already present."""
    if symbol in content:
        return content

    def replacer(m):
        existing = m.group(1)
        return f"from pydantic import {existing}, {symbol}"

    return re.sub(r"from pydantic import (.+)", replacer, content, count=1)


def append_to_last_class_field(content: str, class_name: str, validator_code: str) -> str:
    """Append validator code after the last field of a class, before the next class or EOF."""
    # Find the class definition
    class_pattern = rf"^class {class_name}\(.*?\):"
    class_match = re.search(class_pattern, content, re.MULTILINE)
    if not class_match:
        raise ValueError(f"Class {class_name} not found")

    class_start = class_match.start()

    # Find the next class definition or EOF after this class
    next_class = re.search(r"^\nclass ", content[class_start + 1:], re.MULTILINE)
    if next_class:
        insert_pos = class_start + 1 + next_class.start()
    else:
        insert_pos = len(content)

    # Insert validator before the next class (or at EOF), replacing trailing whitespace
    before = content[:insert_pos].rstrip("\n")
    after = content[insert_pos:]

    return before + "\n" + validator_code + after


def patch_file(patch: dict) -> None:
    path = Path(__file__).parent / patch["file"]
    content = path.read_text()

    # Check if already patched
    if "post_codegen.py" in content:
        print(f"  {patch['file']}: already patched, skipping")
        return

    content = add_import(content, patch["import_add"])
    content = append_to_last_class_field(content, patch["class_name"], patch["validator"])

    path.write_text(content)
    print(f"  {patch['file']}: patched {patch['class_name']}")


def main():
    print("Applying post-codegen patches...")
    for patch in PATCHES:
        patch_file(patch)
    print("Done.")


if __name__ == "__main__":
    main()
