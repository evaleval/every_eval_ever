"""
Post-codegen patches for every_eval_ever/eval_types.py and every_eval_ever/instance_level_types.py.

Run after datamodel-codegen to re-apply model validators that codegen cannot generate.

Usage:
    uv run datamodel-codegen --input every_eval_ever/schemas/eval.schema.json --output every_eval_ever/eval_types.py ...
    uv run datamodel-codegen --input every_eval_ever/schemas/instance_level_eval.schema.json --output every_eval_ever/instance_level_types.py ...
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
        'file': 'every_eval_ever/instance_level_types.py',
        'import_add': 'model_validator',
        'class_name': 'InstanceLevelEvaluationLog',
        'marker': 'def validate_interaction_type_consistency',
        'validator': """
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
""",
    },
    {
        'file': 'every_eval_ever/eval_types.py',
        'import_add': 'model_validator',
        'class_name': 'ModelInfo',
        'marker': 'def default_model_metadata',
        'validator': """
    # --- validator (added by post_codegen.py) ---

    @model_validator(mode="after")
    def default_model_metadata(self):
        '''Emit compatibility placeholders for the new model metadata axes.'''
        details = dict(self.additional_details or {})
        details.setdefault("deployment_type", "unknown")
        details.setdefault("model_availability", "unknown")
        allowed = {
            "deployment_type": {
                "self_deployed", "externally_managed", "unknown"
            },
            "model_availability": {
                "open_weights", "closed_weights", "unknown"
            },
        }
        for name, values in allowed.items():
            if details[name] not in values:
                raise ValueError(
                    f"{name} must be one of {sorted(values)}, "
                    f"got {details[name]!r}"
                )
        self.additional_details = details
        return self
""",
    },
    {
        'file': 'every_eval_ever/eval_types.py',
        'import_add': [
            'model_validator',
            'field_serializer',
            'field_validator',
        ],
        'class_name': 'MetricConfig',
        'marker': 'def validate_score_type_requirements',
        'validator': """
    # --- validators (added by post_codegen.py) ---

    @field_validator("min_score", "max_score", mode="before")
    @classmethod
    def validate_bound_wire_type(cls, value):
        if value == "Infinity":
            return float("inf")
        if value == "-Infinity":
            return float("-inf")
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "metric bounds must be JSON numbers or the exact strings "
                "'Infinity'/'-Infinity'"
            )
        return value

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
        for name in ('min_score', 'max_score'):
            value = getattr(self, name)
            if value is not None and value != value:
                raise ValueError(f'{name} must not be NaN')
        return self

    @field_serializer('min_score', 'max_score', when_used='json')
    def _serialize_bound(self, value):
        if value == float('inf'):
            return 'Infinity'
        if value == float('-inf'):
            return '-Infinity'
        return value
""",
    },
]

GENERATED_FIELD_PATCHES = [
    {
        'file': 'every_eval_ever/eval_types.py',
        'class_name': 'ModelInfo',
        'field_name': 'additional_details',
        'replacement': """    additional_details: dict[str, str] | None = Field(
        None,
        description='Additional parameters (key-value pairs, all values must be strings)',
    )""",
    },
    {
        'file': 'every_eval_ever/eval_types.py',
        'class_name': 'MetricConfig',
        'field_name': 'min_score',
        'replacement': """    min_score: float | None = Field(
        None,
        description='Minimum possible score for a continuous metric. Use -inf if unbounded below; null means not provided.',
    )""",
    },
    {
        'file': 'every_eval_ever/eval_types.py',
        'class_name': 'MetricConfig',
        'field_name': 'max_score',
        'replacement': """    max_score: float | None = Field(
        None,
        description='Maximum possible score for a continuous metric. Use inf if unbounded above; null means not provided.',
    )""",
    },
]

REMOVED_GENERATED_CLASSES = {
    'every_eval_ever/eval_types.py': [
        'AdditionalDetails',
        'DeploymentType',
        'MaxScore',
        'MinScore',
        'ModelAvailability',
    ]
}

# ---------------------------------------------------------------------------
# Discriminator patch for source_data union in EvaluationResult
# ---------------------------------------------------------------------------

DISCRIMINATOR_PATCH = {
    'file': 'every_eval_ever/eval_types.py',
    'target_line': '    source_data: SourceDataUrl | SourceDataHf | SourceDataPrivate = Field(',
    'replacement': '    source_data: Annotated[SourceDataUrl | SourceDataHf | SourceDataPrivate, Discriminator("source_type")] = Field(',
    'imports': ['Annotated', 'Discriminator'],
}


def add_import(content: str, symbol: str) -> str:
    """Add a symbol to the pydantic import line if not already present."""
    block_match = re.search(
        r'from pydantic import \(\n(?P<body>.*?)\n\)',
        content,
        re.DOTALL,
    )
    if block_match:
        imports = [
            line.strip().removesuffix(',')
            for line in block_match.group('body').splitlines()
            if line.strip()
        ]
        if symbol in imports:
            return content
        imports.append(symbol)
        body = ''.join(
            f'    {item},\n' for item in sorted(set(imports))
        ).rstrip()
        replacement = f'from pydantic import (\n{body}\n)'
        return (
            content[: block_match.start()]
            + replacement
            + content[block_match.end() :]
        )

    line_match = re.search(r'from pydantic import (.+)', content)
    if line_match is None:
        raise ValueError('pydantic import not found')
    imports = [item.strip() for item in line_match.group(1).split(',')]
    if symbol in imports:
        return content
    imports.append(symbol)
    replacement = 'from pydantic import ' + ', '.join(sorted(set(imports)))
    return (
        content[: line_match.start()]
        + replacement
        + content[line_match.end() :]
    )


def append_to_last_class_field(
    content: str, class_name: str, validator_code: str
) -> str:
    """Append validator code after the last field of a class, before the next class or EOF."""
    # Find the class definition
    class_pattern = rf'^class {class_name}\(.*?\):'
    class_match = re.search(class_pattern, content, re.MULTILINE)
    if not class_match:
        raise ValueError(f'Class {class_name} not found')

    class_start = class_match.start()

    # Find the next class definition or EOF after this class
    next_class = re.search(
        r'^\nclass ', content[class_start + 1 :], re.MULTILINE
    )
    if next_class:
        insert_pos = class_start + 1 + next_class.start()
    else:
        insert_pos = len(content)

    # Insert validator before the next class (or at EOF), replacing trailing whitespace
    before = content[:insert_pos].rstrip('\n')
    after = content[insert_pos:]

    return before + '\n' + validator_code + after


def replace_class_field(content: str, patch: dict) -> str:
    """Replace one generated field without depending on its generated type."""
    class_match = re.search(
        rf'^class {patch["class_name"]}\(.*?\):',
        content,
        re.MULTILINE,
    )
    if class_match is None:
        raise ValueError(f'class {patch["class_name"]} not found')
    next_class = re.search(
        r'^\nclass ', content[class_match.end() :], re.MULTILINE
    )
    class_end = (
        class_match.end() + next_class.start()
        if next_class is not None
        else len(content)
    )
    class_body = content[class_match.end() : class_end]
    field_pattern = re.compile(
        rf'^    {patch["field_name"]}:.*?'
        r'(?=^    (?:[A-Za-z_]\w*\s*:|@|#)|\Z)',
        re.MULTILINE | re.DOTALL,
    )
    replaced, count = field_pattern.subn(
        patch['replacement'] + '\n\n', class_body, count=1
    )
    if count != 1:
        raise ValueError(
            f'{patch["class_name"]}.{patch["field_name"]} field not found'
        )
    return content[: class_match.end()] + replaced + content[class_end:]


def apply_generated_field_patch(patch: dict) -> None:
    path = Path(__file__).parent / patch['file']
    content = replace_class_field(path.read_text(), patch)
    path.write_text(content)
    print(
        f'  {patch["file"]}: patched '
        f'{patch["class_name"]}.{patch["field_name"]}'
    )


def remove_generated_classes(file: str, class_names: list[str]) -> None:
    """Remove helper enums/models made obsolete by generated field patches."""
    path = Path(__file__).parent / file
    content = path.read_text()
    for class_name in class_names:
        class_pattern = re.compile(
            rf'^class {class_name}\(.*?(?=^\nclass |\Z)',
            flags=re.MULTILINE | re.DOTALL,
        )
        class_match = class_pattern.search(content)
        if class_match is None:
            continue
        without_class = (
            content[: class_match.start()] + content[class_match.end() :]
        )
        if re.search(rf'\b{class_name}\b', without_class):
            continue
        content = without_class
    path.write_text(content)
    print(f'  {file}: removed obsolete generated helper classes')


def patch_file(patch: dict) -> None:
    path = Path(__file__).parent / patch['file']
    content = path.read_text()

    if patch['marker'] in content:
        print(f'  {patch["file"]}: already patched, skipping')
        return

    imports = patch['import_add']
    for symbol in [imports] if isinstance(imports, str) else imports:
        content = add_import(content, symbol)
    content = append_to_last_class_field(
        content, patch['class_name'], patch['validator']
    )

    path.write_text(content)
    print(f'  {patch["file"]}: patched {patch["class_name"]}')


def apply_discriminator_patch(patch: dict) -> None:
    """Add Discriminator annotation to a union field for better error messages."""
    path = Path(__file__).parent / patch['file']
    content = path.read_text()

    # Ruff may expand the replacement across several lines, so detect the
    # resulting annotation structurally instead of relying on exact formatting.
    discriminator_pattern = re.compile(
        r'source_data:\s*Annotated\[\s*'
        r'SourceDataUrl\s*\|\s*SourceDataHf\s*\|\s*SourceDataPrivate\s*,'
        r'\s*Discriminator\([\'"]source_type[\'"]\)',
        re.DOTALL,
    )
    if discriminator_pattern.search(content):
        print(f'  {patch["file"]}: discriminator already patched, skipping')
        return

    # Add imports
    for symbol in patch['imports']:
        if symbol == 'Annotated':
            if 'from typing import' in content:
                if 'Annotated' not in content:
                    content = content.replace(
                        'from typing import ',
                        'from typing import Annotated, ',
                    )
            else:
                # Add typing import after pydantic import
                content = content.replace(
                    'from pydantic import ',
                    'from typing import Annotated\nfrom pydantic import ',
                )
        elif symbol == 'Discriminator':
            content = add_import(content, 'Discriminator')

    # Replace the target line
    target_line = patch['target_line']
    occurrences = content.count(target_line)
    if occurrences == 0:
        raise ValueError(
            f'Target line for discriminator patch not found in {patch["file"]}'
        )
    if occurrences > 1:
        print(
            f'  {patch["file"]}: warning: multiple ({occurrences}) occurrences of '
            'target line found; patching all occurrences'
        )

    content = content.replace(target_line, patch['replacement'])
    path.write_text(content)
    print(f'  {patch["file"]}: patched source_data with Discriminator')


def main():
    print('Applying post-codegen patches...')
    for patch in GENERATED_FIELD_PATCHES:
        apply_generated_field_patch(patch)
    for file, class_names in REMOVED_GENERATED_CLASSES.items():
        remove_generated_classes(file, class_names)
    for patch in PATCHES:
        patch_file(patch)
    apply_discriminator_patch(DISCRIMINATOR_PATCH)
    print('Done.')


if __name__ == '__main__':
    main()
