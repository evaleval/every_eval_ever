"""Shared helpers for SWE-bench family adapters."""

import re


def parse_date_from_dir(dir_name: str) -> str | None:
    """Extract ISO date from directory name prefix like '20250225_sweagent_...'"""
    m = re.match(r'^(\d{4})(\d{2})(\d{2})_', dir_name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def parse_model_from_dir(dir_name: str) -> tuple[str, str]:
    """Return (agent, primary_model) parsed from submission dir name.

    Dir format: YYYYMMDD_<agent>[_<model_parts...>]
    """
    parts = dir_name.split("_")
    if len(parts) >= 3:
        agent = parts[1]
        model = "_".join(parts[2:])
        # If the model part is purely alphabetic (no digits or hyphens), it's
        # part of the agent name rather than an LLM version (e.g. "iSWE_Agent").
        if re.match(r"^[a-zA-Z]+$", model):
            agent = "_".join(parts[1:])
            model = agent
    elif len(parts) == 2:
        agent = parts[1]
        model = agent
    else:
        agent = dir_name
        model = dir_name
    return agent, model
