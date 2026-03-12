"""
Script to convert Terminal-Bench 2.0 leaderboard data to the EvalEval schema format.

Data source:
- Terminal-Bench 2.0 leaderboard: https://www.tbench.ai/leaderboard/terminal-bench/2.0

Terminal-Bench is an agentic coding benchmark that evaluates agent+model pairs on
87 terminal-based tasks with 5 trials each. Each leaderboard entry represents a
unique agent+model combination. Agent metadata is stored in model_info.additional_details.

Usage:
    uv run python -m utils.terminal_bench_2.adapter
"""

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import uuid

from eval_types import (
    AgenticEvalConfig,
    AvailableTool,
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationArgs,
    GenerationConfig,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    SourceMetadata,
    StandardError,
    Uncertainty,
)
from helpers import save_evaluation_log

LEADERBOARD_URL = "https://www.tbench.ai/leaderboard/terminal-bench/2.0"
OUTPUT_DIR = "data/terminal-bench-2.0"

ORG_SLUG_MAP = {
    "Google": "google",
    "OpenAI": "openai",
    "Anthropic": "anthropic",
    "xAI": "xai",
    "Moonshot AI": "moonshot-ai",
    "Z-AI": "zhipu-ai",
    "Z.ai": "zhipu-ai",
    "DeepSeek": "deepseek",
    "Alibaba": "alibaba",
    "MiniMax": "minimax",
    "Minimax": "minimax",
    "Kimi": "moonshot-ai",
    "Multiple": "multiple",
    "Block": "block",
    "Factory": "factory",
    "Forge Code": "forge-code",
    "KRAFTON AI": "krafton-ai",
    "Coder": "coder",
    "OpenBlock Labs": "openblock-labs",
    "Bigai": "bigai",
    "JetBrains": "jetbrains",
    "Feeling AI": "feeling-ai",
    "Antigma Labs": "antigma-labs",
    "Roam": "roam",
    "LangChain": "langchain",
    "OpenSage": "opensage",
    "Terminal Bench": "terminal-bench",
    "Intelligent Internet": "intelligent-internet",
    "Warp": "warp",
    "Letta": "letta",
    "Abacus.AI": "abacus-ai",
    "OpenHands": "openhands",
    "Anomaly Innovations": "anomaly-innovations",
    "CAMEL-AI": "camel-ai",
    "ADYA": "adya",
    "Princeton": "princeton",
    "TUM": "tum",
    "iflow": "iflow",
}

# fmt: off
LEADERBOARD_DATA = [
    {"rank": 1, "agent": "Forge Code", "model": "Gemini 3.1 Pro", "date": "2026-03-02", "agent_org": "Forge Code", "model_org": "Google", "accuracy": 78.4, "stderr": 1.8},
    {"rank": 2, "agent": "Droid", "model": "GPT-5.3-Codex", "date": "2026-02-24", "agent_org": "Factory", "model_org": "OpenAI", "accuracy": 77.3, "stderr": 2.2},
    {"rank": 3, "agent": "Simple Codex", "model": "GPT-5.3-Codex", "date": "2026-02-06", "agent_org": "OpenAI", "model_org": "OpenAI", "accuracy": 75.1, "stderr": 2.4},
    {"rank": 4, "agent": "Terminus-KIRA", "model": "Gemini 3.1 Pro", "date": "2026-02-23", "agent_org": "KRAFTON AI", "model_org": "Google", "accuracy": 74.8, "stderr": 2.6},
    {"rank": 5, "agent": "Terminus-KIRA", "model": "Claude Opus 4.6", "date": "2026-02-22", "agent_org": "KRAFTON AI", "model_org": "Anthropic", "accuracy": 74.7, "stderr": 2.6},
    {"rank": 6, "agent": "Mux", "model": "GPT-5.3-Codex", "date": "2026-03-06", "agent_org": "Coder", "model_org": "OpenAI", "accuracy": 74.6, "stderr": 2.5},
    {"rank": 7, "agent": "OB-1", "model": "Multiple", "date": "2026-03-05", "agent_org": "OpenBlock Labs", "model_org": "Multiple", "accuracy": 72.4, "stderr": 2.3},
    {"rank": 8, "agent": "TongAgents", "model": "Claude Opus 4.6", "date": "2026-02-22", "agent_org": "Bigai", "model_org": "Anthropic", "accuracy": 71.9, "stderr": 2.7},
    {"rank": 9, "agent": "Junie CLI", "model": "Multiple", "date": "2026-03-07", "agent_org": "JetBrains", "model_org": "Multiple", "accuracy": 71.0, "stderr": 2.9},
    {"rank": 10, "agent": "CodeBrain-1", "model": "GPT-5.3-Codex", "date": "2026-02-10", "agent_org": "Feeling AI", "model_org": "OpenAI", "accuracy": 70.3, "stderr": 2.6},
    {"rank": 11, "agent": "Droid", "model": "Claude Opus 4.6", "date": "2026-02-05", "agent_org": "Factory", "model_org": "Anthropic", "accuracy": 69.9, "stderr": 2.5},
    {"rank": 12, "agent": "Ante", "model": "Gemini 3 Pro", "date": "2026-01-06", "agent_org": "Antigma Labs", "model_org": "Google", "accuracy": 69.4, "stderr": 2.1},
    {"rank": 13, "agent": "Crux", "model": "Claude Opus 4.6", "date": "2026-02-23", "agent_org": "Roam", "model_org": "Anthropic", "accuracy": 66.9, "stderr": None},
    {"rank": 14, "agent": "Deep Agents", "model": "GPT-5.2-Codex", "date": "2026-02-12", "agent_org": "LangChain", "model_org": "OpenAI", "accuracy": 66.5, "stderr": 3.1},
    {"rank": 15, "agent": "Mux", "model": "Claude Opus 4.6", "date": "2026-02-13", "agent_org": "Coder", "model_org": "Anthropic", "accuracy": 66.5, "stderr": 2.5},
    {"rank": 16, "agent": "SageAgent", "model": "Gemini 3 Pro", "date": "2026-02-23", "agent_org": "OpenSage", "model_org": "Google", "accuracy": 65.2, "stderr": 2.1},
    {"rank": 17, "agent": "Droid", "model": "GPT-5.2", "date": "2025-12-24", "agent_org": "Factory", "model_org": "OpenAI", "accuracy": 64.9, "stderr": 2.8},
    {"rank": 18, "agent": "Terminus 2", "model": "GPT-5.3-Codex", "date": "2026-02-05", "agent_org": "Terminal Bench", "model_org": "OpenAI", "accuracy": 64.7, "stderr": 2.7},
    {"rank": 19, "agent": "Junie CLI", "model": "Gemini 3 Flash", "date": "2025-12-23", "agent_org": "JetBrains", "model_org": "Google", "accuracy": 64.3, "stderr": 2.8},
    {"rank": 20, "agent": "Droid", "model": "Claude Opus 4.5", "date": "2025-12-11", "agent_org": "Factory", "model_org": "Anthropic", "accuracy": 63.1, "stderr": 2.7},
    {"rank": 21, "agent": "Codex CLI", "model": "GPT-5.2", "date": "2025-12-18", "agent_org": "OpenAI", "model_org": "OpenAI", "accuracy": 62.9, "stderr": 3.0},
    {"rank": 22, "agent": "Terminus 2", "model": "Claude Opus 4.6", "date": "2026-02-06", "agent_org": "Terminal Bench", "model_org": "Anthropic", "accuracy": 62.9, "stderr": 2.7},
    {"rank": 23, "agent": "CodeBrain-1", "model": "Gemini 3 Pro", "date": "2026-02-05", "agent_org": "Feeling AI", "model_org": "Google", "accuracy": 62.2, "stderr": 2.6},
    {"rank": 24, "agent": "II-Agent", "model": "Gemini 3 Pro", "date": "2025-12-23", "agent_org": "Intelligent Internet", "model_org": "Google", "accuracy": 61.8, "stderr": 2.8},
    {"rank": 25, "agent": "Warp", "model": "Multiple", "date": "2025-12-12", "agent_org": "Warp", "model_org": "Multiple", "accuracy": 61.2, "stderr": 3.0},
    {"rank": 26, "agent": "Droid", "model": "Gemini 3 Pro", "date": "2025-12-24", "agent_org": "Factory", "model_org": "Google", "accuracy": 61.1, "stderr": 2.8},
    {"rank": 27, "agent": "Mux", "model": "GPT-5.2", "date": "2026-01-17", "agent_org": "Coder", "model_org": "OpenAI", "accuracy": 60.7, "stderr": None},
    {"rank": 28, "agent": "Codex CLI", "model": "GPT-5.1-Codex-Max", "date": "2025-11-24", "agent_org": "OpenAI", "model_org": "OpenAI", "accuracy": 60.4, "stderr": 2.7},
    {"rank": 29, "agent": "Letta Code", "model": "Claude Opus 4.5", "date": "2025-12-17", "agent_org": "Letta", "model_org": "Anthropic", "accuracy": 59.1, "stderr": 2.4},
    {"rank": 30, "agent": "Warp", "model": "Multiple", "date": "2025-11-20", "agent_org": "Warp", "model_org": "Multiple", "accuracy": 59.1, "stderr": 2.8},
    {"rank": 31, "agent": "Abacus AI Desktop", "model": "Multiple", "date": "2025-12-11", "agent_org": "Abacus.AI", "model_org": "Multiple", "accuracy": 58.4, "stderr": 2.8},
    {"rank": 32, "agent": "Mux", "model": "Claude Opus 4.5", "date": "2026-01-17", "agent_org": "Coder", "model_org": "Anthropic", "accuracy": 58.4, "stderr": None},
    {"rank": 33, "agent": "Claude Code", "model": "Claude Opus 4.6", "date": "2026-02-07", "agent_org": "Anthropic", "model_org": "Anthropic", "accuracy": 58.0, "stderr": 2.9},
    {"rank": 34, "agent": "Crux", "model": "GPT-5.1-Codex", "date": "2025-11-16", "agent_org": "Roam", "model_org": "OpenAI", "accuracy": 57.8, "stderr": 2.9},
    {"rank": 35, "agent": "Terminus 2", "model": "Claude Opus 4.5", "date": "2025-11-22", "agent_org": "Terminal Bench", "model_org": "Anthropic", "accuracy": 57.8, "stderr": 2.5},
    {"rank": 36, "agent": "Terminus 2", "model": "Gemini 3 Pro", "date": "2025-11-21", "agent_org": "Terminal Bench", "model_org": "Google", "accuracy": 56.9, "stderr": 2.5},
    {"rank": 37, "agent": "Letta Code", "model": "Gemini 3 Pro", "date": "2025-12-17", "agent_org": "Letta", "model_org": "Google", "accuracy": 56.0, "stderr": 3.0},
    {"rank": 38, "agent": "Goose", "model": "Claude Opus 4.5", "date": "2025-12-11", "agent_org": "Block", "model_org": "Anthropic", "accuracy": 54.3, "stderr": 2.6},
    {"rank": 39, "agent": "Terminus 2", "model": "GPT-5.2", "date": "2025-12-12", "agent_org": "Terminal Bench", "model_org": "OpenAI", "accuracy": 54.0, "stderr": 2.9},
    {"rank": 40, "agent": "Letta Code", "model": "GPT-5.1-Codex", "date": "2025-12-17", "agent_org": "Letta", "model_org": "OpenAI", "accuracy": 53.5, "stderr": 2.8},
    {"rank": 41, "agent": "Terminus 2", "model": "GLM 5", "date": "2026-02-23", "agent_org": "Terminal Bench", "model_org": "Z-AI", "accuracy": 52.4, "stderr": 2.6},
    {"rank": 42, "agent": "Claude Code", "model": "Claude Opus 4.5", "date": "2025-12-18", "agent_org": "Anthropic", "model_org": "Anthropic", "accuracy": 52.1, "stderr": 2.5},
    {"rank": 43, "agent": "OpenHands", "model": "Claude Opus 4.5", "date": "2026-01-04", "agent_org": "OpenHands", "model_org": "Anthropic", "accuracy": 51.9, "stderr": 2.9},
    {"rank": 44, "agent": "OpenCode", "model": "Claude Opus 4.5", "date": "2026-01-12", "agent_org": "Anomaly Innovations", "model_org": "Anthropic", "accuracy": 51.7, "stderr": None},
    {"rank": 45, "agent": "Terminus 2", "model": "Gemini 3 Flash", "date": "2026-01-07", "agent_org": "Terminal Bench", "model_org": "Google", "accuracy": 51.7, "stderr": 3.1},
    {"rank": 46, "agent": "Gemini CLI", "model": "Gemini 3 Flash", "date": "2025-12-23", "agent_org": "Google", "model_org": "Google", "accuracy": 51.0, "stderr": 3.0},
    {"rank": 47, "agent": "Warp", "model": "Multiple", "date": "2025-11-11", "agent_org": "Warp", "model_org": "Multiple", "accuracy": 50.1, "stderr": 2.7},
    {"rank": 48, "agent": "Codex CLI", "model": "GPT-5", "date": "2025-11-04", "agent_org": "OpenAI", "model_org": "OpenAI", "accuracy": 49.6, "stderr": 2.9},
    {"rank": 49, "agent": "Terminus 2", "model": "GPT-5.1", "date": "2025-11-16", "agent_org": "Terminal Bench", "model_org": "OpenAI", "accuracy": 47.6, "stderr": 2.8},
    {"rank": 50, "agent": "Gemini CLI", "model": "Gemini 3 Flash", "date": "2026-03-06", "agent_org": "Google", "model_org": "Google", "accuracy": 47.4, "stderr": 3.0},
    {"rank": 51, "agent": "CAMEL-AI", "model": "Claude Sonnet 4.5", "date": "2025-12-24", "agent_org": "CAMEL-AI", "model_org": "Anthropic", "accuracy": 46.5, "stderr": 2.4},
    {"rank": 52, "agent": "Codex CLI", "model": "GPT-5-Codex", "date": "2025-11-04", "agent_org": "OpenAI", "model_org": "OpenAI", "accuracy": 44.3, "stderr": 2.7},
    {"rank": 53, "agent": "OpenHands", "model": "GPT-5", "date": "2025-11-02", "agent_org": "OpenHands", "model_org": "OpenAI", "accuracy": 43.8, "stderr": 3.0},
    {"rank": 54, "agent": "Terminus 2", "model": "GPT-5-Codex", "date": "2025-10-31", "agent_org": "Terminal Bench", "model_org": "OpenAI", "accuracy": 43.4, "stderr": 2.9},
    {"rank": 55, "agent": "Terminus 2", "model": "Kimi K2.5", "date": "2026-02-04", "agent_org": "Terminal Bench", "model_org": "Kimi", "accuracy": 43.2, "stderr": 2.9},
    {"rank": 56, "agent": "Crux", "model": "GPT-5.1-Codex-Mini", "date": "2025-11-17", "agent_org": "Roam", "model_org": "OpenAI", "accuracy": 43.1, "stderr": 3.0},
    {"rank": 57, "agent": "Goose", "model": "Claude Sonnet 4.5", "date": "2025-12-11", "agent_org": "Block", "model_org": "Anthropic", "accuracy": 43.1, "stderr": 2.6},
    {"rank": 58, "agent": "Terminus 2", "model": "Claude Sonnet 4.5", "date": "2025-10-31", "agent_org": "Terminal Bench", "model_org": "Anthropic", "accuracy": 42.8, "stderr": 2.8},
    {"rank": 59, "agent": "MAYA", "model": "Claude Sonnet 4.5", "date": "2026-01-04", "agent_org": "ADYA", "model_org": "Anthropic", "accuracy": 42.7, "stderr": None},
    {"rank": 60, "agent": "OpenHands", "model": "Claude Sonnet 4.5", "date": "2025-11-02", "agent_org": "OpenHands", "model_org": "Anthropic", "accuracy": 42.6, "stderr": 2.8},
    {"rank": 61, "agent": "Mini-SWE-Agent", "model": "Claude Sonnet 4.5", "date": "2025-11-03", "agent_org": "Princeton", "model_org": "Anthropic", "accuracy": 42.5, "stderr": 2.8},
    {"rank": 62, "agent": "Terminus 2", "model": "Minimax m2.5", "date": "2026-02-23", "agent_org": "Terminal Bench", "model_org": "Minimax", "accuracy": 42.2, "stderr": 2.6},
    {"rank": 63, "agent": "Mini-SWE-Agent", "model": "GPT-5-Codex", "date": "2025-11-03", "agent_org": "Princeton", "model_org": "OpenAI", "accuracy": 41.3, "stderr": 2.8},
    {"rank": 64, "agent": "Claude Code", "model": "Claude Sonnet 4.5", "date": "2025-11-04", "agent_org": "Anthropic", "model_org": "Anthropic", "accuracy": 40.1, "stderr": 2.9},
    {"rank": 65, "agent": "Terminus 2", "model": "DeepSeek-V3.2", "date": "2026-02-10", "agent_org": "Terminal Bench", "model_org": "DeepSeek", "accuracy": 39.6, "stderr": 2.8},
    {"rank": 66, "agent": "Terminus 2", "model": "Claude Opus 4.1", "date": "2025-10-31", "agent_org": "Terminal Bench", "model_org": "Anthropic", "accuracy": 38.0, "stderr": 2.6},
    {"rank": 67, "agent": "OpenHands", "model": "Claude Opus 4.1", "date": "2025-11-02", "agent_org": "OpenHands", "model_org": "Anthropic", "accuracy": 36.9, "stderr": 2.7},
    {"rank": 68, "agent": "Terminus 2", "model": "GPT-5.1-Codex", "date": "2025-11-17", "agent_org": "Terminal Bench", "model_org": "OpenAI", "accuracy": 36.9, "stderr": 3.2},
    {"rank": 69, "agent": "Crux", "model": "MiniMax M2.1", "date": "2025-12-22", "agent_org": "Roam", "model_org": "MiniMax", "accuracy": 36.6, "stderr": 2.9},
    {"rank": 70, "agent": "Terminus 2", "model": "Kimi K2 Thinking", "date": "2025-11-11", "agent_org": "Terminal Bench", "model_org": "Moonshot AI", "accuracy": 35.7, "stderr": 2.8},
    {"rank": 71, "agent": "Goose", "model": "Claude Haiku 4.5", "date": "2025-12-11", "agent_org": "Block", "model_org": "Anthropic", "accuracy": 35.5, "stderr": 2.9},
    {"rank": 72, "agent": "Terminus 2", "model": "GPT-5", "date": "2025-10-31", "agent_org": "Terminal Bench", "model_org": "OpenAI", "accuracy": 35.2, "stderr": 3.1},
    {"rank": 73, "agent": "Mini-SWE-Agent", "model": "Claude Opus 4.1", "date": "2025-11-03", "agent_org": "Princeton", "model_org": "Anthropic", "accuracy": 35.1, "stderr": 2.5},
    {"rank": 74, "agent": "spoox-m", "model": "GPT-5-Mini", "date": "2025-12-24", "agent_org": "TUM", "model_org": "OpenAI", "accuracy": 34.8, "stderr": 2.7},
    {"rank": 75, "agent": "Claude Code", "model": "Claude Opus 4.1", "date": "2025-11-04", "agent_org": "Anthropic", "model_org": "Anthropic", "accuracy": 34.8, "stderr": 2.9},
    {"rank": 76, "agent": "Mini-SWE-Agent", "model": "GPT-5", "date": "2025-11-03", "agent_org": "Princeton", "model_org": "OpenAI", "accuracy": 33.9, "stderr": 2.9},
    {"rank": 77, "agent": "Terminus 2", "model": "GLM 4.7", "date": "2026-01-28", "agent_org": "Terminal Bench", "model_org": "Z-AI", "accuracy": 33.4, "stderr": 2.8},
    {"rank": 78, "agent": "Crux", "model": "GLM 4.7", "date": "2026-02-08", "agent_org": "Roam", "model_org": "Z-AI", "accuracy": 33.3, "stderr": 2.5},
    {"rank": 79, "agent": "Terminus 2", "model": "Gemini 2.5 Pro", "date": "2025-10-31", "agent_org": "Terminal Bench", "model_org": "Google", "accuracy": 32.6, "stderr": 3.0},
    {"rank": 80, "agent": "Codex CLI", "model": "GPT-5-Mini", "date": "2025-11-04", "agent_org": "OpenAI", "model_org": "OpenAI", "accuracy": 31.9, "stderr": 3.0},
    {"rank": 81, "agent": "Terminus 2", "model": "MiniMax M2", "date": "2025-11-01", "agent_org": "Terminal Bench", "model_org": "MiniMax", "accuracy": 30.0, "stderr": 2.7},
    {"rank": 82, "agent": "Mini-SWE-Agent", "model": "Claude Haiku 4.5", "date": "2025-11-03", "agent_org": "Princeton", "model_org": "Anthropic", "accuracy": 29.8, "stderr": 2.5},
    {"rank": 83, "agent": "Terminus 2", "model": "MiniMax M2.1", "date": "2025-12-23", "agent_org": "Terminal Bench", "model_org": "MiniMax", "accuracy": 29.2, "stderr": 2.9},
    {"rank": 84, "agent": "OpenHands", "model": "GPT-5-Mini", "date": "2025-11-02", "agent_org": "OpenHands", "model_org": "OpenAI", "accuracy": 29.2, "stderr": 2.8},
    {"rank": 85, "agent": "Terminus 2", "model": "Claude Haiku 4.5", "date": "2025-10-31", "agent_org": "Terminal Bench", "model_org": "Anthropic", "accuracy": 28.3, "stderr": 2.9},
    {"rank": 86, "agent": "Terminus 2", "model": "Kimi K2 Instruct", "date": "2025-11-01", "agent_org": "Terminal Bench", "model_org": "Moonshot AI", "accuracy": 27.8, "stderr": 2.5},
    {"rank": 87, "agent": "Claude Code", "model": "Claude Haiku 4.5", "date": "2025-11-04", "agent_org": "Anthropic", "model_org": "Anthropic", "accuracy": 27.5, "stderr": 2.8},
    {"rank": 88, "agent": "Dakou Agent", "model": "Qwen 3 Coder 480B", "date": "2025-12-28", "agent_org": "iflow", "model_org": "Alibaba", "accuracy": 27.2, "stderr": 2.6},
    {"rank": 89, "agent": "OpenHands", "model": "Grok 4", "date": "2025-11-02", "agent_org": "OpenHands", "model_org": "xAI", "accuracy": 27.2, "stderr": 3.1},
    {"rank": 90, "agent": "OpenHands", "model": "Kimi K2 Instruct", "date": "2025-11-02", "agent_org": "OpenHands", "model_org": "Moonshot AI", "accuracy": 26.7, "stderr": 2.7},
    {"rank": 91, "agent": "Mini-SWE-Agent", "model": "Gemini 2.5 Pro", "date": "2025-11-03", "agent_org": "Princeton", "model_org": "Google", "accuracy": 26.1, "stderr": 2.5},
    {"rank": 92, "agent": "Mini-SWE-Agent", "model": "Grok Code Fast 1", "date": "2025-11-03", "agent_org": "Princeton", "model_org": "xAI", "accuracy": 25.8, "stderr": 2.6},
    {"rank": 93, "agent": "Mini-SWE-Agent", "model": "Grok 4", "date": "2025-11-03", "agent_org": "Princeton", "model_org": "xAI", "accuracy": 25.4, "stderr": 2.9},
    {"rank": 94, "agent": "OpenHands", "model": "Qwen 3 Coder 480B", "date": "2025-11-02", "agent_org": "OpenHands", "model_org": "Alibaba", "accuracy": 25.4, "stderr": 2.6},
    {"rank": 95, "agent": "Terminus 2", "model": "GLM 4.6", "date": "2025-11-01", "agent_org": "Terminal Bench", "model_org": "Z.ai", "accuracy": 24.5, "stderr": 2.4},
    {"rank": 96, "agent": "Terminus 2", "model": "GPT-5-Mini", "date": "2025-10-31", "agent_org": "Terminal Bench", "model_org": "OpenAI", "accuracy": 24.0, "stderr": 2.5},
    {"rank": 97, "agent": "Terminus 2", "model": "Qwen 3 Coder 480B", "date": "2025-11-01", "agent_org": "Terminal Bench", "model_org": "Alibaba", "accuracy": 23.9, "stderr": 2.8},
    {"rank": 98, "agent": "Terminus 2", "model": "Grok 4", "date": "2025-10-31", "agent_org": "Terminal Bench", "model_org": "xAI", "accuracy": 23.1, "stderr": 2.9},
    {"rank": 99, "agent": "Mini-SWE-Agent", "model": "GPT-5-Mini", "date": "2025-11-03", "agent_org": "Princeton", "model_org": "OpenAI", "accuracy": 22.2, "stderr": 2.6},
    {"rank": 100, "agent": "Gemini CLI", "model": "Gemini 2.5 Pro", "date": "2025-11-04", "agent_org": "Google", "model_org": "Google", "accuracy": 19.6, "stderr": 2.9},
    {"rank": 101, "agent": "Terminus 2", "model": "GPT-OSS-120B", "date": "2025-11-01", "agent_org": "Terminal Bench", "model_org": "OpenAI", "accuracy": 18.7, "stderr": 2.7},
    {"rank": 102, "agent": "Mini-SWE-Agent", "model": "Gemini 2.5 Flash", "date": "2025-11-03", "agent_org": "Princeton", "model_org": "Google", "accuracy": 17.1, "stderr": 2.5},
    {"rank": 103, "agent": "Terminus 2", "model": "Gemini 2.5 Flash", "date": "2025-10-31", "agent_org": "Terminal Bench", "model_org": "Google", "accuracy": 16.9, "stderr": 2.4},
    {"rank": 104, "agent": "OpenHands", "model": "Gemini 2.5 Pro", "date": "2025-11-02", "agent_org": "OpenHands", "model_org": "Google", "accuracy": 16.4, "stderr": 2.8},
    {"rank": 105, "agent": "OpenHands", "model": "Gemini 2.5 Flash", "date": "2025-11-02", "agent_org": "OpenHands", "model_org": "Google", "accuracy": 16.4, "stderr": 2.4},
    {"rank": 106, "agent": "Gemini CLI", "model": "Gemini 2.5 Flash", "date": "2025-11-04", "agent_org": "Google", "model_org": "Google", "accuracy": 15.4, "stderr": 2.3},
    {"rank": 107, "agent": "Mini-SWE-Agent", "model": "GPT-OSS-120B", "date": "2025-11-03", "agent_org": "Princeton", "model_org": "OpenAI", "accuracy": 14.2, "stderr": 2.3},
    {"rank": 108, "agent": "Terminus 2", "model": "Grok Code Fast 1", "date": "2025-10-31", "agent_org": "Terminal Bench", "model_org": "xAI", "accuracy": 14.2, "stderr": 2.5},
    {"rank": 109, "agent": "OpenHands", "model": "Claude Haiku 4.5", "date": "2025-11-02", "agent_org": "OpenHands", "model_org": "Anthropic", "accuracy": 13.9, "stderr": 2.7},
    {"rank": 110, "agent": "Codex CLI", "model": "GPT-5-Nano", "date": "2025-11-04", "agent_org": "OpenAI", "model_org": "OpenAI", "accuracy": 11.5, "stderr": 2.3},
    {"rank": 111, "agent": "OpenHands", "model": "GPT-5-Nano", "date": "2025-11-02", "agent_org": "OpenHands", "model_org": "OpenAI", "accuracy": 9.9, "stderr": 2.1},
    {"rank": 112, "agent": "Terminus 2", "model": "GPT-5-Nano", "date": "2025-10-31", "agent_org": "Terminal Bench", "model_org": "OpenAI", "accuracy": 7.9, "stderr": 1.9},
    {"rank": 113, "agent": "Mini-SWE-Agent", "model": "GPT-5-Nano", "date": "2025-11-03", "agent_org": "Princeton", "model_org": "OpenAI", "accuracy": 7.0, "stderr": 1.9},
    {"rank": 114, "agent": "Mini-SWE-Agent", "model": "GPT-OSS-20B", "date": "2025-11-03", "agent_org": "Princeton", "model_org": "OpenAI", "accuracy": 3.4, "stderr": 1.4},
    {"rank": 115, "agent": "Terminus 2", "model": "GPT-OSS-20B", "date": "2025-11-01", "agent_org": "Terminal Bench", "model_org": "OpenAI", "accuracy": 3.1, "stderr": 1.5},
]
# fmt: on


def get_org_slug(org_name: str) -> str:
    return ORG_SLUG_MAP.get(org_name, org_name.lower().replace(" ", "-").replace(".", "-"))


def get_model_slug(model_name: str) -> str:
    return model_name.lower().replace(" ", "-")


def make_model_id(model_org: str, model_name: str) -> str:
    return f"{get_org_slug(model_org)}/{get_model_slug(model_name)}"


def convert_entry(entry: dict, retrieved_timestamp: str) -> EvaluationLog:
    """Convert a single leaderboard entry to an EvaluationLog."""
    model_id = make_model_id(entry["model_org"], entry["model"])
    agent_slug = entry["agent"].lower().replace(" ", "-")
    model_slug = get_model_slug(entry["model"])

    eval_id = f"terminal-bench-2.0/{agent_slug}__{model_slug}/{retrieved_timestamp}"

    uncertainty = None
    if entry["stderr"] is not None:
        uncertainty = Uncertainty(
            standard_error=StandardError(value=entry["stderr"]),
            num_samples=435,
        )

    eval_result = EvaluationResult(
        evaluation_name="terminal-bench-2.0",
        source_data=SourceDataUrl(
            dataset_name="terminal-bench-2.0",
            source_type="url",
            url=[LEADERBOARD_URL],
        ),
        evaluation_timestamp=entry["date"],
        metric_config=MetricConfig(
            evaluation_description="Task resolution accuracy across 87 terminal tasks with 5 trials each",
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0,
            max_score=100,
        ),
        score_details=ScoreDetails(
            score=entry["accuracy"],
            uncertainty=uncertainty,
        ),
        generation_config=GenerationConfig(
            generation_args=GenerationArgs(
                agentic_eval_config=AgenticEvalConfig(
                    available_tools=[
                        AvailableTool(
                            name="terminal",
                            description="Full terminal/shell access",
                        ),
                    ],
                ),
                execution_command=f'harbor run -d terminal-bench@2.0 -a "{entry["agent"]}" -m "{entry["model"]}" -k 5',
            ),
        ),
    )

    return EvaluationLog(
        schema_version="0.2.0",
        evaluation_id=eval_id,
        retrieved_timestamp=retrieved_timestamp,
        evaluation_timestamp=entry["date"],
        source_metadata=SourceMetadata(
            source_name="Terminal-Bench 2.0",
            source_type="documentation",
            source_organization_name="Terminal-Bench",
            source_organization_url="https://www.tbench.ai",
            evaluator_relationship=EvaluatorRelationship.third_party,
        ),
        eval_library=EvalLibrary(name="harbor", version="unknown"),
        model_info=ModelInfo(
            name=entry["model"],
            id=model_id,
            developer=entry["model_org"],
            additional_details={
                "agent_name": entry["agent"],
                "agent_organization": entry["agent_org"],
            },
        ),
        evaluation_results=[eval_result],
    )


def save_as_v020(eval_log: EvaluationLog, output_dir: str, org_slug: str, model_slug: str) -> Path:
    """Save an EvaluationLog as schema v0.2.0 JSON (without eval_library)."""
    from helpers import sanitize_filename

    data = json.loads(eval_log.model_dump_json(exclude_none=True))
    data.pop("eval_library", None)

    dir_path = Path(output_dir) / sanitize_filename(org_slug) / sanitize_filename(model_slug)
    dir_path.mkdir(parents=True, exist_ok=True)
    filepath = dir_path / f"{uuid.uuid4()}.json"
    filepath.write_text(json.dumps(data, indent=2) + "\n")
    return filepath


def main():
    retrieved_timestamp = str(time.time())
    count = 0

    for entry in LEADERBOARD_DATA:
        try:
            eval_log = convert_entry(entry, retrieved_timestamp)
            org_slug = get_org_slug(entry["model_org"])
            model_slug = get_model_slug(entry["model"])
            filepath = save_as_v020(eval_log, OUTPUT_DIR, org_slug, model_slug)
            print(f"[{entry['rank']:3d}] {filepath}")
            count += 1
        except Exception as e:
            print(f"Error processing rank {entry['rank']} "
                  f"({entry['agent']} / {entry['model']}): {e}")

    print(f"\nGenerated {count} files in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
