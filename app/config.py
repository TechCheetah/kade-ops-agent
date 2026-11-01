import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

MODEL = os.getenv("AGENT_MODEL", "claude-opus-5-5")
EFFORT = os.getenv("AGENT_EFFORT", "medium")
MAX_STEPS = 8
MAX_QUESTION_CHARS = 1000

# USD per million tokens: input, output, cache read, cache write (5 min)
PRICING = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}

# Models with server-side refusal fallback ("default" form)
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}

RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "12"))
DAILY_BUDGET_USD = float(os.getenv("DAILY_BUDGET_USD", "3"))
