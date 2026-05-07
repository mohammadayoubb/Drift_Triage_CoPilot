"""
conftest.py

Makes sure pytest can import project packages from the repository root
and from src/ so that both v1 packages (agent/, service/) and v2 packages
(src/agent/, src/ml/) are importable.
"""

import sys
from pathlib import Path

# Project root — contains service/, agent/, src/, async_queue/, etc.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# src/ — resolves `from agent.xxx` to src/agent/ (v2 LangGraph agent)
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
