"""
conftest.py

This file makes sure pytest can import project packages from the repository root.

It prevents ModuleNotFoundError for packages like:
- service
- agent
- async_queue
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))