# src/common/logging.py

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure application-wide logging.

    This should be called once from script entry points, FastAPI startup,
    workers, or dashboard startup.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )