"""
tasks.py

This file defines the slow task handlers executed by the worker.

These are safe placeholders for now:
- replay_test
- retrain
- rollback

Later they will call real platform logic.
"""


def run_replay_test(payload: dict) -> dict:
    """
    Simulate replaying the test set against the current model.
    """

    return {
        "task": "replay_test",
        "status": "completed",
        "details": "Replay test placeholder completed.",
        "payload": payload,
    }


def run_retrain(payload: dict) -> dict:
    """
    Simulate retraining a candidate model.
    """

    return {
        "task": "retrain",
        "status": "completed",
        "details": "Retrain placeholder completed.",
        "payload": payload,
    }


def run_rollback(payload: dict) -> dict:
    """
    Simulate rollback preparation.

    Production rollback still requires human approval before registry changes.
    """

    return {
        "task": "rollback",
        "status": "completed",
        "details": "Rollback placeholder completed. No Production change made.",
        "payload": payload,
    }