"""
test_imports.py

This test verifies that the main project modules can be imported.

It catches broken package paths, missing __init__.py files, and import-time
errors before runtime.
"""


def test_service_app_imports():
    """
    The FastAPI model service should import successfully.
    """

    from service.main import app

    assert app.title == "Bank Marketing Model Service"


def test_agent_app_imports():
    """
    The FastAPI agent service should import successfully.
    """

    from agent.main import app

    assert app.title == "Drift Triage Agent Service"


def test_dashboard_client_imports():
    """
    Dashboard API client should import successfully.
    """

    import dashboard.api_client

    assert dashboard.api_client is not None