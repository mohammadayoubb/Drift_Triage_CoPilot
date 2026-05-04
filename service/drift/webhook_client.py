"""
webhook_client.py

This file sends drift events from the model service to the agent service.

The platform sends a webhook when drift severity changes.
"""

import requests


class DriftWebhookClient:
    """
    HTTP client for notifying the agent about drift events.
    """

    def __init__(self, agent_webhook_url: str):
        self.agent_webhook_url = agent_webhook_url

    def send_drift_event(self, event: dict) -> dict:
        """
        Send drift event to the agent service.
        """

        response = requests.post(
            self.agent_webhook_url,
            json=event,
            timeout=10
        )

        response.raise_for_status()

        return response.json()