"""
approval_client.py

This file lets the agent request human approval from the model platform.

Any action that touches Production must create an approval request instead of
being executed directly.
"""

import os
import requests


class ApprovalClient:
    """
    Client used by the agent to create human approval requests.
    """

    def __init__(self):
        self.base_url = os.getenv(
            "MODEL_SERVICE_URL",
            "http://127.0.0.1:8000"
        )

    def request_approval(
        self,
        investigation_id: str,
        action_type: str,
        reason: str,
        target_model_version: str | None = None,
    ) -> dict:
        """
        Create an approval request in the model service.
        """

        response = requests.post(
            f"{self.base_url}/approvals/request",
            params={
                "investigation_id": investigation_id,
                "action_type": action_type,
                "reason": reason,
                "target_model_version": target_model_version,
            },
            timeout=10,
        )

        response.raise_for_status()

        return response.json()