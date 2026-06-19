"""
Generic outbound webhook client — for CRM sync, escalation alerts, etc.
Uses httpx (already a transitive dependency via anthropic).

Payload shape is left to the caller: pass a Slack-compatible {"text": ...}
to hit a Slack/Discord incoming webhook, or any JSON body for a custom
endpoint (Zapier, Make, internal service).
"""
import logging

import httpx

logger = logging.getLogger(__name__)


class WebhookClient:

    def __init__(self, url: str):
        self._url = url

    def post(self, payload: dict) -> bool:
        """POST a JSON payload to the configured URL. Returns True on 2xx."""
        try:
            r = httpx.post(self._url, json=payload, timeout=10)
            r.raise_for_status()
            return True
        except Exception as exc:
            logger.error("Webhook POST failed url=%s: %s", self._url, exc)
            return False
