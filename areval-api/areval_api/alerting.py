"""Multi-channel alert notifications (Slack / Email / Webhook / stdout).

A single ``alert_callback`` dispatches to all configured channels so
QualityMonitor alerts are never silently discarded.  Channels are
auto-configured from environment variables — zero channels configured
means alerts only print to stderr.

Environment variables
---------------------
AREVAL_SLACK_WEBHOOK : str, optional
    Slack incoming-webhook URL.
AREVAL_SMTP_HOST : str, optional
    SMTP server for email alerts.
AREVAL_SMTP_PORT : int, optional
    SMTP port (default 587).
AREVAL_SMTP_USER : str, optional
    SMTP login user.
AREVAL_SMTP_PASSWORD : str, optional
    SMTP login password (use app-specific password for Gmail).
AREVAL_ALERT_EMAIL : str, optional
    Comma-separated recipient addresses.
AREVAL_ALERT_WEBHOOK : str, optional
    Generic HTTP POST webhook (e.g. PagerDuty, custom endpoint).
"""

from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText
from typing import Callable, List

import requests   # type: ignore[import-untyped]

from areval.online.monitors import Alert


# ---------------------------------------------------------------------------
# Handler factories
# ---------------------------------------------------------------------------

def _slack_handler(webhook_url: str) -> Callable[[Alert], None]:
    def _send(alert: Alert) -> None:
        color = "#ff0000" if alert.severity == "critical" else "#ffa500"
        requests.post(webhook_url, json={
            "attachments": [{
                "color": color,
                "title": f"AREval: {alert.alert_type}",
                "text": alert.message,
                "fields": [
                    {"title": "Current", "value": f"{alert.current_value:.3f}", "short": True},
                    {"title": "Threshold", "value": f"{alert.threshold_value:.3f}", "short": True},
                ],
                "ts": int(alert.timestamp.timestamp()),
            }],
        }, timeout=5)
    return _send


def _email_handler(
    host: str,
    port: int,
    user: str,
    password: str,
    recipients: List[str],
) -> Callable[[Alert], None]:
    def _send(alert: Alert) -> None:
        subject = f"[AREval {alert.severity.upper()}] {alert.alert_type.replace('_', ' ').title()}"
        body = (
            f"Quality alert detected at {alert.timestamp.isoformat()}\n"
            f"\n"
            f"Type:      {alert.alert_type}\n"
            f"Severity:  {alert.severity}\n"
            f"Current:   {alert.current_value}\n"
            f"Threshold: {alert.threshold_value}\n"
            f"Message:   {alert.message}\n"
            f"\n"
            f"Window stats: {alert.window_stats}\n"
        )
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = user or "areval@localhost"
        msg["To"] = ", ".join(recipients)

        with smtplib.SMTP(host, port, timeout=10) as server:
            server.starttls()
            if user and password:
                server.login(user, password)
            server.send_message(msg)
    return _send


def _webhook_handler(url: str) -> Callable[[Alert], None]:
    def _send(alert: Alert) -> None:
        requests.post(url, json={
            "type": alert.alert_type,
            "severity": alert.severity,
            "message": alert.message,
            "current_value": alert.current_value,
            "threshold_value": alert.threshold_value,
            "timestamp": alert.timestamp.isoformat(),
            "window_stats": alert.window_stats,
        }, timeout=5)
    return _send


def _stdout_handler(alert: Alert) -> None:
    marker = "CRITICAL" if alert.severity == "critical" else "WARNING"
    print(f"[AREval {marker}] {alert.alert_type}: {alert.message} (value={alert.current_value})")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_alert_chain() -> Callable[[Alert], None]:
    """Build and return a dispatch function that fans out alerts.

    Channels are configured via environment variables.  At least
    ``_stdout_handler`` is always included so alerts are never
    fully silent.
    """
    handlers: List[Callable[[Alert], None]] = []

    # Slack
    webhook = os.environ.get("AREVAL_SLACK_WEBHOOK")
    if webhook:
        handlers.append(_slack_handler(webhook))

    # Email
    smtp_host = os.environ.get("AREVAL_SMTP_HOST")
    if smtp_host:
        handlers.append(_email_handler(
            host=smtp_host,
            port=int(os.environ.get("AREVAL_SMTP_PORT", "587")),
            user=os.environ.get("AREVAL_SMTP_USER", ""),
            password=os.environ.get("AREVAL_SMTP_PASSWORD", ""),
            recipients=[
                addr.strip()
                for addr in os.environ.get("AREVAL_ALERT_EMAIL", "").split(",")
                if addr.strip()
            ],
        ))

    # Generic webhook
    generic_url = os.environ.get("AREVAL_ALERT_WEBHOOK")
    if generic_url:
        handlers.append(_webhook_handler(generic_url))

    # Fallback: always print to stderr
    handlers.append(_stdout_handler)

    def _dispatch(alert: Alert) -> None:
        for handler in handlers:
            try:
                handler(alert)
            except Exception:
                pass  # one channel fails — keep going

    return _dispatch
