"""Consent / disclosed synthetic-voice gates."""

from __future__ import annotations

import os
import sys

CONSENT_TEXT = """\
This tool produces SYNTHETIC voice conversion for research, VFX, avatars,
filmmaking, consenting demos, and disclosed synthetic media.

You must ONLY convert voices where you have consent (or own the rights),
and you must disclose synthetic audio when required by law or platform policy.

This tool makes NO anonymity claims. Misuse for impersonation, fraud, or
harassment is prohibited. Upstream MeanVC2 likewise asks for consent.
"""

ACK_ENV = "AIVOICE_CONSENT_ACK"


def require_consent(*, ack: bool = False) -> None:
    env_ack = os.environ.get(ACK_ENV, "").strip() in ("1", "true", "yes", "YES")
    if ack or env_ack:
        return
    print(CONSENT_TEXT, file=sys.stderr)
    print(
        "Re-run with --consent-ack (or export AIVOICE_CONSENT_ACK=1) after reading the above.",
        file=sys.stderr,
    )
    raise SystemExit(2)
