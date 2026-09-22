import pytest

from aivoice.consent import require_consent


def test_blocks(monkeypatch):
    monkeypatch.delenv("AIVOICE_CONSENT_ACK", raising=False)
    with pytest.raises(SystemExit) as ei:
        require_consent(ack=False)
    assert ei.value.code == 2


def test_ack(monkeypatch):
    monkeypatch.delenv("AIVOICE_CONSENT_ACK", raising=False)
    require_consent(ack=True)
