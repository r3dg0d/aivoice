# Security Policy

## Reporting

Report vulnerabilities privately. Do not attach illegal materials.

## Product boundaries

- Consent acknowledgment required for live/file/virtualmic (`--consent-ack` or `AIVOICE_CONSENT_ACK=1`).
- Model downloads require explicit `--yes`; never silent multi-GB pulls.
- Upstream MeanVC2 LICENSE file gap is documented in NOTICE — do not strip notices.
- Do not use for non-consensual voice impersonation, fraud, or harassment.
- Experimental mode is local by default; remote TTS destinations require explicit, per-destination consent
  (`aivoice experimental remote allow`). Microphone audio is never uploaded; there is no telemetry.
- API keys/tokens are read from the environment or the system keyring; `settings.json` refuses secret-like keys,
  and error messages are redacted. Hugging Face tokens are handled only by `huggingface_hub`.
- Downloaded models are treated as untrusted: aivoice never enables `trust_remote_code`, warns about pickle
  weights and `.nemo` archives, and verifies SHA-256 against the Hub manifest before marking a snapshot verified.
- TTS voice profiles require an authorization attestation; hosted voice ids are never scraped or exported.
