# Contributing

- Python 3.11+.
- `pip install -e ".[dev]" && AIVOICE_MOCK_DEVICES=1 pytest`
- Keep consent gates; never add silent large downloads.
- Do not relicense MeanVC2 as MIT; keep Apache-2.0 for our wrapper + upstream notices.
- Prefer subprocess/API integration to vendoring their entire tree into git.
- Apache-2.0 for contributions to *this* wrapper.
