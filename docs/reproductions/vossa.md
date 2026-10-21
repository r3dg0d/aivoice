# VOSSA (official release, integrated)

VOSSA has public official code **and** official checkpoints (github.com/Morris88826/VOSSA @ `5dd1a5c`,
`morris88826/VOSSA` on Hugging Face), so aivoice does not reproduce it. Engine `vossa` runs the official
`VossaStreamer` in a worker process (`AIVOICE_VOSSA_PYTHON`, `AIVOICE_VOSSA_REPO_DIR`,
`AIVOICE_VOSSA_CHECKPOINT_DIR`). The release is **CC BY-NC 4.0** (non-commercial only).

Lightning checkpoints are pickles: loading them requires `AIVOICE_VOSSA_TRUST_PICKLE=1`, the same explicit
opt-in as the other engines.

```bash
git clone https://github.com/Morris88826/VOSSA ~/.cache/aivoice/vendor/VOSSA
cd ~/.cache/aivoice/vendor/VOSSA && hf download morris88826/VOSSA --local-dir checkpoints
python3.10 -m venv ~/.local/share/aivoice/vossa-venv && ~/.local/share/aivoice/vossa-venv/bin/pip install -r requirements.txt
export AIVOICE_VOSSA_PYTHON=~/.local/share/aivoice/vossa-venv/bin/python AIVOICE_VOSSA_TRUST_PICKLE=1
aivoice live --engine vossa --voice "Target" --consent-ack
```

The VOSSA content encoder is the TVTSyn backbone. aivoice's TVTSyn reproduction (`tvtsyn-repro`) was
checked against this code (see [tvtsyn.md](tvtsyn.md)), but no VOSSA file is copied into aivoice.
