set -e
R=~/.local/share/aivoice/rvc-venv
export LD_LIBRARY_PATH=$(cat ~/.local/share/aivoice/ldpath)
"${PYTHON310:-$(uv python find 3.10 2>/dev/null || command -v python3.10)}" -m venv $R
$R/bin/pip install -q "pip<24.1" setuptools wheel
$R/bin/pip install torch==2.4.1 torchaudio==2.4.1 --extra-index-url https://download.pytorch.org/whl/cu121
$R/bin/pip install rvc-python
echo RVC_DONE
