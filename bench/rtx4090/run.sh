#!/usr/bin/env bash
# aivoice zero-shot cloning benchmark for one GPU machine (built for an RTX 4090 on NixOS).
#
#   nix run github:r3dg0d/aivoice#bench-4090                # from anywhere
#   nix run .#bench-4090 -- --librispeech 6 --no-record    # from a checkout, options after --
#   bash bench/rtx4090/run.sh ...                          # any Linux with python3.11, uv, git, espeak-ng
#
# What it does (each step logs to $BENCH_HOME/logs, and a failing engine never stops the others):
#   1. one virtualenv per engine (VoxCPM2, VoiceStar, X-Voice: their torch/transformers pins conflict)
#      plus one for aivoice itself (+ faster-whisper for WER)
#   2. downloads the model weights (Hugging Face; ~10 GB total) after asking
#   3. builds an authorized dataset: your own voice (recorded now, or --reference) + LibriSpeech speakers
#   4. aivoice benchmark research: every engine x speaker x text x seed -> similarity, WER, RTF, VRAM, audio
#   5. ablations: raw references (no preparation), experiments C,F,G (CPU) and D,E,H (engines)
#   6. writes results/<stamp>/SUMMARY.md and a .tar.gz you can share
set -uo pipefail

usage() {
  cat <<'EOF'
Options:
  --home DIR            working directory (default ~/aivoice-bench; venvs, weights, data, results)
  --engines LIST        comma list (default voxcpm2,voicestar,xvoice)
  --record NAME         record your own voice now as speaker NAME (default: me; asks first)
  --reference FILE      use an existing recording of your own voice instead of recording
  --reference-text TXT  exact words in --reference (needed by VoiceStar)
  --no-record           no own-voice speaker
  --librispeech N       LibriSpeech test-clean speakers to add (default 4; 0 = none)
  --seeds LIST          default 0,1
  --quick               1 seed, 3 texts, skip the raw-reference ablation
  --skip-setup          reuse existing venvs and weights
  --yes                 do not ask before downloading
  --fake                plumbing test with fake engine packages (no GPU, no downloads of models)
Environment: UV_TORCH_BACKEND (default auto: torch build matching your driver; e.g. cu128 to force one)
EOF
}

REPO="${AIVOICE_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BENCH_HOME="${AIVOICE_BENCH_HOME:-$HOME/aivoice-bench}"
ENGINES="voxcpm2,voicestar,xvoice"
RECORD_NAME="me"
REFERENCE=""
REFERENCE_TEXT=""
LIBRI=4
SEEDS="0,1"
QUICK=0
SKIP_SETUP=0
YES=0
FAKE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --home) BENCH_HOME="$2"; shift ;;
    --engines) ENGINES="$2"; shift ;;
    --record) RECORD_NAME="$2"; shift ;;
    --reference) REFERENCE="$2"; shift ;;
    --reference-text) REFERENCE_TEXT="$2"; shift ;;
    --no-record) RECORD_NAME="" ;;
    --librispeech) LIBRI="$2"; shift ;;
    --seeds) SEEDS="$2"; shift ;;
    --quick) QUICK=1; SEEDS="0" ;;
    --skip-setup) SKIP_SETUP=1 ;;
    --yes|-y) YES=1 ;;
    --fake) FAKE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option $1"; usage; exit 2 ;;
  esac
  shift
done

STAMP="$(date +%Y%m%d-%H%M%S)"
V="$BENCH_HOME/venvs"; M="$BENCH_HOME/models"; DS="$BENCH_HOME/dataset"; LOGS="$BENCH_HOME/logs"
R="$BENCH_HOME/results/$STAMP"
mkdir -p "$V" "$M" "$LOGS" "$R"
# aivoice settings / profiles for the benchmark stay in the bench home, not your ~/.config.
export XDG_CONFIG_HOME="$BENCH_HOME/config" XDG_DATA_HOME="$BENCH_HOME/share"
export AIVOICE_CONSENT_ACK=1 AIVOICE_EVAL_ASR=whisper PYTHONUNBUFFERED=1
# uv picks the PyTorch build (CUDA 12.x / 13.x / CPU) that matches the installed NVIDIA driver.
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-auto}"
# NixOS: the driver's libcuda lives here; nvidia-smi in the system profile.
[ -d /run/opengl-driver/lib ] && export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[ -d /run/current-system/sw/bin ] && export PATH="$PATH:/run/current-system/sw/bin"
for lib in /usr/lib/libespeak-ng.so.1 /usr/lib64/libespeak-ng.so.1 /usr/lib/x86_64-linux-gnu/libespeak-ng.so.1; do
  [ -e "$lib" ] && export PHONEMIZER_ESPEAK_LIBRARY="$lib" && break
done

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok=(); failed=()
step() {  # step NAME CMD...: run, log, record outcome, never abort
  local name="$1"; shift
  say "$name"
  if "$@" > >(tee "$LOGS/$STAMP-$name.log") 2>&1; then ok+=("$name"); return 0; fi
  failed+=("$name (log: $LOGS/$STAMP-$name.log)"); echo "!! $name failed; continuing"; return 1
}
confirm() { [ "$YES" = 1 ] && return 0; read -r -p "$1 [y/N] " a; [[ "$a" =~ ^[Yy] ]]; }

PY="$(command -v python3.11 || command -v python3.12 || command -v python3)"
command -v uv >/dev/null || { echo "uv is required (nix run .#bench-4090 provides it)"; exit 1; }
MAIN="$V/main/bin/python"
AIV="$V/main/bin/aivoice"

say "aivoice benchmark  repo=$REPO  home=$BENCH_HOME  engines=$ENGINES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi not found: GPU numbers will be missing"

# ---------------------------------------------------------------- 1. venvs
setup_main() {
  [ -x "$MAIN" ] || uv venv -q -p "$PY" "$V/main"
  uv pip install -q -p "$MAIN" --reinstall-package aivoice "${REPO}[research,dev,audio]" faster-whisper
}
setup_engine() {  # setup_engine KEY PACKAGE
  [ -x "$V/$1/bin/python" ] || uv venv -q -p "$PY" "$V/$1"
  uv pip install -q -p "$V/$1/bin/python" "$2"
}
if [ "$SKIP_SETUP" = 0 ]; then
  step setup-aivoice setup_main || { echo "cannot continue without the aivoice venv"; exit 1; }
  if [ "$FAKE" = 0 ]; then
    [[ ",$ENGINES," == *",voxcpm2,"* ]] && step setup-voxcpm2 setup_engine voxcpm2 "voxcpm==2.0.3"
    [[ ",$ENGINES," == *",voicestar,"* ]] && step setup-voicestar setup_engine voicestar "voicestar==0.1.0"
    [[ ",$ENGINES," == *",xvoice,"* ]] && step setup-xvoice setup_engine xvoice "x-voice==0.1.1"
  fi
fi
# faster-whisper (CTranslate2) finds cuBLAS/cuDNN through LD_LIBRARY_PATH: reuse torch's pip-installed CUDA libs.
NV_LIBS="$("$MAIN" - <<'EOF' 2>/dev/null
import glob, os, site
print(":".join(sorted({os.path.dirname(p) for s in site.getsitepackages() for p in glob.glob(f"{s}/nvidia/*/lib/*.so*")})))
EOF
)"
[ -n "$NV_LIBS" ] && export LD_LIBRARY_PATH="$NV_LIBS:${LD_LIBRARY_PATH:-}"

cfg() { "$AIV" experimental config set "$1" "$2" >/dev/null; }

# ---------------------------------------------------------------- 2. weights
download() {  # download REPO_ID DIR [allow patterns...]
  local repo="$1" dir="$2"; shift 2
  "$MAIN" - "$repo" "$dir" "$@" <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo, d, *pats = sys.argv[1:]
print(snapshot_download(repo, local_dir=d, allow_patterns=pats or None))
EOF
}
if [ "$FAKE" = 1 ]; then
  FAKES="$REPO/tests/fakes/fakepkgs"
  export PYTHONPATH="$FAKES${PYTHONPATH:+:$PYTHONPATH}" AIVOICE_FAKE_LOG="$LOGS/$STAMP-fake-calls.jsonl"
  mkdir -p "$M/fake-voxcpm2" "$M/fake-xvoice"
  touch "$M/fake-xvoice/model_stage2.safetensors" "$M/fake-xvoice/srp_model.safetensors" "$M/fake-xvoice/vocab.txt"
  for e in voxcpm2 voicestar xvoice; do cfg "engines.$e.python" "$MAIN"; done
  cfg engines.voxcpm2.model_path "$M/fake-voxcpm2"; cfg engines.xvoice.model_path "$M/fake-xvoice"
  export AIVOICE_EVAL_ASR=none
else
  if [ "$SKIP_SETUP" = 0 ] && confirm "Download model weights from Hugging Face (VoxCPM2 ~5 GB, X-Voice ~2 GB; VoiceStar ~3 GB on first use)?"; then
    [[ ",$ENGINES," == *",voxcpm2,"* ]] && step weights-voxcpm2 download openbmb/VoxCPM2 "$M/VoxCPM2"
    [[ ",$ENGINES," == *",xvoice,"* ]] && step weights-xvoice download XRXRX/X-Voice "$M/X-Voice" "*.safetensors" "*.txt" "*.yaml" "*.json"
  fi
  for e in voxcpm2 voicestar xvoice; do [ -x "$V/$e/bin/python" ] && cfg "engines.$e.python" "$V/$e/bin/python"; done
  [ -d "$M/VoxCPM2" ] && cfg engines.voxcpm2.model_path "$M/VoxCPM2"
  [ -d "$M/X-Voice" ] && cfg engines.xvoice.model_path "$M/X-Voice"
fi
step engines-list "$AIV" engines list

# ---------------------------------------------------------------- 3. dataset
if [ ! -d "$DS" ] || [ -z "$(ls -A "$DS" 2>/dev/null)" ] || confirm "Rebuild the dataset in $DS?"; then
  rm -rf "$DS"
  args=(--out "$DS" --texts "$REPO/bench/rtx4090/texts.txt" --cache "$BENCH_HOME/cache")
  if [ -n "$REFERENCE" ]; then args+=(--reference "$REFERENCE" --name "${RECORD_NAME:-me}" --reference-text "$REFERENCE_TEXT")
  elif [ -n "$RECORD_NAME" ]; then args+=(--record "$RECORD_NAME"); fi
  [ "$LIBRI" -gt 0 ] && args+=(--librispeech "$LIBRI")
  if [ "$FAKE" = 1 ] && [ -z "$REFERENCE" ] && [ "$LIBRI" = 0 ]; then
    "$AIV" benchmark research --synthetic --engines espeak --output "$BENCH_HOME/_synth" >/dev/null 2>&1
    cp -r "$BENCH_HOME/_synth/_synthetic_dataset" "$DS"
  else
    "$MAIN" "$REPO/bench/rtx4090/prepare_speakers.py" "${args[@]}" || { echo "dataset preparation failed"; exit 1; }
  fi
fi
if [ "$QUICK" = 1 ]; then head -n 3 "$REPO/bench/rtx4090/texts.txt" > "$DS/texts.txt"; else cp "$REPO/bench/rtx4090/texts.txt" "$DS/texts.txt"; fi

# ---------------------------------------------------------------- 4-5. runs
step system-info bash -c "'$AIV' hardware; nvidia-smi 2>/dev/null; uname -a; cd '$REPO' && git rev-parse HEAD 2>/dev/null; for v in '$V'/*/; do echo \"## \$v\"; uv pip freeze -p \"\$v/bin/python\" 2>/dev/null | grep -Ei '^(torch|voxcpm|voicestar|x-voice|transformers|faster-whisper|ctranslate2|aivoice)'; done" \
  && cp "$LOGS/$STAMP-system-info.log" "$R/system-info.txt"
step benchmark "$AIV" benchmark research --dataset "$DS" --engines "$ENGINES" --seeds "$SEEDS" --save-audio --output "$R/benchmark"
[ "$QUICK" = 0 ] && step benchmark-raw-references "$AIV" benchmark research --dataset "$DS" --engines "$ENGINES" --seeds 0 --no-prepare --output "$R/benchmark-raw"
step experiments-CFG "$AIV" research experiments --run C,F,G --output "$R/experiments"
step experiments-DEH "$AIV" research experiments --run D,E,H --dataset "$DS" --results "$R/benchmark/results.json" \
  --engines "$ENGINES" --output "$R/experiments" --audio-out "$R/experiments/audio"

# ---------------------------------------------------------------- 6. summary
{
  echo "# aivoice benchmark $STAMP"; echo
  echo "Engines: $ENGINES. Seeds: $SEEDS. Quick: $QUICK. Fake engines: $FAKE."; echo
  [ -f "$R/benchmark/report.md" ] && sed '1d' "$R/benchmark/report.md"
  if [ -f "$R/benchmark-raw/report.md" ]; then echo; echo "## Same run with raw (unprepared) references"; echo; grep '^|' "$R/benchmark-raw/report.md"; fi
  echo; echo "## Experiments"; echo
  for f in "$R"/experiments/*.json; do
    [ -f "$f" ] && "$MAIN" -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"- **{d['id']}** {d['title']}: {d['status']}. {d['decision']}\")" "$f"
  done
  echo; echo "## Steps"; for s in "${ok[@]}"; do echo "- ok: $s"; done; for s in "${failed[@]}"; do echo "- FAILED: $s"; done
  echo; echo "Audio: benchmark/audio/<engine>/<speaker>/ and experiments/audio/ (all labelled synthetic)."
  echo "LibriSpeech-based clones are for benchmarking only; publish only speakers whose consent.json allows it."
} > "$R/SUMMARY.md"
cp "$DS"/*/consent.json "$R/" 2>/dev/null || true
for d in "$DS"/*/; do [ -f "$d/consent.json" ] && mkdir -p "$R/consent" && cp "$d/consent.json" "$R/consent/$(basename "$d").json"; done
rm -f "$R"/consent.json
mkdir -p "$R/logs" && cp "$LOGS/$STAMP"-*.log "$R/logs/" 2>/dev/null || true
tar -C "$BENCH_HOME/results" -czf "$BENCH_HOME/results/aivoice-bench-$STAMP.tar.gz" "$STAMP"

say "done"
cat "$R/SUMMARY.md"
echo
echo "Results: $R"
echo "Archive: $BENCH_HOME/results/aivoice-bench-$STAMP.tar.gz"
[ ${#failed[@]} -eq 0 ]
