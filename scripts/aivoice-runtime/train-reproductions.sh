#!/usr/bin/env bash
# Train aivoice's paper-reproduction voice-conversion engines end to end (the "Experimental engines" in the
# QuickShell widget): RT-VC, Zero-VC, TVTSyn (paper preset) and TVTSyn-aivoice (80 ms variant), optionally
# StreamVoiceAnon+ when you have the ESD corpus.
#
#   train-reproductions.sh                    full run (LibriTTS train-clean-100), detached under systemd --user
#   train-reproductions.sh --quick            ~1 h smoke run on 200 dev-clean utterances (proves the pipeline)
#   train-reproductions.sh --status           what is done / running, step counts
#   train-reproductions.sh --stop             stop after a checkpoint (resume by running the script again)
#   train-reproductions.sh --engines rtvc,tvtsyn-aivoice --subset 360 --foreground
#
# Safe to stop and re-run at any time: every download resumes, every finished step leaves a marker, and
# training resumes from its last checkpoint. Weights trained here are RETRAINED_PAPER_REPRODUCTION, never
# "official", and are refused by aivoice until they pass the mandatory tiny-overfit gate. Nothing is uploaded.
# No sudo, never kills other processes; runs at moderate priority so the desktop stays usable.
set -euo pipefail

# ----------------------------------------------------------------------------------------------- config
ENGINES="rtvc,zerovc-repro,tvtsyn-repro,tvtsyn-aivoice"
ROOT="${AIVOICE_REPRO_HOME:-$HOME/.local/share/aivoice/repro}"
SUBSET="100"          # 100 | 360 | both  (LibriTTS train-clean-*)
QUICK=0; FOREGROUND=0; ESD=""; DEVICE="cuda:0"; DRY=0
AIV_VENV="${AIVOICE_VENV:-$HOME/.local/share/aivoice/venv}"
LDPATH_FILE="$HOME/.local/share/aivoice/ldpath"
UNIT="aivoice-repro-train"
THREADS="${AIVOICE_REPRO_THREADS:-8}"

# Pinned teachers (Hugging Face revisions checked 2026-10-23). aivoice never downloads teachers on its own;
# this script does, once, at these exact commits.
declare -A TEACHER_REPO=(
  [wavlm-large]="microsoft/wavlm-large@c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"
  [hubert-base]="facebook/hubert-base-ls960@dba3bb02fda4248b6e082697eee756de8fe8aa8a"
  [w2v-bert-2.0]="facebook/w2v-bert-2.0@da985ba0987f70aaeb84a80f2851cfac8c697a7b"
  [xvector]="speechbrain/spkrec-xvect-voxceleb@56895a2df401be4150a159f3a1c653f00051d477"
  [ecapa]="speechbrain/spkrec-ecapa-voxceleb@0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
  [emotion2vec]="emotion2vec/emotion2vec_plus_base@b318240bfe67db81a8c572ecb37ce9c3759b81c9"
)
SPARC_REV="2e6a07d4b35022d4366a2896a6293fe16aaed78a"   # cheoljun95/Speech-Articulatory-Coding (MIT)
SPARC_CKPT_SHA="8ba0900d6acf9827da01cc702000880543d62563bf14fc20404e6a1292d726cf"

# LibriTTS (CC BY 4.0), SHA-256 from torchaudio.datasets.libritts; OpenSLR publishes no checksums for it.
LIBRITTS_MIRRORS=(https://openslr.trmal.net/resources/60 https://openslr.elda.org/resources/60 https://www.openslr.org/resources/60)
declare -A LIBRITTS_SHA=(
  [dev-clean]=da0864e1bd26debed35da8a869dd5c04dfc27682921936de7cff9c8a254dbe1a
  [train-clean-100]=c5608bf1ef74bb621935382b8399c5cdd51cd3ee47cec51f00f885a64c6c7f6b
  [train-clean-360]=ce7cff44dcac46009d18379f37ef36551123a1dc4e5c8e4eb73ae57260de4886
)

# Steps per stage (full run). Paper-scale; on an RTX 4090 expect days for the whole set. Override with
# STEPS_<ENGINE>_<STAGE>=N, e.g. STEPS_RTVC_VOCODER=100000.
declare -A STEPS=(
  [rtvc.pitch]=100000 [rtvc.ema]=100000 [rtvc.vocoder]=300000
  [zerovc-repro.content]=200000 [zerovc-repro.vc]=500000
  [tvtsyn-repro.content]=200000 [tvtsyn-repro.synth]=500000
  [tvtsyn-aivoice.content]=200000 [tvtsyn-aivoice.synth]=500000
  [streamvoiceanon-plus.finetune]=50000
)
declare -A BATCH=(
  [rtvc.pitch]=64 [rtvc.ema]=64 [rtvc.vocoder]=32 [zerovc-repro.content]=32 [zerovc-repro.vc]=16
  [tvtsyn-repro.content]=32 [tvtsyn-repro.synth]=16 [tvtsyn-aivoice.content]=32 [tvtsyn-aivoice.synth]=16
  [streamvoiceanon-plus.finetune]=8
)
# The tiny-overfit gate keeps its 0.5x pass mark; these stages need longer to get there on real LibriTTS
# (measured: tvtsyn content 0.74x at 300 steps, passes at 1000; RT-VC vocoder 0.484x at ~2500).
declare -A OVERFIT=( [tvtsyn-repro.content]=1000 [tvtsyn-aivoice.content]=1000 [rtvc.vocoder]=2500 )

# ----------------------------------------------------------------------------------------------- helpers
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$ROOT/logs/train.log" >&2; }
die() { log "ERROR: $*"; exit 1; }
usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --engines) ENGINES="$2"; shift ;;
    --subset) SUBSET="$2"; shift ;;
    --root) ROOT="$2"; shift ;;
    --quick) QUICK=1 ;;
    --esd) ESD="$2"; ENGINES="$ENGINES,streamvoiceanon-plus"; shift ;;
    --device) DEVICE="$2"; shift ;;
    --foreground) FOREGROUND=1 ;;
    --dry-run) DRY=1; FOREGROUND=1 ;;
    --status) ACTION=status ;;
    --stop) ACTION=stop ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 2 ;;
  esac
  shift
done
ACTION="${ACTION:-train}"
[ "$QUICK" = 1 ] && ROOT="$ROOT/quick"
mkdir -p "$ROOT"/{data,models,runs,logs,exports}
DATA="$ROOT/data"; MODELS="$ROOT/models"; RUNS="$ROOT/runs"; EXPORTS="$ROOT/exports"

ldp="$(cat "$LDPATH_FILE" 2>/dev/null || true)"
export LD_LIBRARY_PATH="$ldp${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$AIV_VENV/bin:$PATH"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
export HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
AIV="$AIV_VENV/bin/aivoice"; PY="$AIV_VENV/bin/python"

latest_ckpt() {  # newest checkpoints/stepNNNNNNNN.pt (zero-padded, so glob order is step order); "" if none
  local f last=""
  for f in "$1"/checkpoints/step*.pt; do [ -e "$f" ] && last="$f"; done
  printf '%s' "$last"
}
done_mark() { [ -e "$RUNS/.done-$1" ]; }
mark() { [ "$DRY" = 1 ] || date '+%F %T' > "$RUNS/.done-$1"; }
run() {
  log "+ $*"; [ "$DRY" = 1 ] && return 0
  local rc=0; "$@" || rc=$?
  if [ "$rc" != 0 ]; then
    log "!! FAILED (exit $rc): $1 ${2:-} ${3:-} ${4:-}  -- full traceback: journalctl --user -u $UNIT (or the run's status.json)"
    log "!! fix the cause and re-run the same command; finished stages are skipped and training resumes"
    exit "$rc"
  fi
}

# ----------------------------------------------------------------------------------------------- status / stop
if [ "$ACTION" = status ]; then
  systemctl --user --no-pager status "$UNIT" 2>/dev/null | sed -n '1,4p' || echo "unit $UNIT not running"
  echo "root: $ROOT"; ls -1 "$RUNS"/.done-* 2>/dev/null | sed 's|.*/.done-|  done: |' || true
  for st in "$RUNS"/*/*/status.json "$RUNS"/*/status.json; do
    [ -f "$st" ] && "$PY" -c "import json,sys; s=json.load(open(sys.argv[1])); print(f\"  {sys.argv[1].split('/runs/')[1].rsplit('/',1)[0]}: {s.get('state')} step {s.get('step')}/{s.get('max_steps')} ETA {round((s.get('eta_s') or 0)/3600,1)} h\")" "$st"
  done
  exit 0
fi
if [ "$ACTION" = stop ]; then
  for d in "$RUNS"/*/*/ "$RUNS"/*/; do [ -f "$d/status.json" ] && "$AIV" reproduce control stop --output "$d" >/dev/null 2>&1 || true; done
  log "asked every running stage to save a checkpoint and stop"; exit 0
fi

# ----------------------------------------------------------------------------------------------- detach
if [ "$FOREGROUND" = 0 ]; then
  command -v systemd-run >/dev/null || die "systemd-run not found; use --foreground"
  if systemctl --user is-active --quiet "$UNIT"; then die "already running: $0 --status (stop with --stop)"; fi
  args=(--foreground --engines "$ENGINES" --subset "$SUBSET" --root "${AIVOICE_REPRO_HOME:-$HOME/.local/share/aivoice/repro}" --device "$DEVICE")
  [ "$QUICK" = 1 ] && args+=(--quick)
  [ -n "$ESD" ] && args+=(--esd "$ESD")
  systemd-run --user --unit="$UNIT" --collect --same-dir \
      -p CPUWeight=20 -p IOWeight=20 -p Nice=10 \
      -E HOME="$HOME" -E PATH="$PATH" -E XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-}" \
      "$(readlink -f "$0")" "${args[@]}" >/dev/null
  echo "started as systemd user unit $UNIT (CPUWeight=20 IOWeight=20 Nice=10)"
  echo "  follow:  journalctl --user -fu $UNIT      or  tail -f $ROOT/logs/train.log"
  echo "  status:  $0 --status    stop: $0 --stop    (re-run to resume)"
  exit 0
fi

# ----------------------------------------------------------------------------------------------- preflight
log "== aivoice paper-reproduction training  engines=$ENGINES  quick=$QUICK  subset=$SUBSET  root=$ROOT"
[ -x "$AIV" ] || die "aivoice venv not found at $AIV_VENV (set AIVOICE_VENV)"
"$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || die "CUDA not available to $PY"
free_gb=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc 0-9)
need=$([ "$QUICK" = 1 ] && echo 20 || ([ "$SUBSET" = 100 ] && echo 120 || echo 260))
[ "$free_gb" -ge "$need" ] || die "need about ${need} GB free under $ROOT, have ${free_gb} GB"
"$PY" - <<'EOF' || die "GPU is already busy with >16 GiB; training needs headroom (stop other GPU jobs first)"
import pynvml as n
n.nvmlInit(); m = n.nvmlDeviceGetMemoryInfo(n.nvmlDeviceGetHandleByIndex(0))
raise SystemExit(0 if m.used < 16 * 2**30 else 1)
EOF

pip_deps=(speech-articulatory-coding "speechbrain>=1.1,<1.2")
[ -n "$ESD" ] && pip_deps+=(funasr)
missing=$("$PY" - "${pip_deps[@]}" <<'EOF'
import importlib.util, sys
mods = {"speech-articulatory-coding": "sparc", "speechbrain>=1.1,<1.2": "speechbrain", "funasr": "funasr"}
print(" ".join(p for p in sys.argv[1:] if importlib.util.find_spec(mods.get(p, p)) is None))
EOF
)
if [ -n "$missing" ]; then
  cons="$HOME/.local/share/aivoice/constraints.txt"
  [ -f "$cons" ] || "$PY" -c "import torch,torchaudio;print(f'torch=={torch.__version__}\ntorchaudio=={torchaudio.__version__}')" > "$cons"
  log "installing training-only packages into the aivoice venv: $missing (torch pinned by $cons)"
  # shellcheck disable=SC2086
  run uv pip install --python "$PY" -c "$cons" $missing
fi

# ----------------------------------------------------------------------------------------------- data
fetch_libritts() {  # subset
  local s="$1" tgz="$DATA/$1.tar.gz" ok=0
  [ -e "$DATA/.extracted-$s" ] && return 0
  for m in "${LIBRITTS_MIRRORS[@]}"; do
    log "LibriTTS $s from $m"
    if [ "$DRY" = 1 ] || curl -fL --retry 5 --retry-delay 10 -C - -o "$tgz" "$m/$s.tar.gz"; then ok=1; break; fi
  done
  [ "$ok" = 1 ] || die "could not download LibriTTS $s from any mirror"
  [ "$DRY" = 1 ] && return 0
  log "verifying $s (SHA-256)"
  echo "${LIBRITTS_SHA[$s]}  $tgz" | sha256sum -c --quiet - || { rm -f "$tgz"; die "LibriTTS $s checksum mismatch (deleted; re-run)"; }
  tar -xzf "$tgz" -C "$DATA" && rm -f "$tgz" && touch "$DATA/.extracted-$s"
}
fetch_libritts dev-clean
if [ "$QUICK" = 0 ]; then
  case "$SUBSET" in 100) fetch_libritts train-clean-100 ;; 360) fetch_libritts train-clean-360 ;;
    both) fetch_libritts train-clean-100; fetch_libritts train-clean-360 ;; *) die "--subset 100|360|both" ;; esac
fi
if [ "$QUICK" = 1 ]; then
  # dev-clean rows are labelled split=dev by the LibriTTS scanner, so a quick run trains on a speaker split of it:
  # 6 utterances from each of 36 speakers for training, 4 held-out speakers for validation.
  TRAIN_DS="$DATA/quick-manifest.jsonl"; LIMIT=()
  if [ ! -s "$TRAIN_DS" ] && [ "$DRY" = 0 ]; then
    "$PY" - "$DATA/LibriTTS/dev-clean" "$TRAIN_DS" <<'PYEOF'
import collections, json, sys
from pathlib import Path
from aivoice.research_reproductions.common.data import build_manifest
tmp = Path(sys.argv[2]).with_suffix(".all.jsonl")
build_manifest(sys.argv[1], "libritts", tmp)
by = collections.defaultdict(list)
for line in tmp.read_text().splitlines():
    if line.strip():
        r = json.loads(line); by[r["speaker"]].append(r)
spk = sorted(by)
rows = [dict(r, split="dev" if i >= 36 else "train") for i, s in enumerate(spk[:40]) for r in by[s][:6]]
Path(sys.argv[2]).write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"quick manifest: {sum(r['split'] == 'train' for r in rows)} train / {sum(r['split'] == 'dev' for r in rows)} dev rows")
PYEOF
  fi
else TRAIN_DS="libritts:$DATA/LibriTTS/train-clean-$SUBSET"; [ "$SUBSET" = both ] && TRAIN_DS="libritts:$DATA/LibriTTS"; LIMIT=(); fi

# ----------------------------------------------------------------------------------------------- teachers
need_teachers=(wavlm-large)
[[ ",$ENGINES," == *",zerovc-repro,"* ]] && need_teachers+=(w2v-bert-2.0)
[[ ",$ENGINES," == *",tvtsyn-"* ]] && need_teachers+=(hubert-base xvector ecapa)
[ -n "$ESD" ] && need_teachers+=(emotion2vec)
for t in "${need_teachers[@]}"; do
  spec="${TEACHER_REPO[$t]}"; repo="${spec%@*}"; rev="${spec#*@}"; dest="$MODELS/$t"
  [ -e "$dest/.complete-$rev" ] && continue
  log "teacher $t: $repo @ ${rev:0:7}"
  [ "$DRY" = 1 ] && continue
  "$PY" - "$repo" "$rev" "$dest" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3],
                  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.ckpt", "*.yaml", "*.txt", "*.pt", "*.model"])
EOF
  touch "$dest/.complete-$rev"
done
# the default place `aivoice` looks for teachers when no --teacher/AIVOICE_TEACHER_* is given: exports and the live
# engines need them too. Only our own symlinks are replaced; a real directory the user put there is left alone.
TEACHERS_HOME="${XDG_DATA_HOME:-$HOME/.local/share}/aivoice/teachers"
if [ "$DRY" = 0 ]; then
  mkdir -p "$TEACHERS_HOME"
  for t in "${need_teachers[@]}"; do
    if [ -L "$TEACHERS_HOME/$t" ] || [ ! -e "$TEACHERS_HOME/$t" ]; then ln -sfn "$MODELS/$t" "$TEACHERS_HOME/$t"
    else log "teacher $t: keeping existing $TEACHERS_HOME/$t"; fi
  done
fi
if [[ ",$ENGINES," == *",rtvc,"* ]] && [ ! -e "$MODELS/sparc/.complete-$SPARC_REV" ]; then
  log "teacher sparc: cheoljun95/Speech-Articulatory-Coding @ ${SPARC_REV:0:7} (English 1500k)"
  if [ "$DRY" = 0 ]; then
    mkdir -p "$MODELS/sparc"
    base="https://huggingface.co/cheoljun95/Speech-Articulatory-Coding/resolve/$SPARC_REV"
    curl -fL --retry 5 -C - -o "$MODELS/sparc/model_english_1500k.ckpt" "$base/model_english_1500k.ckpt"
    echo "$SPARC_CKPT_SHA  $MODELS/sparc/model_english_1500k.ckpt" | sha256sum -c --quiet - || die "SPARC checkpoint checksum mismatch"
    curl -fsL -o "$MODELS/sparc/model_english_1500k.yaml" "$base/model_english_1500k.yaml"
    # SPARC loads `speech_model: microsoft/wavlm-large` from the Hub by name; point it at the pinned local copy
    sed -i "s|^speech_model:.*|speech_model: $MODELS/wavlm-large|" "$MODELS/sparc/model_english_1500k.yaml"
    touch "$MODELS/sparc/.complete-$SPARC_REV"
  fi
fi
T() { echo "--teacher $1=$MODELS/$1"; }

# ----------------------------------------------------------------------------------------------- training
train_stage() {  # engine stage outdir dataset [extra args...]
  local eng="$1" stage="$2" out="$3" ds="$4"; shift 4
  local key="$eng.$stage"
  local steps="${STEPS[$key]:-100000}" envsteps; envsteps="STEPS_$(echo "$eng" | tr 'a-z.-' 'A-Z__')_$(echo "$stage" | tr 'a-z' 'A-Z')"
  steps="${!envsteps:-$steps}"; [ "$QUICK" = 1 ] && steps=$(( ${OVERFIT[$key]:-300} + 400 ))
  done_mark "$key" && { log "$key: done"; return 0; }
  local ck; ck=$(latest_ckpt "$out")
  if [ -n "$ck" ]; then
    run "$AIV" reproduce resume "$eng" --output "$out"
  else
    run "$AIV" reproduce train "$eng" --stage "$stage" --dataset "$ds" --output "$out" --device "$DEVICE" \
        --precision bf16 --batch-size "${BATCH[$key]:-16}" --workers 4 --max-steps "$steps" \
        --checkpoint-every "$([ "$QUICK" = 1 ] && echo 200 || echo 5000)" --eval-every 2500 \
        --overfit-steps "${OVERFIT[$key]:-300}" "$@"
  fi
  [ "$DRY" = 1 ] || [ -n "$(latest_ckpt "$out")" ] || die "$key produced no checkpoint (see $out)"
  mark "$key"
}
prepare_once() {  # key engine out [args...]
  local key="$1" eng="$2" out="$3"; shift 3
  done_mark "$key" && return 0
  # shellcheck disable=SC2068
  run "$AIV" reproduce prepare "$eng" --dataset "$TRAIN_DS" --output "$out" --device "$DEVICE" "${LIMIT[@]}" $@
  mark "$key"
}
export_once() {  # engine [args...]
  local eng="$1"; shift
  done_mark "$eng.export" && return 0
  run "$AIV" reproduce export "$eng" "$@" --output "$EXPORTS/$eng.pt" --install
  mark "$eng.export"
}

IFS=, read -ra ENG_LIST <<< "$ENGINES"
for eng in "${ENG_LIST[@]}"; do
  R="$RUNS/$eng"
  log "======== $eng"
  case "$eng" in
    rtvc)
      # shellcheck disable=SC2046
      prepare_once rtvc.prepare rtvc "$R/prep" $(T sparc)
      train_stage rtvc pitch "$R/pitch" "$R/prep"
      train_stage rtvc ema "$R/ema" "$R/prep"
      # shellcheck disable=SC2046
      train_stage rtvc vocoder "$R/vocoder" "$R/prep" $(T wavlm-large)
      export_once rtvc --pitch "$(latest_ckpt "$R/pitch")" --ema "$(latest_ckpt "$R/ema")" --vocoder "$(latest_ckpt "$R/vocoder")" ;;
    zerovc-repro)
      # shellcheck disable=SC2046
      prepare_once zerovc.prep-content zerovc-repro "$R/prep-content" --stage content $(T w2v-bert-2.0)
      # shellcheck disable=SC2046
      train_stage zerovc-repro content "$R/content" "$R/prep-content" $(T w2v-bert-2.0)
      # shellcheck disable=SC2046
      prepare_once zerovc.prep-vc zerovc-repro "$R/prep-vc" --stage vc $(T wavlm-large)
      # shellcheck disable=SC2046
      train_stage zerovc-repro vc "$R/vc" "$R/prep-vc" --content-checkpoint "$(latest_ckpt "$R/content")" $(T wavlm-large)
      export_once zerovc-repro --checkpoint "$(latest_ckpt "$R/vc")" ;;
    tvtsyn-repro|tvtsyn-aivoice)
      # shellcheck disable=SC2046
      prepare_once "$eng.prepare" "$eng" "$R/prep" $(T hubert-base) $(T xvector) $(T ecapa)
      train_stage "$eng" content "$R/content" "$R/prep"
      train_stage "$eng" synth "$R/synth" "$R/prep" --content-checkpoint "$(latest_ckpt "$R/content")" --grad-checkpoint
      export_once "$eng" --checkpoint "$(latest_ckpt "$R/synth")" ;;
    streamvoiceanon-plus)
      [ -n "$ESD" ] || die "streamvoiceanon-plus needs --esd DIR (the ESD corpus; request it from its authors)"
      # shellcheck disable=SC2046
      TRAIN_DS="custom:$ESD" prepare_once svplus.prepare streamvoiceanon-plus "$R/prep" $(T emotion2vec)
      train_stage streamvoiceanon-plus finetune "$R/ft" "$R/prep" --grad-accum 4
      export_once streamvoiceanon-plus --checkpoint "$(latest_ckpt "$R/ft")" ;;
    *) die "unknown engine $eng (rtvc, zerovc-repro, tvtsyn-repro, tvtsyn-aivoice, streamvoiceanon-plus)" ;;
  esac
done
log "== all requested engines trained, exported and installed. They appear as runnable under"
log "   Experimental engines in the aivoice widget; try one with: aivoice live --engine <name> --voice <voice>"
[ "$QUICK" = 1 ] && log "   (QUICK run: a pipeline check on 211 dev-clean utterances, not usable quality)"
