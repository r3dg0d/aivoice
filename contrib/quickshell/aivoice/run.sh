#!/usr/bin/env bash
# aivoice widget backend. run.sh start VOICE MODE [ENGINE] | modstart PRESET STRENGTH [--flag v ...] | stop | status |
#   vcstatus | vcengines | presets | modset PRESET STRENGTH [k=v ...] | bypass | presetsave NAME PRESET STRENGTH [--flag v ...] |
#   s2sstatus | s2sstart | s2sstop | s2sasr NAME | s2spreset NAME | s2sllm on|off |
#   voices | doctor | devices | search Q | install ID | setup | models-install |
#   repro | reprocontrol pause|resume|stop OUTPUT   (paper reproductions; training runs as its own background process)
# Every live mode (VC, modulation) plays into the same aivoice-virtual-mic; only one runs at a time.
set -u
AIV="${AIVOICE_BIN:-$HOME/.local/bin/aivoice}"
A="$HOME/.local/share/aivoice"
export PATH="$A/gcroot-pulse/bin:$A/gcroot-espeak/bin:$HOME/.local/bin:$HOME/.nix-profile/bin:/etc/profiles/per-user/${USER:-$(id -un)}/bin:/run/current-system/sw/bin:$PATH"
st="${XDG_STATE_HOME:-$HOME/.local/state}/aivoice-widget"; mkdir -p "$st"
log="$HOME/.cache/aivoice-widget.log"; mkdir -p "$HOME/.cache"; pidf="$st/virtualmic.pid"
running() { [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; }
case "${1:-}" in
  start)
    [ -e "$HOME/.cache/aivoice-test.lock" ] && { echo "Busy: a test is running, try again in a minute"; exit 3; }
    running && { echo "Already running (pid $(cat "$pidf"))"; exit 0; }
    [ -n "${2:-}" ] || { echo "Pick a voice first"; exit 2; }
    eng="${4:-}"; extra=()
    case "$eng" in ""|default) ;; *) extra=(--engine "$eng") ;; esac
    [ -n "${5:-}" ] && [ "${5}" != "0" ] && extra+=(--denoise "$5")
    echo "== $(date '+%F %r') virtualmic --voice ${2} --mode ${3:-balanced} ${extra[*]:-}" >>"$log"
    setsid "$AIV" virtualmic --voice "$2" --mode "${3:-balanced}" "${extra[@]}" --consent-ack </dev/null >>"$log" 2>&1 &
    echo $! >"$pidf"; sleep 1.5
    running && echo "Virtual mic live: ${2} → select 'aivoice' as your mic" || { echo "virtualmic exited — see log"; rm -f "$pidf"; } ;;
  modstart)
    running && { echo "Already running (pid $(cat "$pidf")) - stop it first"; exit 0; }
    p="${2:-natural}"; s="${3:-65}"; shift 3 2>/dev/null || shift $#
    echo "== $(date '+%F %r') modulate $p --strength $s $*" >>"$log"
    setsid "$AIV" modulate "$p" --strength "$s" --quiet "$@" </dev/null >>"$log" 2>&1 &
    echo $! >"$pidf"; sleep 1.0
    running && echo "Modulation live: $p @ $s → select 'aivoice' as your mic" || { echo "modulate exited — see log"; rm -f "$pidf"; } ;;
  vcstatus) "$AIV" vc status --json 2>/dev/null || echo '{"running":false}' ;;
  vcengines) "$AIV" vc engines --json 2>/dev/null || echo '[]' ;;
  presets) "$AIV" modulate preset list --json 2>/dev/null || echo '[]' ;;
  modset)
    p="${2:-}"; s="${3:-}"; shift 3 2>/dev/null || shift $#; args=()
    [ -n "$p" ] && args+=(--preset "$p"); [ -n "$s" ] && args+=(--strength "$s")
    for kv in "$@"; do args+=(--param "$kv"); done
    "$AIV" vc set "${args[@]}" 2>&1 | tail -1 ;;
  bypass) "$AIV" vc control toggle-bypass 2>&1 | tail -1 ;;
  presetsave) n="$2"; p="$3"; s="$4"; shift 4; "$AIV" modulate preset save "$n" --from "$p" --strength "$s" "$@" 2>&1 | tail -1 ;;
  s2sstatus) "$AIV" experimental status --json 2>/dev/null || echo '{}' ;;
  s2sstart)
    echo "== $(date '+%F %r') experimental start" >>"$log"
    "$AIV" experimental enable >/dev/null 2>&1
    setsid "$AIV" experimental start --quiet --consent-ack </dev/null >>"$log" 2>&1 &
    sleep 1.5; echo "Speech-to-speech starting (see Logs)" ;;
  s2sstop) "$AIV" experimental stop 2>&1 | tail -1 ;;
  s2sasr) "$AIV" experimental asr use "$2" 2>&1 | tail -1 ;;
  s2spreset) "$AIV" experimental preset use "$2" 2>&1 | tail -1 ;;
  s2sllm) if [ "${2:-off}" = on ]; then "$AIV" llm enable 2>&1 | tail -1; else "$AIV" llm disable 2>&1 | tail -1; fi ;;
  stop) if running; then kill -- -"$(cat "$pidf")" 2>/dev/null || kill "$(cat "$pidf")"; rm -f "$pidf"; echo "Stopped"; else echo "Not running"; fi ;;
  status) printf '{"running":%s,"pid":"%s","version":"%s"}\n' "$(running && echo true || echo false)" "$(cat "$pidf" 2>/dev/null)" "$("$AIV" --version 2>/dev/null | awk '{print $NF}')" ;;
  voices) "$AIV" voices list --json 2>/dev/null || echo "[]" ;;
  doctor) "$AIV" doctor 2>&1 ;;
  devices) "$AIV" devices 2>&1 ;;
  search) "$AIV" search "${2:-}" --json 2>/dev/null || echo "{\"results\":[]}" ;;
  install) "$AIV" voices install --id "$2" 2>&1 | tail -15 ;;
  setup) "$AIV" virtualmic --setup-only --consent-ack 2>&1 | tail -8 ;;
  resynth) "$AIV" voices resynth "$2" 2>&1 | tail -3 ;;
  inputs)
    def=$(pactl get-default-source 2>/dev/null); cur=$(cat "$HOME/.config/aivoice/input-source" 2>/dev/null)
    pactl list sources 2>/dev/null | awk -v cur="${cur:-$def}" '
      /^\tName: /{n=$2} /^\tDescription: /{sub(/^\tDescription: /,""); d=$0
        if (n !~ /^aivoice/ && n !~ /\.monitor$/) { gsub(/"/,"",d); printf "%s{\"name\":\"%s\",\"desc\":\"%s\",\"current\":%s}", (c++?",":"["), n, d, (n==cur?"true":"false") } }
      END{ print (c?"]":"[]") }' ;;
  setinput) mkdir -p "$HOME/.config/aivoice"; printf '%s\n' "$2" >"$HOME/.config/aivoice/input-source"; pactl set-source-mute "$2" 0 2>/dev/null; echo "Input: $2 (restart the mic to apply)" ;;
  level)
    src="${2:-$(cat "$HOME/.config/aivoice/input-source" 2>/dev/null)}"; src="${src:-$(pactl get-default-source)}"
    timeout 0.4 parec -d "$src" --format=s16le --channels=1 --rate=16000 --raw 2>/dev/null | python3 -c "import sys,array,math;a=array.array('h',sys.stdin.buffer.read());r=math.sqrt(sum(x*x for x in a)/len(a)) if a else 0;print(round(min(1,r/3000),3))" ;;
  rt) shift; v="$1"; shift; "$AIV" voices rt "$v" --json "$@" 2>/dev/null || echo '{}' ;;
  models-install) "$AIV" models install meanvc2 --yes 2>&1 | tail -15 ;;
  fix)
    V="$HOME/.local/share/aivoice/venv/bin"; export LD_LIBRARY_PATH="$(cat "$HOME/.local/share/aivoice/ldpath")"
    case "${2:-}" in
      *pactl*|*PipeWire*)
        command -v pactl >/dev/null || { echo "pactl not found even with pulseaudio on PATH"; exit 1; }
        pactl info >/dev/null 2>&1 || systemctl --user restart pipewire pipewire-pulse wireplumber 2>&1
        sleep 1; pactl info 2>&1 | grep -E "Server Name|Default Sink" || echo "PipeWire pulse server not answering" ;;
      *MeanVC2*) "$AIV" models install meanvc2 --yes 2>&1 | tail -4 ;;
      PyTorch*|CUDA*) "$V/pip" install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2 ;;
      *Moonshine*) "$V/pip" install -e "$HOME/src/aivoice[moonshine]" 2>&1 | tail -2 ;;
      *llama*) nix-shell -p cudaPackages.cudatoolkit gcc13 cmake ninja --run "bash $HOME/.local/share/aivoice/build-llama-cuda.sh" 2>&1 | tail -2 ;;
      *"HF model"*) "$V/pip" install -e "$HOME/src/aivoice[hf]" 2>&1 | tail -2 ;;
      *"audio I/O"*) "$V/pip" install -e "$HOME/src/aivoice[realtime]" 2>&1 | tail -2 ;;
      *Silero*) "$V/pip" install --no-deps silero-vad onnxruntime 2>&1 | tail -2 ;;
      *Parakeet*) "$V/pip" install -c "$HOME/.local/share/aivoice/constraints.txt" 'nemo_toolkit[asr]' 2>&1 | tail -2 ;;
      *RVC*) nix-shell -p gcc gnumake --run "bash $HOME/.local/share/aivoice/build-rvc-venv.sh" 2>&1 | tail -2 ;;
      *) echo "No automatic fix for: ${2:-?}" ;;
    esac ;;
  log) tail -n 60 "$log" 2>/dev/null || echo "(no log yet)" ;;
  repro) "$AIV" reproduce panel 2>/dev/null || echo "[]" ;;
  reprocontrol) "$AIV" reproduce control "${2:-}" --output "${3:-}" 2>&1 | tail -1 ;;
  *) echo "usage: run.sh start VOICE [MODE] [ENGINE] [DENOISE]|modstart PRESET STRENGTH|stop|status|vcstatus|vcengines|presets|modset|bypass|presetsave|s2sstatus|s2sstart|s2sstop|voices|doctor|devices|search Q|install ID|setup|models-install|log|repro|reprocontrol" >&2; exit 2 ;;
esac
