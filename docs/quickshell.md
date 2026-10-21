# QuickShell widget

The widget lives in [`contrib/quickshell/aivoice/`](../contrib/quickshell/aivoice/README.md). It is the copy of the
owner's "Galaxy AIVoice" panel that was added to the repository in commit 02bd6fa, with home-folder paths removed;
0.6 extends that copy. (No other QuickShell configuration exists in the build environment, so the installed widget
on the owner's machine is unchanged until they copy these files over.)

## Three modes

| Mode | What it controls | Starts |
|---|---|---|
| **VOICE CONVERSION** | target voice, engine (Voice default / Auto / X-VC / RVC / MeanVC2), experimental engines, quality preset, RVC pitch + index, noise suppression | `aivoice virtualmic --voice V --mode M [--engine E] [--denoise D]` |
| **VOICE MODULATION** | preset (natural, masculine, feminine, deep, warm, bright, broadcast, robotic, + all others), strength, advanced sliders (pitch, formants, resonance, body, brightness, roughness, breathiness, intonation), reset, A/B bypass, save preset | `aivoice modulate PRESET --strength S [...]` |
| **SPEECH-TO-SPEECH** | ASR engine, LLM rewrite on/off, latency/expression preset; TTS/voice shown | `aivoice experimental start` |

All live modes play into the single `aivoice-virtual-mic`; only one runs at a time, so switching mode or engine never
changes what Discord / OBS / VRChat record from.

## Honest controls

- Engine buttons come from `aivoice vc engines --json` (refreshed every 60 s). An engine that is not runnable on this
  machine is disabled and labelled with its real state: *missing model*, *missing runtime*, *research only*,
  *no public code/weights*, *license restriction*. Experimental engines (StyleStream, StreamVoiceAnon) sit in a
  collapsed section and are never picked by Auto.
- Sliders only exist for parameters the backend implements; modulation changes are applied to the running session
  without restarting it (`aivoice vc set`).
- Live metrics come from the session's own status file (`aivoice vc status --json`, written once a second; the widget
  polls every 2 s): engine (and fallback), algorithmic latency, engine p95, RTF, chunk size, GPU %, VRAM, underruns /
  drops, sample rates, input/output levels.

## Backend commands (`run.sh`)

`start VOICE MODE [ENGINE] [DENOISE]`, `modstart PRESET STRENGTH [--flag v ...]`, `stop`, `status`, `vcstatus`,
`vcengines`, `presets`, `modset PRESET STRENGTH [name=value ...]`, `bypass`, `presetsave NAME PRESET STRENGTH [...]`,
`s2sstatus`, `s2sstart`, `s2sstop`, `s2sasr NAME`, `s2spreset NAME`, `s2sllm on|off`, plus the existing `voices`,
`search`, `install`, `doctor`, `fix`, `inputs`, `setinput`, `level`, `rt`, `resynth`, `setup`, `models-install`, `log`.
`AIVOICE_BIN` overrides the `aivoice` binary (used by the tests).

IPC (`quickshell ipc -c aivoice call aivoice ...`): `show`, `open`, `toggle`, `tab NAME`, `mode vc|mod|s2s`, `bypass`.

## Tests

- `tests/test_quickshell.py` runs `run.sh` against a fake `aivoice` and checks the exact command lines.
- With PySide6 installed, the same test loads `shell.qml` offscreen against stub `Quickshell` / `Quickshell.Io`
  modules (`tests/fakes/quickshell`), feeds it the **real** JSON printed by `aivoice vc engines`, `modulate preset
  list`, `experimental status` and a live pipeline snapshot, switches through all three modes and fails on any QML
  warning. It caught four undefined-binding bugs and a missing-`~/.cache` bug in `run.sh` before release.
- Not tested here: the real Quickshell runtime, fonts and the compositor window rules.
