# AI Voice Quickshell widget

Galaxy-styled [Quickshell](https://quickshell.org) panel for aivoice: voice library, engine/pitch/index controls (RVC v2 / MeanVC2), input picker with level meter, doctor/health check with one-click fixes, and virtual-mic start/stop.

## Install

```sh
mkdir -p ~/.config/quickshell/aivoice ~/.local/bin ~/.local/share/applications ~/.local/share/icons/hicolor/scalable/apps
cp shell.qml GalaxyButton.qml run.sh ~/.config/quickshell/aivoice/
install -m755 aivoice-widget ~/.local/bin/
cp galaxy-aivoice.desktop ~/.local/share/applications/
cp galaxy-aivoice.svg ~/.local/share/icons/hicolor/scalable/apps/
```

Requires `aivoice` on `~/.local/bin`, `quickshell`, and `pactl`/`parec` (pipewire-pulse).

## Launch

`aivoice-widget` (or "AI Voice" from fuzzel/any launcher). A second launch reopens the running instance via `quickshell ipc`.

## niri window rule

```kdl
window-rule {
    match title="^Galaxy AIVoice$"
    open-floating true
}
```

The `fix` actions in `run.sh` call the helpers in `scripts/aivoice-runtime/` (expected at `~/.local/share/aivoice/`).
