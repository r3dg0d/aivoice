# NixOS and PipeWire

## Dev shell

```bash
nix develop            # python 3.13, pipewire, pactl/pacat/parec, portaudio, libsndfile, espeak-ng, ffmpeg,
                       # plus soundfile, sounddevice, onnxruntime, huggingface-hub, msgpack, keyring
python -m venv .venv --system-site-packages && . .venv/bin/activate
pip install -e '.[dev,moonshine]'        # moonshine-voice is not in nixpkgs
pip install --no-deps silero-vad         # VAD model file only
```

`nix build .#aivoice` is the minimal CLI; `nix build .#aivoice-experimental`
adds the CPU-side experimental dependencies. Neither forces CUDA or torch.

CUDA (Parakeet, or a local fish-speech server) on NixOS: use the NVIDIA driver from
your system configuration (`hardware.nvidia`), and either a CUDA-enabled torch in a
venv with `LD_LIBRARY_PATH` including `/run/opengl-driver/lib`, or a container
(`virtualisation.docker` + nvidia-container-toolkit) for the fish-speech server.
Ubuntu `apt` instructions do not apply.

Status: `flake.nix` is syntax-checked (`nix-instantiate --parse`); it has not been
built in this repository's CI environment.

## Virtual microphone

At runtime `aivoice experimental start` (and `aivoice virtualmic`) creates, through
pipewire-pulse:

* `aivoice_sink` (null sink, description `aivoice-output`): aivoice plays here.
* `aivoice_mic` (remap source of its monitor, description `aivoice-virtual-mic`):
  select this in Discord / OBS / VRChat / games.

It reuses existing ones instead of duplicating them, records the module ids it
loaded in `$XDG_RUNTIME_DIR/aivoice/`, and on exit unloads only those. Output is
written with `pacat --device=aivoice_sink`, which targets the sink by name.

### Persistent (optional, needs your approval to change system config)

The flake exports a NixOS module that loads the same two modules at login, so
apps always see the mic and aivoice simply reuses it:

```nix
# flake.nix of your system
inputs.aivoice.url = "github:r3dg0d/aivoice";
# ...
modules = [ inputs.aivoice.nixosModules.virtual-mic { services.aivoice-virtual-mic.enable = true; } ];
```

This writes `services.pipewire.extraConfig.pipewire-pulse."90-aivoice-virtual-mic"`.
Nothing else in your PipeWire configuration changes. Without NixOS, the
equivalent is a `pipewire-pulse.conf.d/90-aivoice.conf` with the same `pulse.cmd`
entries.

### Monitoring and feedback

* `experimental start --monitor` adds a loopback from the virtual mic to your
  default output. **Use headphones**: speakers feed your own synthesized voice back
  into the microphone.
* aivoice refuses to use `aivoice_mic` or `aivoice_sink.monitor` as its input.
* Choose the input with `aivoice experimental config set audio.input_device <pulse source name>`
  (`pactl list short sources`); `parec` is used for named sources.

### Latency on PipeWire

Default quantum is often 1024/48000 (21 ms). For lower latency:
`pw-metadata -n settings 0 clock.force-quantum 256` (temporary), or set
`default.clock.min-quantum` in your PipeWire config.

## Tested

Virtual mic create/reuse/cleanup and real audio routing (espeak-ng TTS ->
`pacat` -> `aivoice_sink` -> recorded back from `aivoice_mic` with `parec`) were
run against a real PulseAudio daemon. PipeWire-pulse uses the same protocol, but a
real PipeWire session has not been exercised in this repository's CI.
