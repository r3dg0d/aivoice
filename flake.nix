{
  description = "aivoice — local AI voice changer (MeanVC2) + experimental ASR -> LLM -> Fish Drama 3 mode";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python313;
        py = python.pkgs;

        # Optional runtime pieces packaged in nixpkgs. Moonshine Voice, NeMo and
        # fish-speech are not: the dev shell installs them into a venv on demand
        # (see docs/nixos-pipewire.md), so nothing here forces CUDA or torch.
        experimentalPy = with py; [
          soundfile
          sounddevice
          onnxruntime
          huggingface-hub
          tqdm
          msgpack
          keyring
        ];

        bench = import ./bench/rtx4090/env.nix { inherit pkgs; src = self; };

        mkAivoice = extra: py.buildPythonApplication {
          pname = "aivoice";
          version = "0.5.0";
          src = ./.;
          format = "pyproject";
          nativeBuildInputs = [ py.hatchling ];
          propagatedBuildInputs = [ py.click py.numpy ] ++ extra;
          # pactl/parec/pacat and espeak-ng are found on PATH at runtime
          makeWrapperArgs = [
            "--suffix" "PATH" ":" (pkgs.lib.makeBinPath [ pkgs.pulseaudio pkgs.espeak-ng pkgs.ffmpeg ])
          ];
          doCheck = false;
          meta = with pkgs.lib; {
            description = "Local real-time AI voice changer (MeanVC2 wrapper + experimental ASR/TTS mode)";
            license = licenses.asl20;
            mainProgram = "aivoice";
            platforms = platforms.linux;
          };
        };
      in {
        packages.default = mkAivoice [ ];
        packages.aivoice = mkAivoice [ ];
        # CLI with the CPU-side experimental dependencies (VAD, HF manager, audio I/O).
        packages.aivoice-experimental = mkAivoice experimentalPy;
        apps.default = flake-utils.lib.mkApp { drv = self.packages.${system}.default; };
        # GPU cloning benchmark (docs/benchmark-rtx4090.md): `nix run .#bench-4090 -- --help`
        packages.bench-4090 = bench.run;
        apps.bench-4090 = flake-utils.lib.mkApp { drv = bench.run; exePath = "/bin/aivoice-bench-4090"; };
        # A shell inside the benchmark's FHS environment (venvs, uv, CUDA driver libs).
        apps.bench-env = flake-utils.lib.mkApp { drv = bench.env; exePath = "/bin/aivoice-bench-env"; };

        devShells.default = pkgs.mkShell {
          buildInputs = [
            python
            py.pip
            py.virtualenv
            py.click
            py.numpy
            py.pytest
            pkgs.pipewire
            pkgs.pulseaudio # pactl / pacat / parec (talk to pipewire-pulse)
            pkgs.wireplumber
            pkgs.alsa-utils
            pkgs.portaudio
            pkgs.libsndfile
            pkgs.espeak-ng
            pkgs.ffmpeg
          ] ++ experimentalPy;
          # PortAudio / libsndfile are dlopen()ed by the Python wheels.
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.portaudio pkgs.libsndfile pkgs.stdenv.cc.cc.lib ];
          shellHook = ''
            echo "aivoice dev shell (CPU-friendly; CUDA/torch are not forced)."
            echo "  Moonshine (CPU ASR):   python -m venv .venv && . .venv/bin/activate && pip install -e '.[dev,moonshine]'"
            echo "  Parakeet (GPU ASR):    install a CUDA torch build, then pip install 'nemo_toolkit[asr]'"
            echo "  Drama 3 / S2-Pro:      run fish-speech's api_server separately (docs/drama3.md)"
          '';
        };
      }) // {
        # Optional NixOS module: a persistent aivoice virtual mic loaded by pipewire-pulse
        # at login (survives restarts; aivoice reuses it). Nothing changes unless
        # you import it and set services.aivoice-virtual-mic.enable = true.
        nixosModules.virtual-mic = { config, lib, ... }:
          let cfg = config.services.aivoice-virtual-mic; in {
            options.services.aivoice-virtual-mic = {
              enable = lib.mkEnableOption "a persistent aivoice virtual microphone (PipeWire null sink + source)";
              name = lib.mkOption { type = lib.types.str; default = "aivoice"; };
            };
            config = lib.mkIf cfg.enable {
              # Same modules and names `aivoice` creates at runtime, so it finds and reuses them.
              services.pipewire.extraConfig.pipewire-pulse."90-aivoice-virtual-mic" = {
                "pulse.cmd" = [
                  {
                    cmd = "load-module";
                    args = "module-null-sink sink_name=${cfg.name}_sink channel_map=mono sink_properties=device.description=${cfg.name}-output";
                    flags = [ ];
                  }
                  {
                    cmd = "load-module";
                    args = "module-remap-source master=${cfg.name}_sink.monitor source_name=${cfg.name}_mic channel_map=mono source_properties=device.description=${cfg.name}-virtual-mic";
                    flags = [ ];
                  }
                ];
              };
            };
          };
      };
}
