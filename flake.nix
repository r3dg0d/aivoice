{
  description = "aivoice — local AI voice changer (MeanVC2 wrapper)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python313;
        # CUDA/torch OPTIONAL — devices/models list/--help work without NVIDIA.
        aivoice = python.pkgs.buildPythonApplication {
          pname = "aivoice";
          version = "0.2.0";
          src = ./.;
          format = "pyproject";
          nativeBuildInputs = with python.pkgs; [ hatchling ];
          propagatedBuildInputs = with python.pkgs; [ click numpy ];
          meta = with pkgs.lib; {
            description = "Local real-time AI voice changer (MeanVC2 wrapper)";
            license = licenses.asl20;
            mainProgram = "aivoice";
            platforms = platforms.linux;
          };
        };
      in {
        packages.default = aivoice;
        packages.aivoice = aivoice;
        apps.default = flake-utils.lib.mkApp { drv = aivoice; };
        devShells.default = pkgs.mkShell {
          buildInputs = [
            python
            python.pkgs.pip
            python.pkgs.click
            python.pkgs.numpy
            python.pkgs.pytest
            pkgs.pipewire
            pkgs.pulseaudio
            pkgs.alsa-utils
          ];
          shellHook = ''
            echo "aivoice dev shell (CPU-friendly). For MeanVC2 GPU: install torch+CUDA yourself."
          '';
        };
      });
}
