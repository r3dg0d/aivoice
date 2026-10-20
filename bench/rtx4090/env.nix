# FHS environment for the GPU benchmark. pip/uv wheels (torch, CTranslate2, voxcpm, ...) expect a
# normal Linux filesystem; on NixOS they get one here, plus the driver's libcuda from /run/opengl-driver.
#
#   nix run .#bench-4090 -- --help          (flake)
#   nix-build bench/rtx4090/env.nix -A run && ./result/bin/aivoice-bench-4090 --help
#   nix run .#bench-env                     (an interactive shell inside the same environment)
{ pkgs ? import <nixpkgs> { }, src ? ../.. }:
let
  env = pkgs.buildFHSEnv {
    name = "aivoice-bench-env";
    targetPkgs = p: with p; [
      python311 uv git git-lfs cacert curl which coreutils gnutar gzip xz
      espeak-ng ffmpeg sox libsndfile portaudio alsa-lib pipewire pulseaudio
      gcc gnumake cmake pkg-config rustc cargo # sdists without wheels (pyopenjtalk, ...)
      zlib stdenv.cc.cc.lib openssl libxml2 glib libGL
    ];
    profile = ''
      export LD_LIBRARY_PATH=/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
      export SSL_CERT_FILE=''${SSL_CERT_FILE:-/etc/ssl/certs/ca-bundle.crt}
      export UV_PYTHON_PREFERENCE=only-system
    '';
    runScript = "bash";
  };
in
{
  inherit env;
  run = pkgs.writeShellScriptBin "aivoice-bench-4090" ''
    export AIVOICE_REPO="''${AIVOICE_REPO:-${src}}"
    exec ${env}/bin/aivoice-bench-env "$AIVOICE_REPO/bench/rtx4090/run.sh" "$@"
  '';
}
