"""QuickShell widget: run.sh backend commands (with a fake aivoice) and an offscreen load against stub Quickshell modules."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parent.parent
WIDGET = ROOT / "contrib" / "quickshell" / "aivoice"


@pytest.fixture
def fake_aivoice(tmp_path):
    log = tmp_path / "calls.log"
    b = tmp_path / "aivoice"
    b.write_text(f"#!/usr/bin/env bash\necho \"$*\" >> {log}\n"
                 "case \"$1 $2\" in\n"
                 "  'vc status') echo '{\"running\": false}' ;;\n"
                 "  'vc engines') echo '[]' ;;\n"
                 "  'modulate '*) [ \"$2\" = preset ] && echo '[]' || sleep 3 ;;\n"
                 "  'virtualmic '*) sleep 3 ;;\n"
                 "  *) echo ok ;;\n"
                 "esac\n")
    b.chmod(0o755)
    return b, log


def run_sh(fake, *args, home):
    env = {**os.environ, "AIVOICE_BIN": str(fake), "HOME": str(home), "XDG_STATE_HOME": str(home / "state")}
    return subprocess.run(["bash", str(WIDGET / "run.sh"), *args], capture_output=True, text=True, env=env, timeout=20)


def test_runsh_live_commands(fake_aivoice, tmp_path):
    fake, log = fake_aivoice
    home = tmp_path / "home"
    home.mkdir()
    assert json.loads(run_sh(fake, "vcstatus", home=home).stdout) == {"running": False}
    run_sh(fake, "modset", "masculine", "65", "pitch_semitones=-2", "formant_ratio=0.9", home=home)
    run_sh(fake, "bypass", home=home)
    r = run_sh(fake, "modstart", "masculine", "65", "--formants", "-10", home=home)
    assert "Modulation live" in r.stdout
    assert "Already running" in run_sh(fake, "start", "V", "balanced", "xvc", home=home).stdout  # one virtual mic at a time
    assert "Stopped" in run_sh(fake, "stop", home=home).stdout
    r = run_sh(fake, "start", "My Voice", "balanced", "xvc", "0.3", home=home)
    assert "Virtual mic live" in r.stdout
    run_sh(fake, "stop", home=home)
    run_sh(fake, "start", "My Voice", "best-quality", "default", "0", home=home)
    run_sh(fake, "stop", home=home)
    time.sleep(0.2)
    calls = log.read_text().splitlines()
    assert "vc status --json" in calls
    assert "vc set --preset masculine --strength 65 --param pitch_semitones=-2 --param formant_ratio=0.9" in calls
    assert "vc control toggle-bypass" in calls
    assert "modulate masculine --strength 65 --quiet --formants -10" in calls
    assert "virtualmic --voice My Voice --mode balanced --engine xvc --denoise 0.3 --consent-ack" in calls
    assert "virtualmic --voice My Voice --mode best-quality --consent-ack" in calls  # voice default: previous engine path


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash")
def test_runsh_s2s_commands(fake_aivoice, tmp_path):
    fake, log = fake_aivoice
    home = tmp_path / "home"
    home.mkdir()
    for a in (("s2sasr", "moonshine-small"), ("s2spreset", "balanced"), ("s2sllm", "on"), ("s2sllm", "off"), ("s2sstop",)):
        run_sh(fake, *a, home=home)
    calls = log.read_text().splitlines()
    assert {"experimental asr use moonshine-small", "experimental preset use balanced", "llm enable", "llm disable",
            "experimental stop"} <= set(calls)


def _qt_ok() -> bool:
    if importlib.util.find_spec("PySide6") is None:
        return False
    r = subprocess.run([sys.executable, "-c", "from PySide6.QtGui import QGuiApplication"], capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not _qt_ok(), reason="PySide6 (with libEGL) not installed")
def test_widget_loads_and_binds_real_cli_output(tmp_path):
    """The real `aivoice ... --json` outputs drive every mode of the widget without QML warnings."""
    import numpy as np

    from aivoice.cli import main
    from aivoice.modulation.presets import get_preset
    from aivoice.vc.engines import DspEngine
    from aivoice.vc.live import LiveConfig, LivePipeline

    r = CliRunner()
    canned = tmp_path / "canned"
    canned.mkdir()
    (canned / "vcengines.out").write_text(r.invoke(main, ["vc", "engines", "--json"]).output)
    (canned / "presets.out").write_text(r.invoke(main, ["modulate", "preset", "list", "--json"]).output)
    (canned / "s2sstatus.out").write_text(r.invoke(main, ["experimental", "status", "--json"]).output)
    p = LivePipeline(LiveConfig(mode="modulation", engine="dsp", post_preset="masculine", post_strength=0.65),
                     DspEngine(get_preset("masculine").params()))
    p.start()
    p.run_array(np.random.default_rng(0).normal(0, 0.1, 24000).astype(np.float32), 48000)
    snap = {**p.snapshot(), "underruns": 0, "dropped_blocks": 0, "gpu": {"util_pct": 10.0, "mem_used_mib": 900},
            "worker": {"vram_allocated_mib": 512.0}}
    (canned / "vcstatus.out").write_text(json.dumps(snap))
    (canned / "status.out").write_text('{"running": true, "pid": "1", "version": "0.6.0"}')
    (canned / "voices.out").write_text('[{"name": "Test Voice", "id": "test-voice"}]')
    (canned / "rt.out").write_text('{"engine": "rvc", "pitch": 0, "index_rate": 0.5, "rvc_model": "/m.pth", "rvc_index": null}')
    (canned / "inputs.out").write_text('[{"name": "alsa_input.usb", "desc": "USB Mic", "current": true}]')
    (canned / "level.out").write_text("0.2")
    panel = json.loads(r.invoke(main, ["reproduce", "panel"]).output)
    panel[1].update(state="Training", output="/tmp/run", run={"state": "training", "step": 120, "max_steps": 1000,
                    "loss": {"mel": 1.2}, "samples_per_s": 3.5, "eta_s": 900, "gpu": "RTX 4090", "vram_gib": 9.1})
    (canned / "repro.out").write_text(json.dumps(panel))
    out = subprocess.run([sys.executable, str(ROOT / "tests" / "qml_harness.py"), str(canned), str(WIDGET / "shell.qml")],
                         capture_output=True, text=True, timeout=90)
    lines = (out.stdout + out.stderr).splitlines()
    warns = [ln for ln in lines if ln.startswith("WARN")]
    assert not warns, "\n".join(warns[:20])
    assert any(ln.startswith("PROPS") for ln in lines)
    cmds = {json.loads(ln.split(" ", 1)[1])[0] for ln in lines if ln.startswith("STUBCMD")}
    assert {"vcengines", "vcstatus", "presets", "status", "s2sstatus", "repro"} <= cmds
