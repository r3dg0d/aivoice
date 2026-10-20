"""CLI: zero-shot cloning (clone), engine catalogue (engines), research corpus (research)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click


def _fail(e: BaseException) -> click.ClickException:
    return click.ClickException(str(e))


def _json(d: Any) -> None:
    click.echo(json.dumps(d, indent=2, default=str))


def register(main: click.Group) -> None:
    main.add_command(clone_group)
    main.add_command(engines_group)
    main.add_command(research_group)


# =============================================================================
# clone
# =============================================================================


@click.group("clone")
def clone_group():
    """Zero-shot voice cloning profiles and synthesis (VoxCPM2, X-Voice, VoiceStar, Fish, MeanVC2).

    Profiles live in ~/.local/share/aivoice/speakers (v2). Existing `voices` /
    `voice` profiles are untouched; `clone migrate` copies them in.
    """


@clone_group.command("create")
@click.argument("name")
@click.option("--reference", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--transcript", default=None, help="Exact words in the reference (optional; some engines need it)")
@click.option("--language", default=None)
@click.option("--engine", default="auto", help="Engine to prepare for (informational; any compatible engine can use the profile)")
@click.option("--denoise", type=click.Choice(["auto", "on", "off"]), default="auto", show_default=True,
              help="auto: only when the reference is measurably noisy (identity-guarded)")
@click.option("--max-seconds", type=float, default=20.0, show_default=True, help="Keep the best window of this length")
@click.option("--no-prepare", is_flag=True, help="Use the reference as-is")
@click.option("--keep-original/--no-keep-original", default=True, show_default=True,
              help="Retain an untouched copy for comparison")
@click.option("--publishable", is_flag=True, help="The speaker also agreed to publication of synthetic samples")
@click.option("--i-have-rights", is_flag=True, help="Attest you may clone this voice")
def clone_create(name, reference, transcript, language, engine, denoise, max_seconds, no_prepare, keep_original,
                 publishable, i_have_rights):
    """Analyse, prepare and store a reference as a v2 cloning profile."""
    from .research.speaker_identity import get_encoder
    from .speakers import ProfileError, create_profile

    try:
        p = create_profile(name, reference, i_have_rights=i_have_rights, transcript=transcript, language=language,
                           keep_original=keep_original, publishable=publishable, prepare=not no_prepare,
                           denoise={"on": True, "off": False}.get(denoise, "auto"), max_seconds=max_seconds,
                           encoder=get_encoder())
    except (ProfileError, ValueError) as e:
        raise _fail(e) from e
    r = p.reference_report
    click.echo(f"created {p.voice_name} ({p.id})")
    click.echo(f"  speech {r.get('speech_s', 0):.1f}s of {r.get('duration_s', 0):.1f}s, SNR ~{r.get('snr_db') or 0:.0f} dB, "
               f"quality score {r.get('score')}")
    click.echo(f"  preparation: {', '.join(p.preparation.get('steps', [])) or 'none'}"
               + (f"; reverted {p.preparation.get('reverted')}" if p.preparation.get("reverted") else ""))
    for i in r.get("issues", []):
        click.echo(f"  note: {i}")
    if not p.identity:
        click.echo("  (no speaker embedding: install torch + librosa + resemblyzer for similarity metrics)")


@clone_group.command("list")
@click.option("--json", "as_json", is_flag=True)
def clone_list(as_json):
    from .speakers import list_profiles

    items = list_profiles()
    if as_json:
        _json([p.to_dict() for p in items])
        return
    if not items:
        click.echo("(no clone profiles)  aivoice clone create NAME --reference ref.wav --i-have-rights")
        return
    click.echo(f"{'NAME':<24} {'ID':<20} {'KIND':<11} {'SPEECH':>7} {'TRANSCRIPT':<10} CONSENT")
    for p in items:
        sp = p.reference_report.get("speech_s") or p.analysis.get("speech_s") or 0
        click.echo(f"{p.voice_name[:24]:<24} {p.id[:20]:<20} {p.reference_kind:<11} {sp:>6.1f}s "
                   f"{'yes' if p.transcript else 'no':<10} {'yes' if p.consent_recorded else 'NO (attest)'}")


@clone_group.command("inspect")
@click.argument("name")
@click.option("--json", "as_json", is_flag=True)
def clone_inspect(name, as_json):
    from .speakers import load_profile

    try:
        p = load_profile(name)
    except FileNotFoundError as e:
        raise _fail(e) from e
    if as_json:
        _json(p.to_dict())
        return
    d = p.to_dict()
    for k in ("voice_name", "id", "profile_version", "reference_kind", "language", "transcript", "audio_hash",
              "prepared_audio", "original_audio", "publishable", "created"):
        click.echo(f"{k:<16} {d.get(k)}")
    click.echo(f"{'consent':<16} {'recorded' if p.consent_recorded else 'missing'}")
    a = p.analysis
    if a:
        click.echo(f"{'F0 median':<16} {a.get('f0_median_hz') and round(a['f0_median_hz'])} Hz, range "
                   f"{a.get('f0_range_semitones') and round(a['f0_range_semitones'], 1)} st")
        click.echo(f"{'speaking rate':<16} {a.get('speaking_rate_syl_per_s') and round(a['speaking_rate_syl_per_s'], 2)} syl/s (speech only)")
    if p.identity:
        click.echo(f"{'embedding':<16} {p.identity.get('encoder')} ({p.identity.get('use')})")
    if p.preparation:
        click.echo(f"{'preparation':<16} {p.preparation}")
    for i in p.reference_report.get("issues", []):
        click.echo(f"{'issue':<16} {i}")
    if p.migrated_from:
        click.echo(f"{'migrated from':<16} {p.migrated_from}")


@clone_group.command("delete")
@click.argument("name")
@click.option("--yes", is_flag=True)
def clone_delete(name, yes):
    """Delete a profile and its cached embeddings / prepared audio."""
    from .speakers import delete_profile, load_profile

    try:
        p = load_profile(name)
    except FileNotFoundError as e:
        raise _fail(e) from e
    if not yes and not click.confirm(f"Delete {p.voice_name!r} and all its cached data?"):
        raise click.Abort()
    delete_profile(p.id)
    click.echo(f"deleted {p.voice_name}")


@clone_group.command("attest")
@click.argument("name")
@click.option("--i-have-rights", is_flag=True)
def clone_attest(name, i_have_rights):
    """Record consent for a (migrated) profile."""
    from .speakers import ProfileError, attest, load_profile

    try:
        attest(load_profile(name), i_have_rights=i_have_rights)
    except (FileNotFoundError, ProfileError) as e:
        raise _fail(e) from e
    click.echo("consent recorded")


@clone_group.command("migrate")
def clone_migrate():
    """Copy v1 profiles (voices library, experimental voice profiles) into v2. Originals stay as they are."""
    from .speakers import migrate_v1

    done = migrate_v1()
    for src, new in done:
        click.echo(f"{src} -> {new}")
    click.echo(f"migrated {len(done)} profile(s)" + ("; run `aivoice clone attest NAME --i-have-rights` for MeanVC2 voices" if done else ""))


def _options(language, style, style_reference, emotion, speed, duration, seed, steps, guidance, allow_unsupported):
    from .engines.base import SynthesisOptions

    return SynthesisOptions(language=language, style_prompt=style, style_reference=style_reference, emotion=emotion,
                            speed=speed, target_duration_s=duration, seed=seed, steps=steps, guidance=guidance,
                            allow_unsupported=allow_unsupported)


def _synth_options(f):
    for opt in reversed([
        click.option("--engine", default="auto", show_default=True),
        click.option("--priority", type=click.Choice(["balanced", "latency", "similarity", "naturalness", "multilingual", "expressiveness"]),
                     default="balanced", show_default=True),
        click.option("--language", default=None),
        click.option("--style", default=None, help="Natural-language delivery instruction (engines that support it)"),
        click.option("--style-reference", type=click.Path(path_type=Path, exists=True), default=None,
                     help="Separate style recording (only engines with native support; none today)"),
        click.option("--emotion", default=None),
        click.option("--speed", type=float, default=None),
        click.option("--duration", type=float, default=None, help="Target seconds (engines with duration control)"),
        click.option("--seed", type=int, default=None),
        click.option("--steps", type=int, default=None),
        click.option("--guidance", type=float, default=None),
        click.option("--enhance", type=click.Choice(["off", "auto", "high-fidelity"]), default="off", show_default=True),
        click.option("--no-normalize", is_flag=True, help="Skip text normalisation"),
        click.option("--no-duration-plan", is_flag=True),
        click.option("--lexicon", type=click.Path(path_type=Path, exists=True), default=None, help="JSON word->spoken form"),
        click.option("--allow-unsupported", is_flag=True, help="Drop unsupported options with a warning instead of failing"),
    ]):
        f = opt(f)
    return f


@clone_group.command("synthesize")
@click.option("--voice", default=None, help="Clone profile (omit for an engine's default voice)")
@click.option("--text", required=True)
@click.option("-o", "--output", type=click.Path(path_type=Path), required=True)
@_synth_options
@click.option("--consent-ack", is_flag=True)
def clone_synthesize(voice, text, output, engine, priority, language, style, style_reference, emotion, speed, duration,
                     seed, steps, guidance, enhance, no_normalize, no_duration_plan, lexicon, allow_unsupported, consent_ack):
    """Synthesize TEXT in a cloned voice (writes WAV labelled as synthetic + a JSON sidecar)."""
    from .consent import require_consent
    from .fusion.pipeline import FusionConfig, save_result, synthesize
    from .research.phonetics import load_lexicon
    from .speakers import load_profile

    require_consent(ack=consent_ack)
    try:
        prof = load_profile(voice) if voice else None
        opts = _options(language, style, style_reference, emotion, speed, duration, seed, steps, guidance, allow_unsupported)
        cfg = FusionConfig(normalize_text=not no_normalize, lexicon=load_lexicon(lexicon), plan_duration=not no_duration_plan,
                           enhance=enhance)
        res = synthesize(text, prof, opts, engine=engine, priority=priority, config=cfg)
        wav, side = save_result(res, output)
    except Exception as e:  # noqa: BLE001
        raise _fail(e) from e
    click.echo(f"{wav}  ({res.engine}, {res.audio.size / res.sample_rate:.2f}s @ {res.sample_rate} Hz)")
    for r in res.decision.get("reasons", []):
        click.echo(f"  route: {r}")
    for w in res.warnings:
        click.echo(f"  warning: {w}")
    sim = res.metrics.get("similarity_to_reference", {})
    if sim.get("ge2e_cosine") is not None:
        click.echo(f"  speaker similarity to reference: {sim['ge2e_cosine']:.3f} ({sim.get('ge2e_model')})")
    click.echo(f"  details: {side}")


@clone_group.command("preview")
@click.argument("name")
@click.option("--engine", default="auto")
@click.option("--text", default="Hello! This is a short preview of this synthetic voice.")
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None)
@click.option("--consent-ack", is_flag=True)
def clone_preview(name, engine, text, output, consent_ack):
    """Synthesize a short preview into the profile directory."""
    from .consent import require_consent
    from .fusion.pipeline import save_result, synthesize
    from .speakers import load_profile

    require_consent(ack=consent_ack)
    try:
        p = load_profile(name)
        res = synthesize(text, p, engine=engine)
        wav, _ = save_result(res, output or (p.dir() / f"preview-{res.engine}.wav"))
    except Exception as e:  # noqa: BLE001
        raise _fail(e) from e
    click.echo(str(wav))


@clone_group.command("compare")
@click.option("--voice", required=True)
@click.option("--engines", "engine_list", default="installed", help="Comma list, 'installed' or 'all'")
@click.option("--text", required=True)
@click.option("--source", type=click.Path(path_type=Path, exists=True), default=None,
              help="Source speech for voice-conversion engines (default: espeak-ng reads TEXT)")
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--seed", type=int, default=0)
@click.option("--consent-ack", is_flag=True)
def clone_compare(voice, engine_list, text, source, output_dir, seed, consent_ack):
    """Same voice and text through several engines; TTS and VC reported separately."""
    from .audio_io import read_wav, write_wav
    from .consent import require_consent
    from .engines.base import ConversionOptions, SynthesisOptions
    from .engines.registry import all_status, get_engine
    from .evaluation.metrics import acoustic_metrics, speaker_similarity, write_labeled_wav
    from .speakers import load_profile

    require_consent(ack=consent_ack)
    try:
        prof = load_profile(voice)
    except FileNotFoundError as e:
        raise _fail(e) from e
    output_dir.mkdir(parents=True, exist_ok=True)
    ref = prof.speaker_ref()
    ref_x, ref_sr = read_wav(ref.audio)
    statuses = all_status()
    wanted = [s.name for s in statuses] if engine_list in ("all", "installed") else [e.strip() for e in engine_list.split(",")]
    rows = []
    for name in wanted:
        st = next((s for s in statuses if s.name == name), None)
        if st is None:
            rows.append({"engine": name, "status": "unknown engine"})
            continue
        if not st.ready:
            if engine_list != "installed":
                rows.append({"engine": name, "status": f"unavailable: {st.unavailable}"})
            continue
        caps = st.caps
        if not caps.zero_shot:
            continue
        eng = get_engine(name)
        try:
            if "vc" in caps.tasks:
                src = source
                if src is None:
                    s = get_engine("espeak").synthesize(text, None, SynthesisOptions())
                    src = output_dir / "neutral-source.wav"
                    write_wav(src, s.audio, s.sample_rate)
                r = eng.convert(Path(src), ref, ConversionOptions())
                task = "vc (from neutral source)"
            else:
                r = eng.synthesize(text, ref, SynthesisOptions(seed=seed))
                task = "tts"
        except Exception as e:  # noqa: BLE001
            rows.append({"engine": name, "status": f"error: {e}"})
            continue
        out = write_labeled_wav(output_dir / f"{name}.wav", r.audio, r.sample_rate,
                                comment=f"Synthetic speech generated by aivoice ({name}). Not a recording.")
        sim = speaker_similarity(r.audio, r.sample_rate, ref_x, ref_sr)
        rows.append({"engine": name, "task": task, "status": "ok", "file": str(out), "similarity": sim.get("ge2e_cosine"),
                     "rtf": r.rtf, "first_audio_ms": r.first_audio_ms, **acoustic_metrics(r.audio, r.sample_rate)})
    (output_dir / "summary.json").write_text(json.dumps({"voice": prof.id, "text": text, "rows": rows}, indent=2, default=str))
    if not any(r.get("status") == "ok" for r in rows):
        click.echo("no engine produced audio:")
    for r in rows:
        if r.get("status") == "ok":
            click.echo(f"{r['engine']:<12} {r['task']:<26} similarity {r['similarity']}  rtf {r['rtf'] and round(r['rtf'], 3)}  {r['file']}")
        else:
            click.echo(f"{r['engine']:<12} {r['status']}")
    click.echo(f"summary: {output_dir / 'summary.json'}")


@clone_group.command("adapt")
@click.option("--voice", required=True)
@click.option("--dataset", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True)
@click.option("--method", type=click.Choice(["auto", "voxcpm2-lora"]), default="auto")
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--i-have-rights", is_flag=True)
def clone_adapt(voice, dataset, method, output, i_have_rights):
    """Validate an authorized dataset and prepare VoxCPM2 LoRA fine-tuning (training is run by you, explicitly)."""
    from .research.adaptation import validate_dataset, write_voxcpm_manifests
    from .research.speaker_identity import get_encoder
    from .speakers import load_profile

    if not i_have_rights:
        raise click.ClickException("speaker adaptation trains on someone's voice: re-run with --i-have-rights if authorized")
    try:
        p = load_profile(voice)
    except FileNotFoundError as e:
        raise _fail(e) from e
    rep = validate_dataset(dataset, encoder=get_encoder())
    click.echo(f"{rep.usable}/{rep.clips} usable clips, {rep.total_minutes} min; train {len(rep.train)} / val {len(rep.val)}")
    for path, issues in list(rep.issues.items())[:20]:
        click.echo(f"  {Path(path).name}: {'; '.join(issues)}")
    for n in rep.notes:
        click.echo(f"  note: {n}")
    if rep.duplicates:
        click.echo(f"  duplicates: {len(rep.duplicates)} group(s) (extra copies excluded)")
    paths = write_voxcpm_manifests(rep, output, ref_audio=str(p.conditioning_audio()) if p.conditioning_audio() else None)
    click.echo(f"manifests: {paths['train']}, {paths['val']}")
    click.echo(f"next step (manual): see {Path(output) / 'README.txt'}")
    if not rep.ready:
        raise click.ClickException("dataset not ready for adaptation (see notes); zero-shot cloning works without it")


# =============================================================================
# engines
# =============================================================================


@click.group("engines")
def engines_group():
    """Synthesis / conversion engines and what each supports."""


@engines_group.command("list")
@click.option("--json", "as_json", is_flag=True)
def engines_list(as_json):
    from .engines.registry import all_status

    st = all_status()
    if as_json:
        _json([{"name": s.name, "ready": s.ready, "unavailable": s.unavailable, **s.caps.to_dict()} for s in st])
        return
    for s in st:
        c = s.caps
        click.echo(f"{s.name:<12} {'/'.join(c.tasks):<4} {'ready' if s.ready else 'not ready':<10} {c.display_name}")
        if not s.ready:
            click.echo(f"{'':<12} -> {s.unavailable}")


@engines_group.command("capabilities")
def engines_capabilities():
    from .engines.registry import all_status, capability_table

    click.echo(capability_table(all_status()))


@engines_group.command("info")
@click.argument("name")
def engines_info(name):
    from .engines.registry import status

    try:
        s = status(name)
    except Exception as e:  # noqa: BLE001
        raise _fail(e) from e
    for k, v in s.caps.to_dict().items():
        if v not in (None, "", (), [], {}):
            click.echo(f"{k:<26} {v}")
    click.echo(f"{'ready':<26} {s.ready}" + ("" if s.ready else f"  ({s.unavailable})"))


@engines_group.command("benchmark")
@click.option("--engines", "engine_list", default=None, help="Comma list (default: all)")
@click.option("--dataset", type=click.Path(path_type=Path, exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(path_type=Path), default=Path("benchmark-results"), show_default=True)
@click.option("--consent-ack", is_flag=True)
@click.pass_context
def engines_benchmark(ctx, engine_list, dataset, output, consent_ack):
    """Alias for `aivoice benchmark research`."""
    ctx.invoke(benchmark_research, engine_list=engine_list or "all", dataset=dataset, synthetic=dataset is None,
               output=output, seeds="0,1", device=None, no_prepare=False, consent_ack=consent_ack)


# =============================================================================
# research
# =============================================================================


@click.group("research")
def research_group():
    """Papers behind aivoice's techniques, their status, and ablation experiments."""


@research_group.command("list")
@click.option("--json", "as_json", is_flag=True)
def research_list(as_json):
    from .research.catalog import PAPERS, as_dicts

    if as_json:
        _json(as_dicts())
        return
    for p in PAPERS:
        click.echo(f"{p.key:<20} {p.status:<21} {p.title[:70]}")
        click.echo(f"{'':<20} {p.in_aivoice[:150]}")


@research_group.command("citations")
def research_citations():
    from .research.catalog import bibtex

    click.echo(bibtex())


@research_group.command("experiments")
@click.option("--run", "run_ids", default=None,
              help="Comma list of experiment ids to run now (C,F,G synthetic; D,E,H need engines + --dataset); default: show results")
@click.option("--output", type=click.Path(path_type=Path), default=None, help="Default: docs/research/experiments if present")
@click.option("--dataset", type=click.Path(path_type=Path, exists=True, file_okay=False), default=None,
              help="Authorized speakers (benchmark layout) for E and H")
@click.option("--results", "results_json", type=click.Path(path_type=Path), default=None,
              help="results.json of `benchmark research` for D")
@click.option("--engines", "engine_list", default=None, help="Engines for H (default: installed voxcpm2, xvoice, voicestar)")
@click.option("--audio-out", type=click.Path(path_type=Path), default=None, help="Save example outputs of E/H here (labelled WAV)")
def research_experiments(run_ids, output, dataset, results_json, engine_list, audio_out):
    from .evaluation import experiments as ex

    out = output or (Path("docs/research/experiments") if Path("docs/research").is_dir() else Path("research-experiments"))
    if run_ids:
        ids = [i.strip().upper() for i in run_ids.split(",")]
        engine_ids = [i for i in ids if i in ("D", "E", "H") and (dataset is not None or results_json is not None)]
        results = ex.run([i for i in ids if i not in engine_ids], out) if [i for i in ids if i not in engine_ids] else []
        if engine_ids:
            from .evaluation.engine_experiments import run_engine_experiments

            results += run_engine_experiments(engine_ids, dataset=dataset, results_json=results_json, out_dir=out,
                                              engines=[e.strip() for e in engine_list.split(",")] if engine_list else None,
                                              audio_out=audio_out)
        for r in results:
            click.echo(f"{r.id} {r.status}: {r.decision}")
        click.echo(f"written to {out}")
        return
    if not out.is_dir():
        click.echo("no results yet: aivoice research experiments --run C,F,G")
        return
    for f in sorted(out.glob("*.json")):
        d = json.loads(f.read_text())
        click.echo(f"{d['id']:<3} {d['status']:<9} {d['title']}")
        click.echo(f"    {d['decision']}")


# =============================================================================
# benchmark research (attached to the existing `benchmark` group in cli.py)
# =============================================================================


@click.command("research")
@click.option("--engines", "engine_list", default="all", show_default=True)
@click.option("--dataset", type=click.Path(path_type=Path, exists=True, file_okay=False), default=None)
@click.option("--synthetic", is_flag=True, help="Use rights-free espeak-ng speakers when no dataset is given")
@click.option("--output", type=click.Path(path_type=Path), default=Path("benchmark-results"), show_default=True)
@click.option("--seeds", default="0,1", show_default=True)
@click.option("--device", default=None)
@click.option("--no-prepare", is_flag=True, help="Ablation: raw references instead of prepared ones")
@click.option("--save-audio", is_flag=True, help="Keep every output as a labelled WAV under OUTPUT/audio/")
@click.option("--consent-ack", is_flag=True)
def benchmark_research(engine_list, dataset, synthetic, output, seeds, device, no_prepare, save_audio, consent_ack):
    """Every engine on the same references/texts/seeds: similarity, WER, RTF, first audio, RAM, VRAM (JSON + CSV)."""
    from .consent import require_consent
    from .evaluation.benchmark import run_research_benchmark, synthetic_dataset

    require_consent(ack=consent_ack)
    if dataset is None:
        if not synthetic:
            raise click.ClickException("pass --dataset DIR (authorized speakers) or --synthetic")
        dataset = synthetic_dataset(Path(output) / "_synthetic_dataset")
    names_ = None if engine_list in ("all", "installed") else [e.strip() for e in engine_list.split(",")]
    res = run_research_benchmark(names_, dataset, Path(output), seeds=tuple(int(s) for s in seeds.split(",")),
                                 prepare=not no_prepare, device=device, save_audio=save_audio)
    for r in res["rows"]:
        if r.get("status") == "summary":
            click.echo(f"{r['engine']:<12} similarity {r.get('speaker_similarity')}  wer {r.get('wer')}  "
                       f"first-audio {r.get('first_audio_ms')} ms  rtf {r.get('rtf')}  VRAM {r.get('peak_vram_mib')} MiB  "
                       f"RSS {r.get('peak_rss_mib')} MiB")
        elif r.get("status", "").startswith(("unavailable", "not a cloning", "load failed")):
            click.echo(f"{r['engine']:<12} {r['status']}")
    for s in res["meta"]["skipped_speakers"]:
        click.echo(f"skipped speaker {s}")
    click.echo(f"results: {Path(output) / 'results.json'} , {Path(output) / 'results.csv'} , {Path(output) / 'report.md'}")

