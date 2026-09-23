"""High-level SEARCH → SELECT → DOWNLOAD → ADAPT → USE workflow."""

from __future__ import annotations

from pathlib import Path

from .adapt import PERSON_NOTICE, adapt_rvc_to_meanvc2, cached_voice_for_search
from .download import DownloadError, download_cached, safe_extract
from .providers import get_provider
from .providers.base import VoiceModelRef
from .selector import format_results_table, select_result
from .voices import VoiceEntry, load_voice


def search_voices(query: str, *, provider: str = "voice-models", page: int = 1):
    prov = get_provider(provider)
    return prov.search(query, page=page)


def install_from_ref(
    model: VoiceModelRef,
    *,
    display_name: str | None = None,
    reference: Path | None = None,
    progress: bool = True,
    consent_person_ack: bool = False,
) -> VoiceEntry:
    existing = cached_voice_for_search(model.provider, model.id)
    if existing and existing.status == "ready":
        return existing

    if not consent_person_ack:
        print(PERSON_NOTICE)

    url = model.download_url
    if not url:
        raise RuntimeError(
            "No direct download URL for this model. "
            "Download the archive manually, then: aivoice voices import-rvc <file.zip>"
        )
    if "drive.google.com" in url or "mega.nz" in url:
        raise RuntimeError(
            f"Download host requires manual steps ({url}). "
            "Save the file locally, then: aivoice voices import-rvc <file>"
        )

    print("Downloading...")
    try:
        archive = download_cached(
            url,
            provider=model.provider,
            model_id=model.id,
            progress=progress,
        )
    except DownloadError as e:
        raise RuntimeError(str(e)) from e

    extract_dir = archive.parent / "extracted"
    print("Extracting...")
    safe_extract(archive, extract_dir)

    print("Validating RVC package...")
    print("Preparing MeanVC2 voice (reference-audio adaptation)...")
    result = adapt_rvc_to_meanvc2(
        extract_dir,
        display_name=display_name or model.name,
        reference_override=reference,
        source_provider=model.provider,
        source_model_id=model.id,
        source_url=model.page_url or url,
    )
    if result.voice is None:
        raise RuntimeError(result.message)
    if result.engine != "meanvc2":
        print(result.message)
    return result.voice


def interactive_search_install(
    query: str,
    *,
    provider: str = "voice-models",
    index: int | None = None,
    reference: Path | None = None,
) -> VoiceEntry:
    print(f'Searching {provider} for:\n"{query}"\n')
    page = search_voices(query, provider=provider)
    if not page.results:
        raise RuntimeError("No models found.")
    print(f"Found {len(page.results)} models" + (" (more pages available)" if page.has_more else ""))
    print()
    if index is not None:
        if index < 1 or index > len(page.results):
            raise RuntimeError(f"--index out of range 1..{len(page.results)}")
        model = page.results[index - 1]
        print(format_results_table(page.results))
        print(f"\nUsing #{index}: {model.name}")
    else:
        model = select_result(page.results)
        if model is None:
            raise RuntimeError("cancelled")
    voice = install_from_ref(model, reference=reference, consent_person_ack=False)
    print(f"\nSaved:\n{voice.display_name}  ({voice.engine}, {voice.status})")
    return voice


def resolve_voice_arg(voice: str | None) -> VoiceEntry | None:
    if not voice:
        return None
    return load_voice(voice)
