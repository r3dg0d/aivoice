"""Voice catalog provider abstraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ProviderCapabilities:
    search: bool = False
    download: bool = False
    versions: bool = False
    details: bool = False
    offline: bool = False
    notes: str = ""


@dataclass
class VoiceModelRef:
    """A catalog entry from a provider. Only populate fields the provider supplies."""

    provider: str
    id: str
    name: str
    page_url: str | None = None
    download_url: str | None = None
    architecture: str | None = None  # e.g. "RVC v2"
    size_label: str | None = None
    author: str | None = None
    tags: list[str] = field(default_factory=list)
    sample_rate: str | None = None
    license: str | None = None
    downloads: int | None = None
    rating: float | None = None
    updated: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "provider": self.provider,
            "id": self.id,
            "name": self.name,
        }
        for k in (
            "page_url",
            "download_url",
            "architecture",
            "size_label",
            "author",
            "sample_rate",
            "license",
            "downloads",
            "rating",
            "updated",
        ):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.tags:
            d["tags"] = list(self.tags)
        if self.extra:
            d["extra"] = dict(self.extra)
        return d


@dataclass
class SearchPage:
    query: str
    page: int
    results: list[VoiceModelRef]
    has_more: bool = False
    raw_note: str = ""


class VoiceProvider(Protocol):
    name: str

    def capabilities(self) -> ProviderCapabilities: ...

    def search(self, query: str, *, page: int = 1) -> SearchPage: ...

    def get_model(self, model_id: str) -> VoiceModelRef: ...

    def download_url_for(self, model: VoiceModelRef) -> str | None: ...
