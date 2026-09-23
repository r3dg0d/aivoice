"""Interactive numbered / simple TTY selector for search results."""

from __future__ import annotations

import sys

from .providers.base import VoiceModelRef


def format_results_table(results: list[VoiceModelRef]) -> str:
    if not results:
        return "(no results)"
    lines = [f"  {'#':<3} {'Name':<48} {'Type':<10} {'Size':<10}"]
    for i, r in enumerate(results, 1):
        name = (r.name[:45] + "…") if len(r.name) > 46 else r.name
        arch = r.architecture or "-"
        size = r.size_label or "-"
        lines.append(f"  {i:<3} {name:<48} {arch:<10} {size:<10}")
    return "\n".join(lines)


def select_result(results: list[VoiceModelRef], *, prompt: str = "Select a model") -> VoiceModelRef | None:
    if not results:
        return None
    print(format_results_table(results))
    print()
    if not sys.stdin.isatty():
        raise RuntimeError("interactive selection requires a TTY; pass --index N for non-interactive use")
    while True:
        try:
            raw = input(f"{prompt} [1-{len(results)}] (Enter=1, Esc/q=cancel): ").strip()
        except EOFError:
            return None
        if raw.lower() in {"q", "quit", "esc", "cancel"}:
            return None
        if raw == "":
            return results[0]
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(results):
                return results[idx - 1]
        print(f"Enter a number 1–{len(results)}, or q to cancel.")


def select_installed_voice(names: list[str]) -> str | None:
    if not names:
        return None
    print("Choose voice\n")
    for i, n in enumerate(names, 1):
        mark = ">" if i == 1 else " "
        print(f"{mark} {i}. {n}")
    print("\n  s. Search online")
    print("  i. Import RVC")
    print("  r. Import reference audio")
    if not sys.stdin.isatty():
        return names[0]
    raw = input("Select [1/s/i/r]: ").strip().lower()
    if raw in {"", "1"}:
        return names[0]
    if raw.isdigit() and 1 <= int(raw) <= len(names):
        return names[int(raw) - 1]
    if raw == "s":
        return "__search__"
    if raw == "i":
        return "__import_rvc__"
    if raw == "r":
        return "__import_ref__"
    return None
