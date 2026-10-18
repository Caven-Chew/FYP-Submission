"""Read the Checklist D packs as data.

The Checklist D attacks were removed from the app on 3 October 2026 (Task 11.41, user
decision): students now get the example prompts in the briefing slides and the user guide
instead of launch buttons in the app. The packs themselves stay in
``app/data/gdp_checklist.json`` and are still needed by the test suites, the maintainer
harnesses and the tool that writes the guide's example-prompt tables, so the reading
helpers that used to live in ``app/web/views_checklist.py`` live here.

This module holds no routes and no Flask imports: it reads the two JSON files and nothing
else. Everything in the packs is invented test data; nothing comes from Appendix B.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "app" / "data"


def checklist_items() -> list[dict]:
    """The 13 Checklist D packs, in file order."""
    return json.loads((DATA / "gdp_checklist.json").read_text(encoding="utf-8"))


def checklist_notes() -> dict:
    """Shared notes that used to head the Checklist D section ({} if unavailable)."""
    try:
        return json.loads((DATA / "gdp_checklist_notes.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def item_by_id(item_id: str) -> dict | None:
    return next((item for item in checklist_items() if item["id"] == item_id), None)


def item_documents(item: dict) -> list:
    """Keep the legacy fixture first, so an existing document index still means the same file."""
    return ([item["document"]] if item.get("document") else []) + (item.get("documents") or [])


def planted_data(item: dict) -> str | None:
    """The quoted line or record holding this item's marker, for the "set this up first" step."""
    marker = item.get("marker")
    if not marker:
        return None
    for text in (item.get("before_you_launch") or []) + [item.get("reference_facts") or ""]:
        if found := re.search(r"'([^']*" + re.escape(marker) + r"[^']*)'", text):
            return found.group(1)
        if found := re.search(r"([A-Z][\w ]*?)\s*\(([^()]*" + re.escape(marker) + r"[^()]*)\)", text):
            return f'{found.group(1).strip()}: {found.group(2).strip()}'
    return marker
