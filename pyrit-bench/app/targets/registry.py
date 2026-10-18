"""The provider registry — the single place a new provider is wired in."""
from __future__ import annotations

import os
from typing import Any

from .api_providers import (
    AnthropicProvider, AzureOpenAIProvider, HttpGenericProvider,
    HuggingFaceProvider, OllamaProvider, OpenAIProvider
)
from .base import BaseProvider
from .common import ensure_pyrit_initialized, extra_of
from .throttle import install_throttle
from .ui_providers import BrowserUIProvider


# ---------- registry + public API ----------------------------------------

# Order here is the order shown in the form (most common first within each type).
PROVIDERS: dict[str, BaseProvider] = {
    p.slug: p for p in (
        HttpGenericProvider(),   # Custom first — the default starting point
        OpenAIProvider(),
        OllamaProvider(),
        AnthropicProvider(),
        AzureOpenAIProvider(),
        HuggingFaceProvider(),
        BrowserUIProvider(),
    )
}


def get_provider(slug: str) -> BaseProvider:
    """Look up a provider by slug. Raises ValueError on an unknown slug."""
    p = PROVIDERS.get((slug or "").lower())
    if p is None:
        raise ValueError(f"Unknown provider: {slug!r}")
    return p


def build_target(t: Any, multi_turn: bool = False) -> Any:
    """Return a PyRIT-compatible chat target for a stored Target row.

    Set `multi_turn=True` for Crescendo / RedTeaming / TAP / PAIR: those need a
    real PromptChatTarget, not the generic HTTP shim.
    """
    ensure_pyrit_initialized()
    provider = get_provider(t.provider)
    # Prefer a key stored directly on the target; fall back to an env-var name
    # (back-compat with targets created before keys were stored per-target).
    api_key = getattr(t, "api_key", None) or (
        os.getenv(t.api_key_env_var) if getattr(t, "api_key_env_var", None) else None
    )
    target = provider.build_target(t, api_key, multi_turn=multi_turn)
    # Optional per-target burst pause (pause_after_n / pause_rest_s in the
    # target's extra config). A no-op unless both are set, so this is the single
    # funnel that covers static, multi-turn and the SDK/HTTP targets alike —
    # every one of them ends up calling send_prompt_async on this object.
    return install_throttle(target, extra_of(t))


def providers_for_form() -> dict:
    """Group provider form-metadata by top-level type for the Add-Target page.

    Returns {"api": [meta, ...], "ui": [meta, ...]} preserving registry order.
    """
    out: dict[str, list] = {"api": [], "ui": []}
    for p in PROVIDERS.values():
        out.setdefault(p.type, []).append(p.form_meta())
    return out
