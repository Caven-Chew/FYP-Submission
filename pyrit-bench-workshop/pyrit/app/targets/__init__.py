"""Targets — how the tool connects to the system under test.

Two connection types:

  API : an endpoint you call (OpenAI, Anthropic, Azure OpenAI, Ollama,
        HuggingFace, or any custom JSON endpoint)
  UI  : a real chat web page, driven through a browser

Layout of this package:

  common.py        PyRIT memory bootstrap + small shared shims
  base.py          BaseProvider / APIProvider / UIProvider
  api_providers.py one class per API provider
  ui_providers.py  browser-driven providers
  http_target.py   the generic "post this JSON to this URL" target
  registry.py      PROVIDERS -- the one place a new provider is wired in
  browser/         everything Playwright: the target, the interactive capture
                   helpers, the injected JavaScript, screenshot diagnosis

To add a provider: write the class in api_providers.py (or ui_providers.py)
and add one line to PROVIDERS in registry.py. The Add-Target form follows
automatically via providers_for_form().
"""
from __future__ import annotations

from .base import APIProvider, BaseProvider, UIProvider
from .browser import (BrowserUITarget, capture_login_session,
                      capture_ui_selectors, engine_available, install_hint)
from .common import ensure_pyrit_initialized, extra_of, wrap_response
from .http_target import HttpTarget
from .registry import PROVIDERS, build_target, get_provider, providers_for_form

__all__ = [
    "APIProvider", "BaseProvider", "UIProvider",
    "BrowserUITarget", "HttpTarget",
    "PROVIDERS", "build_target", "get_provider", "providers_for_form",
    "capture_login_session", "capture_ui_selectors",
    "engine_available", "install_hint",
    "ensure_pyrit_initialized", "extra_of", "wrap_response",
]
