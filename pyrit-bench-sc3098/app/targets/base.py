"""Provider base classes — the two top-level connection types (API and UI)."""
from __future__ import annotations

from typing import Any

from .common import extra_of
from .http_target import HttpTarget


# ---------- provider class hierarchy -------------------------------------

class BaseProvider:
    """Base class for all providers. `type` is "api" or "ui"."""

    slug: str = ""
    label: str = ""
    type: str = ""
    help: str = ""

    def build_target(self, t: Any, api_key: str | None, multi_turn: bool = False) -> Any:
        """Build a target for `t`.

        `multi_turn=True` asks for a target usable by PyRIT's multi-turn attack
        strategies (Crescendo, RedTeaming, TAP, PAIR). Those introspect the
        target rather than just calling it, so they need a real
        `PromptChatTarget` — the raw HTTP shim is not one.
        """
        raise NotImplementedError

    @classmethod
    def form_meta(cls) -> dict:
        return {"slug": cls.slug, "label": cls.label, "type": cls.type, "help": cls.help}


class APIProvider(BaseProvider):
    """HTTP-reached target. The user pastes the JSON request body; we send it.

    `request_template` pre-fills the Add-Target form (model, endpoint, body,
    response_path, headers, api_key_env_var). `raw=True` means the form shows the
    JSON-body fields; HuggingFace sets it False (in-process, no HTTP body).
    """

    type = "api"
    raw = True
    request_template: dict = {}

    def build_target(self, t: Any, api_key: str | None, multi_turn: bool = False) -> Any:
        extra = extra_of(t)
        # Multi-turn strategies need a real PromptChatTarget: PyRIT reads
        # `configuration`, `get_identifier()` and the `supports_editable_history`
        # capability off the target. `HttpTarget` is a duck-typed shim that has
        # none of those, so prefer the SDK target whenever one exists — even if
        # the Add-Target form filled in a JSON body.
        if multi_turn:
            return self._build_sdk(t, api_key)
        if extra.get("body_template"):
            return HttpTarget(t.endpoint_url, api_key, extra)
        return self._build_sdk(t, api_key)

    def _build_sdk(self, t: Any, api_key: str | None) -> Any:
        # Reached for older targets saved without a JSON body, and for any
        # multi-turn run against a provider that has no SDK target class.
        raise ValueError(
            f"{self.label}: no SDK chat target is available for this provider. "
            "Multi-turn attacks (Crescendo, Red Teaming, TAP, PAIR) need one — "
            "use an OpenAI, Ollama, Anthropic or Azure OpenAI target instead. "
            "For single-turn attacks, enter a Request body (JSON) containing a "
            "{prompt} placeholder."
        )

    @classmethod
    def form_meta(cls) -> dict:
        m = super().form_meta()
        m["raw"] = cls.raw
        m["request_template"] = cls.request_template
        return m


class UIProvider(BaseProvider):
    """Target reached only through a rendered web UI (driven with Playwright)."""
    type = "ui"
