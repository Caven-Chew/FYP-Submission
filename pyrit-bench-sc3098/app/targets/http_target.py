"""Generic HTTP chat target — posts a user-defined JSON body to any endpoint."""
from __future__ import annotations


from .common import wrap_response


# ---------- the one HTTP target (used by every API provider) -------------

class HttpTarget:
    """Generic HTTP target: sends a user-defined JSON body to an endpoint.

    extra_config:
      body_template : JSON string sent as the request body. Use ``{prompt}`` where
                      the user message goes (it is JSON-escaped on substitution),
                      and ``{api_key}`` anywhere the resolved API key should appear.
      response_path : dot-path into the response JSON (supports int list indices),
                      e.g. ``choices.0.message.content``.
      header.<Name> : request headers. Values may contain ``{api_key}``.

    Auth: if the body/headers reference ``{api_key}`` the user is wiring auth in
    themselves. Otherwise, when an api_key is available, a default
    ``Authorization: Bearer <key>`` header is added (the OpenAI convention).
    """

    def __init__(self, endpoint: str | None, api_key: str | None, extra: dict):
        if not endpoint:
            raise ValueError("API target requires an Endpoint URL.")
        self.endpoint = endpoint
        self.api_key = api_key
        self.body_template = extra.get("body_template", '{"prompt": "{prompt}"}')
        self.response_path = extra.get("response_path", "response")
        self.headers = {k[7:]: v for k, v in extra.items() if k.startswith("header.")}
        self._uses_key_placeholder = (
            "{api_key}" in self.body_template
            or any("{api_key}" in v for v in self.headers.values())
        )

    def get_identifier(self):
        try:
            from pyrit.models import ComponentIdentifier  # type: ignore  # PyRIT >=0.13
        except ImportError:  # older PyRIT (pyrit.identifiers is a deprecated shim in 0.14)
            from pyrit.identifiers import ComponentIdentifier  # type: ignore
        return ComponentIdentifier.of(self, params={"endpoint": self.endpoint})

    def _build_identifier(self):
        return self.get_identifier()

    @property
    def capabilities(self):
        from pyrit.prompt_target.common.target_capabilities import TargetCapabilities  # type: ignore
        return TargetCapabilities()

    @property
    def configuration(self):
        # PyRIT 0.14's executor.attack module (RedTeamingAttack, CrescendoAttack,
        # ConversationManager) reads target.configuration.includes(...) /
        # .ensure_can_handle(...) instead of the older direct `.capabilities`
        # check — a duck-typed target (not a PromptTarget subclass) needs this
        # wrapper or it raises AttributeError as soon as a dynamic attack runs.
        from pyrit.prompt_target.common.target_configuration import TargetConfiguration  # type: ignore
        return TargetConfiguration(capabilities=self.capabilities)

    def render_body(self, prompt: str, history: list[dict] | None = None) -> str:
        """Build the saved JSON body for one prompt, optionally with earlier turns.

        ``history`` is a list of ``{"role": "user"|"assistant", "content": str}``.
        Earlier turns are carried in one of two ways, so a scripted conversation
        uses exactly the request the connection test used (same system message,
        headers and response path):

        * a ``{history}`` placeholder anywhere in the body (quoted or not) is
          replaced by the JSON array of earlier turns — for custom JSON APIs such
          as ``{"message": "{prompt}", "history": {history}}``;
        * otherwise, if the body has a ``messages`` array holding the ``{prompt}``
          message (OpenAI, Anthropic, Ollama, Azure shapes), the earlier turns are
          inserted just before that message, after any system message.
        """
        import json as _json
        import re as _re
        turns = [{"role": t["role"], "content": t["content"]} for t in (history or [])]
        array = _json.dumps(turns, ensure_ascii=False)

        # All placeholders are replaced in ONE pass over the saved template, so
        # inserted text is never scanned again: a prompt that literally contains
        # "{api_key}" or "{history}" stays literal and cannot pull in the key or
        # the history (review finding #47). Prompt and key are JSON-escaped.
        def _fill(m):
            name = m.group(1) or "history"
            if name == "prompt":
                return _json.dumps(prompt)[1:-1]
            if name == "api_key":
                return _json.dumps(self.api_key)[1:-1] if self.api_key else m.group(0)
            return array
        body = _re.sub(r'"\{history\}"|\{(prompt|api_key|history)\}', _fill, self.body_template)
        if "{history}" in self.body_template:
            _json.loads(body)  # validate JSON before sending
            return body
        data = _json.loads(body)  # validate JSON before sending
        if not turns:
            return body
        messages = data.get("messages") if isinstance(data, dict) else None
        if isinstance(messages, list):
            for index, item in enumerate(messages):
                if isinstance(item, dict) and item.get("content") == prompt:
                    messages[index:index] = turns
                    return _json.dumps(data, ensure_ascii=False)
        raise ValueError(
            "This target's request body cannot carry earlier turns, so a multi-turn "
            "conversation would lose its history. Use a body with a \"messages\" array "
            "containing the {prompt} message, or add a {history} placeholder where your "
            "API expects the earlier turns (for example \"history\": {history})."
        )

    def supports_history(self) -> bool:
        try:
            self.render_body("probe", [{"role": "user", "content": "a"},
                                       {"role": "assistant", "content": "b"}])
            return True
        except (ValueError, KeyError, TypeError):
            return False

    async def _post(self, body: str) -> str:
        import httpx
        h = {"Content-Type": "application/json"}
        for k, v in self.headers.items():
            h[k] = v.replace("{api_key}", self.api_key) if (self.api_key and "{api_key}" in v) else v
        if self.api_key and not self._uses_key_placeholder and "Authorization" not in h:
            h["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=120, trust_env=False, mounts={}) as c:
            r = await c.post(self.endpoint, content=body.encode(), headers=h)
            r.raise_for_status()
            data = r.json()
        for key in self.response_path.split("."):
            if not key:
                continue
            data = data[int(key)] if isinstance(data, list) else data[key]
        return str(data)

    async def send_turn_async(self, prompt: str, history: list[dict]) -> str:
        """Send one conversation turn with the earlier turns; return the reply text.

        Routed through ``send_prompt_async`` so a per-target throttle, which wraps
        that method, also paces conversation turns.
        """
        self._history = list(history)
        try:
            pieces = await self.send_prompt_async(message=prompt)
        finally:
            self._history = None
        return pieces[0].get_value() if pieces else ""

    async def send_prompt_async(self, *, message) -> list:
        prompt = message.get_value() if hasattr(message, "get_value") else str(message)
        history = getattr(self, "_history", None)
        return wrap_response(await self._post(self.render_body(prompt, history)))
