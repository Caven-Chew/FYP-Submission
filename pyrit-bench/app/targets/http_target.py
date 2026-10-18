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

    async def send_prompt_async(self, *, message) -> list:
        import json as _json
        import httpx
        prompt = message.get_value() if hasattr(message, "get_value") else str(message)

        # Substitute {prompt} via a sentinel so a prompt that literally contains
        # "{prompt}" can't double-substitute; the value is JSON-escaped.
        _SENTINEL = "\x00PROMPT_PLACEHOLDER\x00"
        body = self.body_template.replace("{prompt}", _SENTINEL)
        body = body.replace(_SENTINEL, _json.dumps(prompt)[1:-1])
        if self.api_key:
            # JSON-escape the key too, so a key containing a quote or backslash
            # can't produce invalid JSON.
            body = body.replace("{api_key}", _json.dumps(self.api_key)[1:-1])
        _json.loads(body)  # validate JSON before sending

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
        return wrap_response(str(data))
