"""Judge / attacker / converter LLMs, and the per-run override plumbing."""
from __future__ import annotations

import contextvars
import os
from typing import Any
from ..models import Benchmark
from ..targets import ensure_pyrit_initialized


# ---------- judge + attacker LLMs from env --------------------------------

# Provider-appropriate defaults so JUDGE_MODEL / ATTACKER_MODEL (and the API-key
# env var) can be left blank without silently inheriting an OpenAI value.
# Previously {ROLE}_PROVIDER=anthropic with no model fell back to "gpt-4o-mini"
# (and to OPENAI_API_KEY), which Anthropic rejects.
_ROLE_DEFAULT_MODEL = {
    "openai": "gpt-4o-mini", "anthropic": "claude-3-5-haiku-latest",
    "venice": "qwen-3-6-plus",
    # University HPC endpoint supplied by the supervisor. It exposes an
    # OpenAI-compatible API, so PyRIT can use it without LangChain or a custom
    # transport. Keeping it as its own provider makes the free workshop path
    # explicit instead of pretending it is api.openai.com.
    "prof_llm": "qwen3.6-35b-a3b-fp8",
}


_ROLE_DEFAULT_KEY_ENV = {
    "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
    "venice": "VENICE_API_KEY",
    "prof_llm": "LLM_API_KEY",
}

# Providers reached through PyRIT's OpenAIChatTarget against an OpenAI-compatible
# /chat/completions endpoint. Venice is included so an uncensored attacker/
# converter (e.g. qwen-3-6-plus) can stand in for the OpenAI models that get
# moderation-blocked on harmful seeds — only the base URL and key differ.
_OPENAI_COMPAT_ENDPOINT = {
    "openai": "https://api.openai.com/v1",
    "venice": "https://api.venice.ai/api/v1",
}


def _normalise_openai_endpoint(endpoint: str) -> str:
    """Return an OpenAI-compatible *base* endpoint ending in ``/v1``.

    The HPC LLM connection note says the supplied URL usually ends in /v1,
    but operators sometimes paste the full /chat/completions route. PyRIT's
    OpenAIChatTarget wants the base URL and appends the route itself; accepting
    either form here avoids a workshop-day 404.
    """
    endpoint = (endpoint or "").strip().rstrip("/")
    for suffix in ("/v1/chat/completions", "/chat/completions"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)].rstrip("/")
            break
    if endpoint and not endpoint.endswith("/v1"):
        endpoint += "/v1"
    return endpoint


def _role_endpoint(role: str, provider: str) -> str:
    """Resolve the endpoint for one OpenAI-compatible role provider."""
    override = role_setting(role, "ENDPOINT")
    if provider == "prof_llm":
        endpoint = override or os.getenv("LLM_ENDPOINT", "")
        endpoint = _normalise_openai_endpoint(endpoint)
        if not endpoint:
            raise ValueError(
                "HPC LLM is selected but LLM_ENDPOINT is not configured. "
                "Set LLM_ENDPOINT in the project-root .env file (usually the "
                "supervisor-provided URL ending in /v1)."
            )
        return endpoint
    return _normalise_openai_endpoint(
        override or _OPENAI_COMPAT_ENDPOINT[provider]
    )


def _openai_target_options(provider: str) -> dict:
    """Provider-specific options passed to PyRIT's OpenAIChatTarget.

    The shared HPC queue regularly takes longer than the public-client default,
    especially for the 35B judge model. A longer timeout is cheap here (the
    service is free) and prevents a healthy queued request being recorded as a
    judge failure. Paid providers keep their existing defaults.
    """
    if provider != "prof_llm":
        return {}
    try:
        timeout = float(os.getenv("PROF_LLM_TIMEOUT", "120"))
    except ValueError:
        timeout = 120.0
    return {"httpx_client_kwargs": {"timeout": max(10.0, timeout)}}

# Per-benchmark LLM overrides. A benchmark runs in its own thread with its own
# asyncio loop, and a ContextVar is set/reset for exactly that run, so two
# benchmarks with different attacker/judge models never see each other's config.
# When empty (the default) everything falls back to the {ROLE}_* env vars, so
# existing .env-only setups behave exactly as before.
llm_overrides: "contextvars.ContextVar[dict]" = contextvars.ContextVar(
    "llm_overrides", default={}
)


def role_setting(role: str, key: str) -> str | None:
    """Resolve one {ROLE}_{KEY} setting: per-benchmark override wins over env."""
    ov = llm_overrides.get().get(role) or {}
    val = ov.get(key.lower())
    if val is not None and str(val).strip():
        return str(val).strip()
    return os.getenv(f"{role}_{key}")


def build_role_llm(role: str) -> Any:
    """Build a judge/attacker chat target for `role` ("JUDGE" or "ATTACKER").

    Configuration is resolved per setting: a per-benchmark override (set in the
    UI, carried on the ContextVar) takes precedence, otherwise the {ROLE}_*
    env vars are used. Returns None when no provider is configured either way.
    The model name and API-key env var default per-provider, so configuring only
    the provider yields a working target instead of passing an OpenAI model name
    (or key) to Anthropic.

    A per-benchmark override may also carry an inline `api_key`, so a key entered
    in the UI never has to be written to the environment.
    """
    provider = (role_setting(role, "PROVIDER") or "").strip().lower()
    if not provider:
        return None
    ensure_pyrit_initialized()
    if provider not in _ROLE_DEFAULT_MODEL:
        raise ValueError(
            f"Unsupported {role}_PROVIDER: {provider!r} "
            f"(use 'prof_llm', 'openai', 'anthropic', or 'venice')"
        )
    model = role_setting(role, "MODEL") or _ROLE_DEFAULT_MODEL[provider]
    key_env = role_setting(role, "API_KEY_ENV") or _ROLE_DEFAULT_KEY_ENV[provider]
    api_key = (llm_overrides.get().get(role) or {}).get("api_key") or os.getenv(key_env)
    if provider == "prof_llm" and not api_key:
        raise ValueError(
            f"HPC LLM is selected for {role.lower()} but {key_env} is not set. "
            "Add the supervisor-provided HPC key to the project-root .env file."
        )
    if provider in (*_OPENAI_COMPAT_ENDPOINT, "prof_llm"):
        # OpenAI, Venice, and the university HPC service use the same wire
        # protocol; only endpoint, key, and model differ.
        from pyrit.prompt_target import OpenAIChatTarget
        return OpenAIChatTarget(
            model_name=model, api_key=api_key,
            endpoint=_role_endpoint(role, provider),
            **_openai_target_options(provider),
        )
    # provider == "anthropic"
    # PyRIT 0.14 removed the dedicated AnthropicChatTarget — Anthropic is now
    # reached through OpenAIChatTarget against Anthropic's OpenAI-compatible
    # endpoint (PyRIT special-cases api.anthropic.com). Fall back to the native
    # target on older PyRIT (<0.14) where AnthropicChatTarget still exists.
    from pyrit.prompt_target import OpenAIChatTarget
    try:
        from pyrit.prompt_target import AnthropicChatTarget  # type: ignore
    except ImportError:
        AnthropicChatTarget = None  # type: ignore
    if AnthropicChatTarget is not None:
        return AnthropicChatTarget(model_name=model, api_key=api_key)
    return OpenAIChatTarget(
        model_name=model, api_key=api_key,
        endpoint="https://api.anthropic.com/v1",
    )


def build_judge_llm() -> Any:
    """Build the judge chat target from JUDGE_* env vars. None if not configured."""
    return build_role_llm("JUDGE")


def build_attacker_llm() -> Any:
    """Build the attacker chat target (orchestrators) from ATTACKER_* env vars."""
    return build_role_llm("ATTACKER")


def build_converter_llm_from_spec(spec: str | None) -> Any:
    """Build a converter LLM from an explicit ``"provider:model"`` spec.

    This is the *backup* converter model: when the primary converter model's
    content filter blocks a seed (see engine.converters.is_content_policy_block),
    the chain is rebuilt with this model instead — typically an uncensored
    provider such as ``venice:qwen-3-6-plus`` that will actually perform the
    rewrite. The API key is taken from the provider's default env var (the same
    keys the attacker roster uses), and the target is framed like the primary
    converter (unless CONVERTER_FRAMING is off).

    Returns None for an empty/unparseable spec or an unknown provider, so a
    blank "backup" field simply means "no backup".
    """
    if not spec or not str(spec).strip():
        return None
    provider, _, model = str(spec).strip().partition(":")
    provider, model = provider.strip().lower(), model.strip()
    if provider not in _ROLE_DEFAULT_MODEL:
        return None
    ensure_pyrit_initialized()
    model = model or _ROLE_DEFAULT_MODEL[provider]
    api_key = os.getenv(_ROLE_DEFAULT_KEY_ENV[provider])
    if provider in (*_OPENAI_COMPAT_ENDPOINT, "prof_llm"):
        from pyrit.prompt_target import OpenAIChatTarget
        endpoint = _role_endpoint("ATTACKER", provider)
        llm = OpenAIChatTarget(
            model_name=model, api_key=api_key,
            endpoint=endpoint,
            **_openai_target_options(provider),
        )
    else:  # anthropic — same fallback shape as build_role_llm
        from pyrit.prompt_target import OpenAIChatTarget
        try:
            from pyrit.prompt_target import AnthropicChatTarget  # type: ignore
        except ImportError:
            AnthropicChatTarget = None  # type: ignore
        if AnthropicChatTarget is not None:
            llm = AnthropicChatTarget(model_name=model, api_key=api_key)
        else:
            llm = OpenAIChatTarget(
                model_name=model, api_key=api_key,
                endpoint="https://api.anthropic.com/v1",
            )
    from ..settings import CONVERTER_FRAMING
    if CONVERTER_FRAMING:
        llm = _frame_converter_target(llm)
    return llm


# Cache the converter LLM per run (keyed on the ContextVar token) rather than in
# a plain module global: with per-benchmark overrides a shared global would leak
# one benchmark's attacker model into another's converters.
converter_llm_cache: "contextvars.ContextVar[Any]" = contextvars.ContextVar(
    "converter_llm", default=None
)


def _frame_converter_target(llm: Any) -> Any:
    """Prepend the authorised-transformation framing to every converter call.

    PyRIT's LLM-backed converters (Tense/Persuasion/Math/Translate/Variation)
    hand the converter model a bare user message, so a safety-trained model
    reads a harmful seed as a request TO IT and refuses — that refusal then comes
    back as the "converted" prompt (see engine.converters.converter_declined).
    Wrapping send_prompt_async to prepend CONVERTER_FRAMING_PREAMBLE restores the
    real context (mechanical text transformation inside an approved red-team run)
    and markedly cuts spurious converter refusals. Only what the CONVERTER model
    is asked changes; the converted attack prompt still reaches the target
    unchanged, so attack semantics are untouched.

    Best-effort per call: if the message can't be read or rebuilt (PyRIT
    reshuffles its Message API between releases), the original call goes through
    unchanged rather than failing the conversion. Mirrors the throttle wrap —
    shadow one method on the instance, leaving class/identity intact.
    """
    from ..prompts import CONVERTER_FRAMING_PREAMBLE
    from pyrit.models import Message  # type: ignore

    original = llm.send_prompt_async

    async def _framed(*args, **kwargs):
        msg = kwargs.get("message", args[0] if args else None)
        try:
            text = msg.get_value() if msg is not None else None
            if text:
                new_msg = Message.from_prompt(
                    prompt=CONVERTER_FRAMING_PREAMBLE + text, role="user")
                if "message" in kwargs:
                    return await original(*args, **{**kwargs, "message": new_msg})
                return await original(new_msg, *args[1:], **kwargs)
        except Exception:  # noqa: BLE001 — any API mismatch => send unchanged
            pass
        return await original(*args, **kwargs)

    llm.send_prompt_async = _framed
    return llm


def converter_llm() -> Any:
    """LLM used to drive the semantic converters.

    Prefers the attacker model (it is the one meant to be doing adversarial
    work); falls back to the judge model so the converters still run on a
    judge-only configuration. Cached for the current run, because the converter
    chain is rebuilt on every best-of-N attempt and each build would otherwise
    open a new client.

    The converter model is framed as an authorised transformation tool (see
    _frame_converter_target) unless CONVERTER_FRAMING is disabled, so a
    safety-trained model actually performs the rewrite instead of refusing it.
    """
    llm = converter_llm_cache.get()
    if llm is None:
        llm = build_attacker_llm() or build_judge_llm()
        if llm is not None:
            from ..settings import CONVERTER_FRAMING
            if CONVERTER_FRAMING:
                llm = _frame_converter_target(llm)
        converter_llm_cache.set(llm)
    if llm is None:
        raise RuntimeError(
            "The poetry / persuasion / tense converters need an LLM. Configure an "
            "attacker or judge model (in the benchmark form, or ATTACKER_PROVIDER / "
            "JUDGE_PROVIDER in .env)."
        )
    return llm


# Sentinel: "leave the ATTACKER override as the benchmark set it" (vs a roster
# entry that replaces it).
NO_ATTACKER_OVERRIDE = object()


def apply_attacker_override(config: Any) -> None:
    """Swap the ATTACKER entry of the current LLM-override ContextVar.

    `config` is a role-config dict (replaces ATTACKER), None (restore the
    benchmark's own ATTACKER override), or `NO_ATTACKER_OVERRIDE` (leave as-is).
    Also clears the per-run converter-LLM cache so the semantic converters pick
    up the new attacker.
    """
    if config is NO_ATTACKER_OVERRIDE:
        return
    base = dict(llm_overrides.get() or {})
    if config is None:
        # Restore whatever the benchmark originally set (stashed on first swap).
        orig = base.pop("_orig_attacker", base.get("ATTACKER"))
        if orig:
            base["ATTACKER"] = orig
        else:
            base.pop("ATTACKER", None)
    else:
        if "_orig_attacker" not in base:
            base["_orig_attacker"] = base.get("ATTACKER")
        base["ATTACKER"] = config
    llm_overrides.set(base)
    converter_llm_cache.set(None)


def attacker_roster(bench: Benchmark) -> list[dict]:
    """Return the attacker roster to run a dynamic attack against.

    A benchmark may define `attacker_roster` (a list of {provider, model}); when
    present, each dynamic attack runs once per entry and the outcomes union per
    objective. When absent, a single-entry roster reproduces the benchmark's
    configured / .env attacker exactly, so nothing changes for existing setups.
    """
    roster = bench.attacker_roster
    if not roster:
        return [{"label": None, "config": NO_ATTACKER_OVERRIDE}]
    out = []
    for entry in roster:
        prov = (entry.get("provider") or "").strip().lower()
        model = (entry.get("model") or "").strip()
        if not prov:
            continue
        cfg = {"provider": prov}
        if model:
            cfg["model"] = model
        if entry.get("api_key"):
            cfg["api_key"] = entry["api_key"]
        out.append({"label": f"{prov}/{model}" if model else prov, "config": cfg})
    return out or [{"label": None, "config": NO_ATTACKER_OVERRIDE}]
