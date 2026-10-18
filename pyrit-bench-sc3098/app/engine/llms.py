"""Judge / attacker / converter LLMs, and the per-run override plumbing."""
from __future__ import annotations

from .. import settings as _settings
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
    "profllm": "qwen3.6-35b-a3b-fp8",
}


_ROLE_DEFAULT_KEY_ENV = {
    "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
    "venice": "VENICE_API_KEY",
    "profllm": "LLM_API_KEY",
}

# Providers reached through PyRIT's OpenAIChatTarget against an OpenAI-compatible
# /chat/completions endpoint. Venice is included so an uncensored attacker/
# converter (e.g. qwen-3-6-plus) can stand in for the OpenAI models that get
# moderation-blocked on harmful seeds — only the base URL and key differ.
_OPENAI_COMPAT_ENDPOINT = {
    "openai": "https://api.openai.com/v1",
    "venice": "https://api.venice.ai/api/v1",
}


def _profllm_endpoint() -> str:
    return os.getenv("LLM_ENDPOINT", "").strip().rstrip("/")

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
    # Model names and credential-variable names belong to their provider.
    # A provider override must not inherit either from another provider's env.
    override_provider = str(ov.get("provider") or "").strip().lower()
    env_provider = (os.getenv(f"{role}_PROVIDER") or "").strip().lower()
    if key in ("MODEL", "API_KEY_ENV") and override_provider and override_provider != env_provider:
        return None
    return os.getenv(f"{role}_{key}")


def _refuse_foreign_key(role: str, provider: str, key_env: str) -> None:
    """Never send one provider's key to another provider's endpoint.

    A course .env that keeps ``JUDGE_API_KEY_ENV=OPENAI_API_KEY`` while switching
    ``JUDGE_PROVIDER`` to the course endpoint (``profllm``) would otherwise post
    the operator's OpenAI key to that endpoint (review finding).
    """
    own = _ROLE_DEFAULT_KEY_ENV.get(provider)
    foreign = {v for k, v in _ROLE_DEFAULT_KEY_ENV.items() if k != provider and v != own}
    if key_env in foreign:
        raise ValueError(
            f"{role}_API_KEY_ENV={key_env} belongs to another provider, but {role}_PROVIDER is "
            f"{provider!r}. Set {role}_API_KEY_ENV={own} (or delete that line) so the key is "
            "only sent to its own service."
        )


# The <...> values in .env.example. A value that still contains one (or the
# same words with the brackets removed) has not been filled in (finding #4).
_PLACEHOLDER_WORDS = ("your-key", "your-openai-key", "endpoint-address-from-the-teaching-team")


def _is_placeholder(value: str | None) -> bool:
    text = (value or "").strip().lower()
    return ("<" in text and ">" in text) or any(word in text for word in _PLACEHOLDER_WORDS)


def _placeholder_problem(name: str) -> str:
    # Names the variable only; the value is never echoed.
    return (f"{name} in {_settings.SETTINGS_FILE} still contains a template placeholder. Replace every <...> "
            f"value in the judge lines with the real value (see {_settings.JUDGE_GUIDE}), then "
            "restart the app.")


def judge_setup_problem() -> str | None:
    """A plain-English reason the judge cannot run, or None when it is configured.

    Checked before a run is queued, so a student sees what to fix instead of a
    run that fails on its first prompt. No client is built and nothing is sent.
    The .env.example placeholders count as "not configured".
    """
    provider = (role_setting("JUDGE", "PROVIDER") or "").strip().lower()
    if not provider:
        return ("No judge model is configured, so replies cannot be scored. Set JUDGE_PROVIDER "
                f"and its key in {_settings.SETTINGS_FILE} (see {_settings.JUDGE_GUIDE}), then restart the app.")
    if _is_placeholder(provider):
        return _placeholder_problem("JUDGE_PROVIDER")
    if provider not in _ROLE_DEFAULT_MODEL:
        return f"JUDGE_PROVIDER={provider!r} is not supported (use openai, anthropic, venice or profllm)."
    if _is_placeholder(role_setting("JUDGE", "MODEL")):
        return _placeholder_problem("JUDGE_MODEL")
    key_env = role_setting("JUDGE", "API_KEY_ENV") or _ROLE_DEFAULT_KEY_ENV[provider]
    if _is_placeholder(key_env):
        return _placeholder_problem("JUDGE_API_KEY_ENV")
    try:
        _refuse_foreign_key("JUDGE", provider, key_env)
    except ValueError as exc:
        return str(exc)
    if not os.getenv(key_env):
        return f"The judge key is missing: set {key_env} in {_settings.SETTINGS_FILE} (see {_settings.JUDGE_GUIDE})."
    if _is_placeholder(os.getenv(key_env)):
        return _placeholder_problem(key_env)
    if provider == "profllm":
        endpoint = _profllm_endpoint()
        if not endpoint:
            return f"LLM_ENDPOINT is required when JUDGE_PROVIDER=profllm (see {_settings.JUDGE_GUIDE})."
        if _is_placeholder(endpoint):
            return _placeholder_problem("LLM_ENDPOINT")
        if not endpoint.lower().startswith(("http://", "https://")):
            return (f"LLM_ENDPOINT in {_settings.SETTINGS_FILE} must be the full address from the teaching team, "
                    f"starting with http:// or https:// (see {_settings.JUDGE_GUIDE}).")
    return None


def role_api_key(role: str) -> str | None:
    """The key value `role` would send, used only to mask it in error text."""
    provider = (role_setting(role, "PROVIDER") or "").strip().lower()
    key_env = role_setting(role, "API_KEY_ENV") or _ROLE_DEFAULT_KEY_ENV.get(provider, "")
    return ((llm_overrides.get().get(role) or {}).get("api_key")
            or (os.getenv(key_env) if key_env else None))


# ---------- dashboard "Test judge" -----------------------------------------

JUDGE_TEST_TIMEOUT_S = 60


def _allow_one_request(judge: Any) -> None:
    """Make an OpenAI-SDK judge target send at most one HTTP request.

    The SDK client retries connection errors, 429s and 5xx twice, and PyRIT
    retries a rate limit or an empty reply up to ten times. Test judge must not
    spend more than one call, so retries are switched off on a copy of the
    client and a second request is refused with the first request's outcome.
    Other judge objects (test doubles, non-SDK targets) are left unchanged.
    """
    client = getattr(judge, "_async_client", None)
    if client is None or not hasattr(client, "with_options"):
        return
    client = client.with_options(max_retries=0, timeout=JUDGE_TEST_TIMEOUT_S)
    completions = client.chat.completions
    create, outcome = completions.create, []

    async def create_once(*args, **kwargs):
        if outcome:
            raise RuntimeError("Test judge sends only one request, and the judge's first reply "
                               f"was not usable ({outcome[0]}).")
        outcome.append("an empty or rate-limited reply")
        try:
            return await create(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — recorded, then re-raised unchanged
            outcome[0] = f"{type(exc).__name__}: {exc}"
            raise

    completions.create = create_once
    judge._async_client = client


def check_judge() -> dict:
    """Send ONE short fixed request to the configured judge (dashboard Test judge).

    Returns {ok, provider, model, reply, error, setup}. Never raises; `error`
    is trimmed and key-masked. `setup` is True when nothing was sent because
    the .env judge settings are incomplete.
    """
    import asyncio
    from ..prompts import JUDGE_TEST_PROMPT
    from .scoring import judge_error_text

    provider = (role_setting("JUDGE", "PROVIDER") or "").strip().lower()
    model = role_setting("JUDGE", "MODEL") or _ROLE_DEFAULT_MODEL.get(provider, "")
    out = {"ok": False, "provider": provider, "model": model, "reply": "", "error": None, "setup": False}
    problem = judge_setup_problem()
    if problem:
        out.update(error=problem, setup=True)
        return out
    try:
        judge = build_judge_llm()
        _allow_one_request(judge)
        from pyrit.models import Message  # type: ignore

        async def _send():
            return await asyncio.wait_for(
                judge.send_prompt_async(message=Message.from_prompt(prompt=JUDGE_TEST_PROMPT, role="user")),
                JUDGE_TEST_TIMEOUT_S)

        responses = asyncio.run(_send())
        first = responses[0] if responses else None
        reply = (first.get_value() if first is not None else "").strip()
        if first is not None and callable(getattr(first, "is_error", None)) and first.is_error() is True:
            raise RuntimeError(f"The judge returned an error: {reply}")
        if not reply:
            raise RuntimeError("The judge returned an empty reply.")
        out.update(ok=True, reply=reply[:200])
    except asyncio.TimeoutError:
        out["error"] = f"TimeoutError: no answer within {JUDGE_TEST_TIMEOUT_S} seconds."
    except Exception as exc:  # noqa: BLE001 — reported to the student, never raised
        out["error"] = judge_error_text(exc)
    return out


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
            f"(use 'openai', 'anthropic', 'venice', or 'profllm')"
        )
    model = role_setting(role, "MODEL") or _ROLE_DEFAULT_MODEL[provider]
    key_env = role_setting(role, "API_KEY_ENV") or _ROLE_DEFAULT_KEY_ENV[provider]
    _refuse_foreign_key(role, provider, key_env)
    api_key = (llm_overrides.get().get(role) or {}).get("api_key") or os.getenv(key_env)
    if not api_key:
        # Passing None lets the SDK look up its own default key, which may
        # belong to a different provider. Fail before constructing a client.
        raise ValueError(f"No API key configured for {role} provider {provider!r}. Set {key_env} or supply an inline key.")
    if provider == "profllm":
        endpoint = _profllm_endpoint()
        if not endpoint:
            raise ValueError("LLM_ENDPOINT is required when using profllm")
        from pyrit.prompt_target import OpenAIChatTarget
        # A J0 run's judge (Version 1, /v1) turns reasoning off, as the study's H1S
        # run did (configs/h1s_hpc_severity_config.json). H1S's 512-token cap is not
        # copied: the study set cut-off replies aside, but parse_severity would read
        # a stray digit from one.
        from .scoring import J0_METHOD
        extra = ({"reasoning_effort": "none"} if role == "JUDGE"
                 and llm_overrides.get().get("JUDGE_METHOD") == J0_METHOD else None)
        return OpenAIChatTarget(model_name=model, api_key=api_key, endpoint=endpoint,
                                extra_body_parameters=extra)
    if provider in _OPENAI_COMPAT_ENDPOINT:
        # openai + venice: identical client, only the base URL differs.
        from pyrit.prompt_target import OpenAIChatTarget
        return OpenAIChatTarget(
            model_name=model, api_key=api_key,
            endpoint=_OPENAI_COMPAT_ENDPOINT[provider],
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
    from .scoring import DUAL_METHOD, J0_METHOD, V2_METHOD
    cfg = llm_overrides.get()
    if cfg.get("JUDGE_METHOD") != DUAL_METHOD:
        return build_role_llm("JUDGE")
    token = llm_overrides.set({**cfg, "JUDGE_METHOD": V2_METHOD})
    try:
        v2 = build_role_llm("JUDGE")
        llm_overrides.set({**cfg, "JUDGE_METHOD": J0_METHOD})
        v1 = build_role_llm("JUDGE")
    finally:
        llm_overrides.reset(token)
    if v1 is None or v2 is None:
        return None
    # Keep the base target usable by internal attack scorers; final judging
    # selects the independent target with the version's request settings.
    v2._judge_versions = {"v1": v1, "v2": v2}
    return v2


def build_attacker_llm() -> Any:
    """Build the attacker chat target (orchestrators) from ATTACKER_* env vars.

    With no attacker configured, the judge model drives the attacks (4 Oct 2026): the
    course kit sets only the HPC judge, as the converters already fall back to it.
    """
    return build_role_llm("ATTACKER") or build_role_llm("JUDGE")


def build_converter_llm_from_spec(spec: str | None, *, frame_rewrite: bool = True) -> Any:
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
    if provider == "profllm":
        endpoint = _profllm_endpoint()
        if not endpoint:
            return None
        from pyrit.prompt_target import OpenAIChatTarget
        llm = OpenAIChatTarget(model_name=model, api_key=api_key, endpoint=endpoint)
    elif provider in _OPENAI_COMPAT_ENDPOINT:
        from pyrit.prompt_target import OpenAIChatTarget
        llm = OpenAIChatTarget(
            model_name=model, api_key=api_key,
            endpoint=_OPENAI_COMPAT_ENDPOINT[provider],
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
    if frame_rewrite and CONVERTER_FRAMING:
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


class AttackerBackupError(RuntimeError):
    """The backup attacker failed before it could produce a prompt."""


def with_attacker_backup(attacker: Any, events: list[str]) -> Any:
    """Retry a refused or policy-blocked attacker call once using the backup.

    Keep the concrete PyRIT target type and original conversation context. Only
    the adversarial target is wrapped; target and judge errors cannot trigger it.
    Backup calls are direct, so another refusal cannot start a fallback loop.
    """
    from copy import copy, deepcopy
    from ..settings import ATTACKER_BACKUP_MODEL
    from .converters import is_content_policy_block
    from .scoring import looks_like_refusal

    if not ATTACKER_BACKUP_MODEL:
        return attacker
    wrapped = copy(attacker)
    primary_send = attacker.send_prompt_async
    backup = None

    async def send_with_backup(*args, **kwargs):
        nonlocal backup
        # A primary request may mutate the message; preserve its original context.
        saved_args, saved_kwargs = deepcopy((args, kwargs))
        try:
            responses = await primary_send(*args, **kwargs)
        except Exception as exc:
            if not is_content_policy_block(exc):
                raise
            reason = "content-policy block"
        else:
            texts = [r.get_value().strip() for r in (responses or []) if r.get_value().strip()]
            if not texts or not all(looks_like_refusal(t) for t in texts):
                return responses
            reason = "refusal"
        events.append(f"Attacker backup {ATTACKER_BACKUP_MODEL} used after primary {reason}.")
        try:
            if backup is None:
                backup = build_converter_llm_from_spec(ATTACKER_BACKUP_MODEL, frame_rewrite=False)
            if backup is None:
                raise ValueError("ATTACKER_BACKUP_MODEL must name a supported provider:model")
            return await backup.send_prompt_async(*saved_args, **saved_kwargs)
        except Exception as exc:
            raise AttackerBackupError(f"Attacker backup failed: {type(exc).__name__}: {exc}") from exc

    wrapped.send_prompt_async = send_with_backup
    return wrapped
