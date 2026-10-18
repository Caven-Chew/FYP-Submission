"""BrowserUITarget — drives a real chat UI in a browser as a PyRIT target."""
from __future__ import annotations

import os

from .engine import load_async_playwright, parse_pre_actions
from .sessions import resolve_local_url, resolve_session_path
from .vision import diagnose_ui_timeout
from ..common import wrap_response
from ...prompts import LOGIN_WALL_MARKERS
from ...settings import (
    UI_CAPTURE_BUDGET, UI_RECOVER_REPLY, UI_RESPONSE_MAX_MS, UI_STREAM_SETTLE_MS,
)


class BrowserUITarget:
    """Playwright-backed target that drives a real browser chat UI.

    On the first call for a conversation it:
      1. Launches a fresh Chromium page (headless by default).
      2. Navigates to `url` and runs any configured pre-actions once.
    Then, for every prompt in that same conversation:
      3. Types the prompt into `input_selector`.
      4. Clicks `submit_selector` (or presses Enter if selector is ``"Enter"``).
      5. Waits for new text to appear inside `response_selector`,
         or falls back to a fixed `wait_ms` sleep.
      6. Returns the trimmed inner-text of the newest matching response element.

    The browser tab is kept open across turns that share the same
    conversation_id (see send_prompt_async) so multi-turn attacks (Crescendo,
    Red-Teaming) actually escalate within one chat session instead of each
    turn opening a blank page with no memory of what came before. A new
    conversation_id — a new attack objective, the next static-dataset prompt,
    a connection test — closes the old tab and opens a clean one.

    extra_config keys (all required unless noted):
      ui_url, ui_input_selector, ui_submit_selector, ui_response_selector,
      ui_headless (opt, default headed; "true" to hide),
      ui_timeout_ms (opt, default 30000) — page load + input box only,
      ui_initial_settle_ms (opt, defaults to ui_timeout_ms) — greeting wait,
      ui_stream_settle_ms (opt, default 800) — how long a reply must hold still,
      ui_response_max_ms (opt, default 50000) — total budget for one reply,
      ui_pre_actions (opt, JSON list of click/key/wait steps run before each
        prompt to dismiss popups), ui_pre_action_timeout_ms (opt, default 4000).

    The target form writes the last four in seconds (ui_page_load_s,
    ui_llm_load_s, ui_response_min_s, ui_response_max_s); views_targets.parse_kv
    converts them to the millisecond keys above. Targets saved before that
    change carry ui_wait_ms / ui_stream_max_ms instead and still resolve to the
    same budget they always had — see the constructor.
    """

    def __init__(self, extra: dict):
        # resolve_local_url() only matters for a file:// page whose path came
        # from another machine; every other URL passes straight through.
        self.url          = resolve_local_url(extra.get("ui_url", ""))
        self.input_sel    = extra.get("ui_input_selector", "").strip()
        self.submit_sel   = extra.get("ui_submit_selector", "").strip()
        self.response_sel  = extra.get("ui_response_selector", "").strip()
        # Page load only: how long to wait for the page and the input box.
        # NOT a response budget — that is response_max_ms below.
        self.timeout_ms   = int(extra.get("ui_timeout_ms", 30000))
        # Total budget for one reply, from pressing send to accepting finished
        # text. Targets saved before this key existed don't carry it, so
        # reproduce what they actually did rather than silently handing them a
        # different budget: the old code slept ui_wait_ms twice and gave up.
        if "ui_response_max_ms" in extra:
            self.response_max_ms = int(extra["ui_response_max_ms"])
        elif "ui_stream_max_ms" in extra:
            self.response_max_ms = int(extra["ui_stream_max_ms"])
        elif "ui_wait_ms" in extra:
            self.response_max_ms = 2 * int(extra["ui_wait_ms"])
        else:
            self.response_max_ms = UI_RESPONSE_MAX_MS
        # Default to a *visible* window so benchmark runs pop up the browser
        # (easier to watch, and headed evades bot detection better). Set
        # ui_headless=true to run hidden.
        self.headless     = str(extra.get("ui_headless", "false")).lower() == "true"
        # ms to keep the window open after a run before closing. 0 in benchmark
        # runs; the connection test sets it so the user can watch / inspect.
        self.linger_ms    = int(extra.get("ui_linger_ms", 0))
        # ms to wait after typing before pressing Enter — heavily-hydrated chat
        # UIs (ChatGPT-style) may not have wired up send the instant text lands.
        self.submit_delay_ms = int(extra.get("ui_submit_delay_ms", 1000))
        # Path to a Playwright storage_state JSON (cookies + localStorage) saved
        # from a one-time manual login, so auth-gated targets stay signed in.
        self.storage_state = (extra.get("ui_storage_state", "") or "").strip()
        # Optional setup steps replayed after page load, before each prompt is
        # typed — clicks / key presses / waits that dismiss cookie banners or
        # modals covering the chat input. Best-effort: a step that no longer
        # applies (popup already gone) is skipped, never fatal.
        self.pre_actions = parse_pre_actions(extra.get("ui_pre_actions", ""))
        self.pre_action_timeout_ms = int(extra.get("ui_pre_action_timeout_ms", 4000))
        # Some chat UIs auto-generate a personalised greeting (often a real LLM
        # call, not static text) the moment the session opens — it can take as
        # long to arrive as any other reply. If the first prompt's "existing
        # text" baseline were captured before it finishes, the greeting landing
        # a moment later looks exactly like a fresh reply and gets returned
        # instead of the real answer — this is how a connection test can
        # "pass" with the bot's welcome message instead of an actual response.
        # Defaults to the same budget as a normal reply (ui_timeout_ms) since
        # the greeting can be just as slow to generate. 0 disables the wait.
        self.initial_settle_ms = int(extra.get("ui_initial_settle_ms", self.timeout_ms))
        # Open browser session for the current conversation: {pw, browser, page,
        # conversation_id}. None when no tab is currently open.
        # Lazily opened by send_prompt_async, closed when a different
        # conversation_id shows up or via aclose().
        self._session: dict | None = None
        # Populated by send_prompt_async: {response_selector, baseline, attempts}
        # — what response_sel actually collected, for debugging a bad capture.
        # See test_target() for how a connection test surfaces this.
        self.last_debug: dict | None = None
        # Failure captures (screenshot + gzipped HTML). The runner sets
        # `capture_dir` to this benchmark's folder before the run starts; left
        # None everywhere else (connection tests, ad-hoc use), which disables
        # writing to disk entirely — a manual test already shows you the page.
        self.capture_dir: str | None = None
        self.capture_budget = UI_CAPTURE_BUDGET
        self.capture_count = 0
        # Once a reply is detected, how long its text must stay unchanged before
        # it counts as complete. This is what stops a half-streamed answer being
        # captured and scored as the whole thing. 0 accepts the first sighting.
        self.response_min_ms = int(extra.get("ui_stream_settle_ms", UI_STREAM_SETTLE_MS))
        # Whether a vision-recovered, page-verified reply may stand in as the
        # response when the selector found nothing.
        self.recover_reply = str(
            extra.get("ui_recover_reply", "1" if UI_RECOVER_REPLY else "0")
        ).strip().lower() not in ("0", "false", "no")

        missing = [k for k, v in [
            ("ui_url", self.url),
            ("ui_input_selector", self.input_sel),
            ("ui_submit_selector", self.submit_sel),
            ("ui_response_selector", self.response_sel),
        ] if not v]
        if missing:
            raise ValueError(
                f"BrowserUITarget: missing extra_config keys: {', '.join(missing)}"
            )

    def get_identifier(self):
        try:
            from pyrit.models import ComponentIdentifier  # type: ignore  # PyRIT >=0.13
        except ImportError:  # older PyRIT (pyrit.identifiers is a deprecated shim in 0.14)
            from pyrit.identifiers import ComponentIdentifier  # type: ignore
        return ComponentIdentifier.of(self, params={"url": self.url, "input_sel": self.input_sel})

    def _build_identifier(self):
        return self.get_identifier()

    @property
    def capabilities(self):
        from pyrit.prompt_target.common.target_capabilities import TargetCapabilities  # type: ignore
        return TargetCapabilities(supports_multi_turn=True)

    @property
    def configuration(self):
        # PyRIT 0.14's executor.attack module (RedTeamingAttack, CrescendoAttack,
        # ConversationManager) reads target.configuration.includes(...) /
        # .ensure_can_handle(...) instead of the older direct `.capabilities`
        # check. BrowserUITarget is duck-typed (not a PromptTarget subclass), so
        # without this it raises "'BrowserUITarget' object has no attribute
        # 'configuration'" the moment a dynamic (multi-turn) attack runs.
        from pyrit.prompt_target.common.target_configuration import TargetConfiguration  # type: ignore
        return TargetConfiguration(capabilities=self.capabilities)

    # JS that reads an <input>/<textarea> value or a contenteditable's text.
    _VALUE_JS = ("el => (el && el.value !== undefined && el.value !== null)"
                 " ? el.value : (el ? el.innerText : '')")

    # JS that finds and clicks the most likely 'send' button near the input.
    _CLICK_SEND_JS = r'''
el => {
  const scope = el.closest('form, [class*="composer"], [class*="Composer"], [class*="input"], [class*="chat"]') || document;
  const cands = Array.from(scope.querySelectorAll('button, [role="button"], input[type="submit"]'));
  const score = (b) => {
    const t = ((b.getAttribute('aria-label')||'') + ' ' + (b.getAttribute('title')||'') + ' ' + (b.textContent||'')).toLowerCase();
    let s = 0;
    if (b.type === 'submit') s += 3;
    if (/\bsend\b|submit/.test(t)) s += 2;
    else if (/send/.test(t)) s += 1;
    return s;
  };
  let best = null, bestScore = 0;
  for (const b of cands) {
    if (b.disabled || b.offsetParent === null) continue;
    const s = score(b);
    if (s > bestScore) { best = b; bestScore = s; }
  }
  if (best) best.click();
}
'''

    async def _input_still_holds(self, page, prompt) -> bool:
        """True if the prompt text is still in the input (i.e. it wasn't sent).

        On a read error we return True (assume not yet sent) so an unreadable
        input doesn't cause us to stop early — better to try the next send
        strategy than to give up before clicking the send button.
        """
        try:
            cur = await page.eval_on_selector(self.input_sel, self._VALUE_JS)
        except Exception:  # noqa: BLE001
            return True
        return (cur or "").strip() == (prompt or "").strip()

    async def _submit_via_enter(self, page, prompt) -> None:
        """Submit the typed prompt via Enter, trying common send mechanisms in turn.

        Chat UIs wire send-on-Enter differently; we try each approach and stop as
        soon as the input clears (the message was sent), so 'Enter' works without
        the user having to locate a send button.
        """
        # let the input's change handler register the typed value first
        await page.wait_for_timeout(self.submit_delay_ms)

        # 1) Enter on the focused input — handles most keydown-to-send apps.
        try:
            await page.focus(self.input_sel)
        except Exception:  # noqa: BLE001
            pass
        await page.press(self.input_sel, "Enter")
        await page.wait_for_timeout(250)
        if not await self._input_still_holds(page, prompt):
            return

        # 2) keyboard-level Enter, in case the element-scoped press was swallowed.
        try:
            await page.keyboard.press("Enter")
            await page.wait_for_timeout(250)
        except Exception:  # noqa: BLE001
            pass
        if not await self._input_still_holds(page, prompt):
            return

        # 3) submit the enclosing <form> via its submit event, if there is one.
        try:
            await page.eval_on_selector(
                self.input_sel,
                "el => { const f = el.form || el.closest('form');"
                " if (f && f.requestSubmit) f.requestSubmit(); }",
            )
            await page.wait_for_timeout(250)
        except Exception:  # noqa: BLE001
            pass
        if not await self._input_still_holds(page, prompt):
            return

        # 4) last resort: click the most likely 'send' button near the input.
        try:
            await page.eval_on_selector(self.input_sel, self._CLICK_SEND_JS)
        except Exception:  # noqa: BLE001
            pass

    async def _run_pre_actions(self, page) -> None:
        """Replay the configured setup steps to clear popups before typing.

        Every step is best-effort: a click whose target isn't present (e.g. the
        cookie banner only shows on the first visit, or was already dismissed via
        a saved session) is swallowed so it never aborts the run.
        """
        for act in self.pre_actions:
            typ = (act.get("type") or "").lower()
            try:
                if typ == "wait":
                    await page.wait_for_timeout(int(act.get("ms", 1000)))
                elif typ == "click":
                    sel = (act.get("selector") or "").strip()
                    if sel:
                        try:
                            await page.click(sel, timeout=self.pre_action_timeout_ms)
                        except Exception:  # noqa: BLE001 — popup may be absent
                            pass
                elif typ == "key":
                    key = (act.get("key") or "Enter").strip() or "Enter"
                    await page.keyboard.press(key)
            except Exception:  # noqa: BLE001 — setup steps never fail a run
                pass

    async def _looks_like_login_wall(self, page) -> bool:
        """Is the page showing a sign-in screen instead of the chatbot?

        Deliberately conservative: it only decides the difference between two
        error messages, never whether a run continues, so a false negative
        costs nothing beyond the older, vaguer wording.
        """
        try:
            body = (await page.inner_text("body"))[:4000].lower()
        except Exception:  # noqa: BLE001 — page detached; fall back to the URL
            body = ""
        try:
            url = (page.url or "").lower()
        except Exception:  # noqa: BLE001
            url = ""
        if any(seg in url for seg in ("/login", "/signin", "/sign-in", "/auth")):
            return True
        return any(m in body for m in LOGIN_WALL_MARKERS)

    async def _settle_response_area(self, page) -> None:
        """Wait for a greeting/intro message (if any) to fully arrive and stop changing.

        Called once, right after the input box appears, before the first
        prompt of a conversation is ever sent. Guards against chat UIs that
        auto-generate a greeting when the session opens — often a real LLM
        call, so it can take just as long to arrive as any other reply. If the
        "existing text" baseline in send_prompt_async were captured before it
        finishes, the greeting landing a moment later looks exactly like a
        fresh reply — the bot's welcome message gets returned as if it
        answered the prompt.

        Two phases, both needed:
          1. Nothing has appeared yet at response_sel — this is NOT "settled",
             it just hasn't started. Keep polling (don't return early just
             because two empty reads matched — that was the original bug:
             a greeting that takes longer than one poll tick to even start
             looked indistinguishable from "no greeting at all").
          2. Content has appeared — wait for it to stop changing (two
             consecutive identical non-empty reads) before declaring settled.

        Gives up after ui_initial_settle_ms either way, so a page with no
        greeting doesn't pay the full timeout — it exits as soon as the first
        (empty) grace window passes with nothing new appearing.
        """
        if self.initial_settle_ms <= 0:
            return
        import time as _time

        async def _snapshot():
            try:
                texts = await page.eval_on_selector_all(
                    self.response_sel, "els => els.map(e => e.innerText.trim())",
                )
            except Exception:  # noqa: BLE001
                return []
            return [t for t in texts if t]

        budget = self.initial_settle_ms / 1000
        deadline = _time.time() + budget
        # Give up quickly (not the full budget) if nothing ever shows up —
        # most targets have no greeting at all and shouldn't pay this cost.
        no_content_deadline = _time.time() + min(2.0, budget)

        prev = await _snapshot()
        seen_content = bool(prev)
        while _time.time() < deadline:
            await page.wait_for_timeout(300)
            cur = await _snapshot()
            if cur:
                if cur == prev and seen_content:
                    return   # stopped changing — greeting (if any) has finished
                seen_content = True
            elif not seen_content and _time.time() > no_content_deadline:
                return       # nothing ever appeared — no greeting to wait for
            prev = cur

    def _accept_recovered_reply(self, result: dict, existing_set, is_own_echo) -> str | None:
        """Decide whether a vision-extracted reply may stand in as the response.

        `vision._verify_extraction` has already confirmed the text actually
        appears on the page, which rules out fabrication. Two page-relative
        checks remain, and only this class knows enough to make them: the text
        must not be something that was already on screen before the prompt was
        sent, and must not be our own prompt echoed back. Both genuinely appear
        on the page, and both would be the wrong answer.

        Returns the text to use, or None to let the timeout stand.
        """
        if not self.recover_reply:
            return None
        text = (result.get("reply_text") or "").strip()
        if not text:
            return None

        if text in existing_set:
            reason = "it was already on the page before the prompt was sent"
        elif is_own_echo(text):
            reason = "it is the prompt echoed back, not a reply"
        else:
            reason = None

        if reason:
            self._note_debug("recovery_rejected", reason)
            return None

        self._note_debug("recovered_reply", text[:2000])
        self._note_debug("recovery_note", (
            f"Response recovered from the page by the vision model, not by "
            f"ui_response_selector ({self.response_sel!r}) — the selector matched nothing. "
            "Fix the selector: recovery costs an LLM call per prompt and only works "
            "when the reply has finished rendering."
        ))
        return text

    def _note_debug(self, key: str, value) -> None:
        """Record a debugging note, creating last_debug if the failure happened
        before send_prompt_async had a chance to initialise it (e.g. during
        page load)."""
        if not isinstance(self.last_debug, dict):
            self.last_debug = {}
        self.last_debug[key] = value

    async def _save_capture(self, page, label: str, png: bytes | None = None) -> str | None:
        """Write a full-page PNG + gzipped HTML for a failed send; return its ref.

        The pair is deliberate: the HTML says which selector *would* have
        matched the reply (the usual misconfiguration), while the screenshot
        catches what markup cannot represent — a CAPTCHA in a cross-origin
        iframe, an element hidden behind a modal, shadow-DOM content, a
        canvas-rendered page. Either alone has a blind spot the other covers.

        Best-effort and never raises: a capture that fails must not replace the
        real error with a filesystem error. Returns None when capture is off
        (no capture_dir), the run's budget is spent, or writing failed.

        `png` lets a caller that already took a screenshot (the response-timeout
        path, which also feeds it to the vision model) avoid taking a second one.
        """
        if not self.capture_dir:
            return None
        if self.capture_count >= self.capture_budget:
            self._note_debug(
                "capture_note",
                f"(not saved — this run's capture budget of {self.capture_budget} is used up; "
                f"raise UI_CAPTURE_BUDGET to keep more)",
            )
            return None

        import gzip
        import os as _os
        import re as _re

        safe = _re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-")[:40] or "failure"
        ref = f"{self.capture_count:03d}-{safe}"
        try:
            _os.makedirs(self.capture_dir, exist_ok=True)
            if png is None:
                png = await page.screenshot(full_page=True)
            with open(_os.path.join(self.capture_dir, f"{ref}.png"), "wb") as fh:
                fh.write(png)
            # Gzipped: a hydrated chat UI serialises to 0.5-2 MB of markup and
            # hashed class names, which compresses ~10-20x.
            html = await page.content()
            with gzip.open(_os.path.join(self.capture_dir, f"{ref}.html.gz"),
                           "wt", encoding="utf-8") as fh:
                fh.write(html)
        except Exception as e:  # noqa: BLE001 — diagnostic only, never masks the real error
            self._note_debug("capture_note", f"(capture failed: {type(e).__name__}: {e})")
            return None

        self.capture_count += 1
        self._note_debug("capture_ref", ref)
        return ref

    async def _open_session(self, conversation_id: str | None) -> None:
        """Launch a fresh browser tab, navigate to `url`, and get it ready to type into.

        Runs pre-actions and waits for the input box once, at the start of the
        conversation — later turns in the same conversation reuse this same
        page (see send_prompt_async), so we don't want to reload or re-dismiss
        popups before every single message.
        """
        async_playwright, _engine = load_async_playwright()   # Patchright (required)
        pw = await async_playwright().start()
        try:
            browser = await pw.chromium.launch(headless=self.headless)
            ctx_kwargs = {}
            saved_login = resolve_session_path(self.storage_state)
            if saved_login:
                ctx_kwargs["storage_state"] = saved_login   # reuse saved login
            context = await browser.new_context(**ctx_kwargs)
            page = await context.new_page()

            try:
                await page.goto(self.url, timeout=self.timeout_ms)
            except Exception as e:  # noqa: BLE001 — turn a raw nav failure into a hint
                # net::ERR_HTTP_RESPONSE_CODE_FAILURE / ERR_BLOCKED_BY_* etc. mean the
                # site rejected the automated request outright (bot detection), before
                # any selector is even in play — distinct from, and upstream of, the
                # wait_for_selector timeout handled below.
                # Capture first: a blocked navigation still leaves a page (the
                # error page, or the challenge that replaced the site) and it is
                # usually the single most informative screenshot of the run.
                await self._save_capture(page, "nav-failed")
                raise TimeoutError(
                    f"Could not load {self.url} ({type(e).__name__}: {e}). "
                    + ("This usually means the site's bot detection blocked the automated "
                       "request outright. Untick 'Run browser hidden (headless)' so the run "
                       "uses a visible window, and use 'Log in & save session' on the target "
                       "form so it starts already signed in.")
                ) from e
            # Clear any cookie banner / modal sitting over the chat input.
            await self._run_pre_actions(page)
            try:
                await page.wait_for_selector(self.input_sel, timeout=self.timeout_ms)
            except Exception as e:  # noqa: BLE001 — turn the raw timeout into a hint
                # The page loaded but the chat box never appeared: a bot check, a
                # login wall, or a stale selector. All three are obvious in a
                # screenshot and ambiguous in the exception text.
                await self._save_capture(page, "input-selector-missing")
                # A stale saved session is by far the most common cause and the
                # least obvious from the exception: the browser lands on a login
                # page, the chat box legitimately never appears, and the message
                # reads as a selector problem. Look before blaming the selector.
                if self.storage_state and await self._looks_like_login_wall(page):
                    raise TimeoutError(
                        f"The saved login session for this target appears to have "
                        f"expired — {self.url} is showing a sign-in page, so the chat "
                        f"box {self.input_sel!r} never appeared. Re-capture it with "
                        f"'Log in & save session' on the target form. (Waited "
                        f"{self.timeout_ms} ms.)"
                    ) from e
                raise TimeoutError(
                    f"Input box {self.input_sel!r} never appeared on {self.url} "
                    f"(waited {self.timeout_ms} ms). "
                    + ("This run used a hidden (headless) browser, which sites like "
                       "ChatGPT block with a bot-check or login wall — untick 'Run browser "
                       "hidden' and use 'Log in & save session' on the target form. "
                       if self.headless else
                       "The page may be showing a bot-check or login wall, or the input "
                       "selector is stale — re-capture it, and check the saved session "
                       "has not expired. ")
                    + "Tip: a connection test forces a visible window, so it can pass while "
                      "a headless run fails."
                ) from e
            # Let a still-streaming greeting/intro message finish rendering
            # before send_prompt_async takes its "existing text" baseline —
            # otherwise it can be mistaken for the reply to the first prompt.
            await self._settle_response_area(page)
        except Exception:
            try:
                await pw.stop()
            except Exception:  # noqa: BLE001
                pass
            raise

        self._session = {
            "pw": pw, "browser": browser, "page": page,
            "conversation_id": conversation_id,
        }

    async def _close_session(self) -> None:
        """Close the open browser session, if any (respects `linger_ms` first)."""
        session = self._session
        if not session:
            return
        self._session = None
        page, browser, pw = session["page"], session["browser"], session["pw"]
        # When lingering is on (manual connection tests), keep the window
        # visible for a moment so the user can inspect the page — on success
        # or on a selector failure — before it closes.
        if self.linger_ms:
            try:
                await page.wait_for_timeout(self.linger_ms)
            except Exception:  # noqa: BLE001
                pass
        try:
            await browser.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            await pw.stop()
        except Exception:  # noqa: BLE001
            pass

    async def aclose(self) -> None:
        """Close any lingering browser session.

        Callers should invoke this once they're done with the target — after
        a connection test, or at the end of a benchmark run — so a tab left
        open after the last conversation's final turn doesn't stay running
        for the rest of the process. A no-op if nothing is open.
        """
        await self._close_session()

    async def send_prompt_async(self, *, message) -> list:
        prompt = message.get_value() if hasattr(message, "get_value") else str(message)
        # PyRIT tags every turn of one multi-turn attack (Crescendo,
        # Red-Teaming) with the same conversation_id, and gives each new
        # conversation (a new objective, the next static-dataset prompt, a
        # connection test) a fresh one. Keying the browser session on it means
        # multi-turn attacks keep typing into the SAME chat tab — actually
        # escalating within one conversation — while unrelated prompts each
        # still get a clean page.
        conv_id = getattr(message, "conversation_id", None)
        # Clear this send's capture state up front. `last_debug` is rebuilt
        # further down, but _open_session below runs BEFORE that — so without
        # this, a page-load failure whose own capture was skipped (budget spent)
        # would leave the previous send's capture_ref visible and get it
        # recorded against the wrong prompt.
        if isinstance(self.last_debug, dict):
            self.last_debug.pop("capture_ref", None)
            self.last_debug.pop("capture_note", None)
        if self._session is None or self._session["conversation_id"] != conv_id:
            await self._close_session()
            await self._open_session(conv_id)

        page = self._session["page"]

        # Enter the prompt with real typing where possible. Rich-text
        # editors (Lexical, ProseMirror, Slate, Draft.js — common in chat
        # UIs) ignore fill(): the text shows on screen but the editor's
        # internal model stays empty, so Enter submits nothing. Typing
        # dispatches the key events those editors track; fall back to
        # fill() only if sequential typing isn't supported.
        loc = page.locator(self.input_sel).first
        try:
            await loc.click()
        except Exception:  # noqa: BLE001
            pass
        try:
            await loc.fill("")
        except Exception:  # noqa: BLE001
            pass
        try:
            await loc.press_sequentially(prompt, delay=15)
        except Exception:  # noqa: BLE001 — older Playwright / element type
            try:
                await loc.type(prompt, delay=15)
            except Exception:  # noqa: BLE001
                await loc.fill(prompt)

        async def _snapshot_texts() -> list[str]:
            """The full current list of response_sel matches, trimmed and non-empty."""
            texts = await page.eval_on_selector_all(
                self.response_sel, "els => els.map(e => e.innerText.trim())",
            )
            return [t for t in texts if t]

        # Record the list of messages as it stands right now — right after
        # the initial-greeting settle for turn 1 of a conversation (see
        # _open_session), or right after the previous reply for later turns.
        # Everything below is diffed against this baseline.
        existing_list = await _snapshot_texts()
        existing_set = set(existing_list)

        # Debugging aid — what response_sel actually collected at each stage,
        # so a connection test can show it back (see test_target()) instead of
        # leaving you guessing why the wrong text got picked. Overwritten each
        # call; not persisted anywhere (not part of the benchmark record).
        self.last_debug = {"response_selector": self.response_sel,
                            "baseline": existing_list, "attempts": []}

        if self.submit_sel.strip().lower() in ("enter", "return"):
            await self._submit_via_enter(page, prompt)
        else:
            try:
                await page.click(self.submit_sel)
            except Exception as e:  # noqa: BLE001 — turn a raw click failure into a hint
                # The send button is missing, disabled, or covered by an overlay
                # the pre-actions didn't clear. Without a capture this surfaces as
                # a bare Playwright timeout naming a selector, which cannot tell
                # you which of those three it was.
                await self._save_capture(page, "submit-failed")
                raise TimeoutError(
                    f"Could not click the send button {self.submit_sel!r} "
                    f"({type(e).__name__}: {e}). The button may be disabled until "
                    "the input registers text, hidden behind a cookie banner or "
                    "modal (add a ui_pre_actions step to dismiss it), or the "
                    "selector may be stale — re-capture it. Some UIs have no send "
                    'button at all: set the submit selector to "Enter" instead.'
                ) from e

        # Many chat UIs render the just-sent prompt into the SAME element
        # list as the reply (no distinct "assistant only" selector), and echo
        # it back almost instantly — long before the real reply, which can
        # take many seconds to generate. Without this filter, "the first new
        # thing we see" is reliably our own prompt talking back to us. Match
        # on a prefix (not equality) since UIs often append a trailing
        # timestamp/newline to the echoed bubble, e.g. "{prompt}\n\n{time}".
        prompt_norm = prompt.strip()

        def _is_own_echo(text: str) -> bool:
            return bool(prompt_norm) and text.strip().startswith(prompt_norm)

        # One poll loop covers both halves of the wait — how long we will wait
        # for a reply to appear at all, and how long it must then hold still
        # before we believe it has finished streaming. Both share
        # response_max_ms as a single deadline, so "max" means what a person
        # expects: the longest this one prompt can take.
        #
        # This replaces two fixed sleeps of ui_wait_ms. That older shape had
        # two faults: the ceiling was hard-capped at 2 x ui_wait_ms regardless
        # of what the form said, and it slept the FULL interval before so much
        # as looking at the page, so a reply that arrived in one second still
        # cost the whole window. Polling fixes both at once.
        import time as _time

        poll_ms = min(300, max(50, self.response_max_ms // 20))
        deadline = _time.time() + (self.response_max_ms / 1000)
        hold_needed = max(0.0, self.response_min_ms / 1000)

        added: list[str] = []
        collected: list[str] = []
        current: str | None = None
        first_seen: str | None = None
        stable_since: float | None = None
        settled = False
        polls = 0

        while _time.time() < deadline:
            await page.wait_for_timeout(poll_ms)
            polls += 1
            try:
                collected = await _snapshot_texts()
            except Exception:  # noqa: BLE001 — page navigated/detached mid-stream
                break
            added = [t for t in collected if t not in existing_set and not _is_own_echo(t)]
            if not added:
                # Either nothing has arrived yet, or the element re-rendered and
                # is momentarily blank. Keep whatever we already accepted rather
                # than resetting the streak on a transient gap.
                continue
            latest = added[-1]
            if first_seen is None:
                first_seen = latest
            if latest != current:
                current, stable_since = latest, _time.time()
                if hold_needed <= 0:
                    settled = True
                    break
            elif stable_since is not None and (_time.time() - stable_since) >= hold_needed:
                settled = True
                break

        # One summary entry, not one per poll: this feeds the connection-test
        # panel, which wants to show what the selector matched, not a trace.
        self.last_debug["attempts"].append({"collected": collected, "added": added})

        if current is not None:
            self._note_debug("stream_settle", {
                "first_seen": (first_seen or "")[:500],
                "final": current[:500],
                "grew": current != first_seen,
                "polls": polls,
                "settled": settled,
            })
            if not settled:
                self._note_debug(
                    "stream_note",
                    f"(reply still changing when the {self.response_max_ms} ms response "
                    f"budget ran out — returned as-is; it may be truncated)",
                )

        if current is None:
            waited_ms = self.response_max_ms
            # One screenshot, two consumers: it is written to disk as durable
            # evidence AND handed to the vision model for the inline summary.
            # The two are deliberately not coupled — the summary still runs when
            # the capture budget is spent, and the files are still written when
            # the vision call is skipped or fails.
            shot = None
            try:
                shot = await page.screenshot(full_page=True)
            except Exception:  # noqa: BLE001 — page detached/crashed
                pass
            try:
                page_text = await page.inner_text("body")
            except Exception:  # noqa: BLE001 — body missing/detached; text is a nice-to-have
                page_text = ""

            await self._save_capture(page, "response-timeout", png=shot)

            if shot is None:
                result = {"summary": "(screenshot unavailable — the page was closed or crashed)",
                          "state": "unknown", "reply_text": "", "rejected": None}
            else:
                result = await diagnose_ui_timeout(shot, page_text, self.response_sel, waited_ms)
            self.last_debug["vision_diagnosis"] = result["summary"]
            self.last_debug["diagnosis_state"] = result["state"]

            # If the reply was on the page all along and only the selector
            # missed it, returning it beats recording a failure: the target DID
            # answer, and calling that a timeout understates its attack surface.
            recovered = self._accept_recovered_reply(result, existing_set, _is_own_echo)
            if recovered is not None:
                return wrap_response(recovered)

            # The diagnosis is NOT concatenated into the exception message. It is
            # LLM prose, and mixing it into the error string caused two problems:
            # the UI showed one unreadable blob with no boundary between fact and
            # guess, and error_hints.py substring-matched phrases *inside* the
            # prose ("...no errors, rate limits, or CAPTCHAs...") and produced a
            # confident, wrong "Rate limit hit" hint for a plain timeout.
            # Callers read it from last_debug["vision_diagnosis"] instead.
            raise TimeoutError(
                f"No new response appeared in '{self.response_sel}' within the "
                f"{waited_ms} ms response budget "
                f"('Wait for response — max' on the target). If this target's "
                f"replies routinely take longer, raise it."
            )

        return wrap_response(current)
