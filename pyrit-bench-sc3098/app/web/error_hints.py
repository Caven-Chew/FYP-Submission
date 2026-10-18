"""Turns a raw provider/driver error into a plain-English hint plus a fix.

One long lookup table; kept apart from the request handlers so both stay
readable. Add a new hint by adding a branch. Written for the SC3098 setup:
the judge is set in .env (README.md, Configuration and troubleshooting), targets are a student's own
bot reached as UI (browser) or API -> Custom, and the tool may run in Docker.
Labels match the app's buttons and the student documents."""
from __future__ import annotations

from ..engine.scoring import JUDGE_ERROR_HINT
from ..settings import ADDRESS_GUIDE as _AG, JUDGE_GUIDE as _JG, SETTINGS_SHORT as _ENV

_DOCKER_ADDRESS = (
    "• Running this tool in Docker? Inside Docker, localhost means the container, not your "
    "laptop: use http://host.docker.internal:PORT for a bot on your laptop (Streamlit uses 8501). "
    "Without Docker, use http://localhost:PORT; host.docker.internal works only inside Docker."
)


# ---------- helpers ------------------------------------------------------

def parse_error_hint(detail: str) -> dict | None:
    """Map known failure strings in status_detail to a friendly action item.

    Returns a dict with keys: title, action, extra (optional second line).
    Returns None if no pattern matched (caller falls back to raw traceback only).
    """
    if not detail:
        return None
    d = detail.lower()

    # --- Judge settings missing, unfilled or unsupported (.env) ---
    if (any(x in d for x in ("no judge model is configured", "judge key is missing",
                             "template placeholder", "llm_endpoint is required", "llm_endpoint in",
                             "no api key configured for judge"))
            or ("judge_provider" in d and any(x in d for x in ("not configured", "not supported",
                                                                 "unsupported")))):
        return {
            "title": "The judge is not set up",
            "action": (
                f"Every result needs a judge model. Open {_ENV} and fill in the HPC judge lines "
                f"exactly as {_JG} shows: JUDGE_PROVIDER=profllm, "
                "JUDGE_MODEL=qwen3.6-35b-a3b-fp8, LLM_ENDPOINT and LLM_API_KEY, "
                "JUDGE_API_KEY_ENV=LLM_API_KEY.\n\n"
                f"Replace every <...> placeholder, save {_ENV}, restart the app, then click "
                "Test judge on the dashboard."
            ),
        }

    # --- Judge key variable belongs to another provider ---
    if "belongs to another provider" in d:
        return {
            "title": "The judge key variable does not match the judge",
            "action": (
                f"Set JUDGE_API_KEY_ENV=LLM_API_KEY, exactly as in {_JG} (HPC judge, "
                "JUDGE_PROVIDER=profllm). The app refuses the "
                "mismatch so that a key is never sent to the wrong service. Restart the app after "
                "the change and click Test judge on the dashboard."
            ),
        }

    # --- The judge failed while scoring (network, VPN, key) ---
    if "judge_error" in d or "judge-error" in d or "judge returned an error" in d:
        return {
            "title": "The judge could not score some replies",
            "action": (
                JUDGE_ERROR_HINT + "\n\n"
                "• Connect to the school network or NTU VPN, and check LLM_ENDPOINT and "
                f"LLM_API_KEY in {_ENV} ({_JG}).\n"
                f"• After changing {_ENV}, restart the app and click Test judge on the dashboard.\n\n"
                "Retry failed & continue scores the saved replies again; your bot is not "
                "contacted again."
            ),
        }

    # --- Circuit breaker: too many consecutive target failures ---
    if "consecutive target calls failed" in d or "calls in a row failed" in d:
        return {
            "title": "Run stopped early: your bot did not answer",
            "action": (
                "Several messages in a row failed, so the run stopped instead of sending the "
                "rest. Results collected before the stop are kept.\n\n"
                "• Check that your bot is running: open its address in your own browser.\n"
                + _DOCKER_ADDRESS + "\n"
                "• UI (browser) target: on Targets, click Test for this target. If the reply is "
                "empty or is your own message, use Duplicate and edit, run Capture from the page "
                "again and save the copy.\n"
                "• API target: check the Endpoint URL and any key saved with the target.\n\n"
                "Then click Retry failed & continue on the run page."
            ),
        }

    # --- Project moved machines: saved login session not found ---
    if "saved login session" in d and "not on this machine" in d:
        return {
            "title": "Saved login session missing",
            "action": (
                "This target signs in with a session saved on another computer, and that file "
                "is not here. The run stopped instead of continuing signed out, because a login "
                "page read as the bot's reply would look like a result.\n\n"
                "• Copied the project from another computer? Copy its instance/ui_sessions/ "
                "folder across too.\n"
                "• Otherwise save a new session: on Targets, click Duplicate and edit for this "
                "target, click Log in & save session, sign in with a test account and click "
                "Create target."
            ),
        }

    # --- Project moved machines: local file:// page not found ---
    if "points at a local page that is not on this machine" in d:
        return {
            "title": "Local page not found",
            "action": (
                "This target's Page URL is a file:// path from another computer, and no file "
                "with that name is in this project folder or in test/.\n\n"
                "Copy the page into the project folder, then use Duplicate and edit on Targets "
                "to point the Page URL at it."
            ),
        }

    # --- Authentication / API key errors ---
    if any(x in d for x in ("401", "unauthorized", "authentication", "invalid api key", "incorrect api key")):
        return {
            "title": "A key was rejected",
            "action": (
                "The service answered, but refused the key.\n\n"
                f"• The judge (a judge error, or Test judge fails): check the judge key in {_ENV} as "
                f"in {_JG} (LLM_API_KEY). Restart the app and click Test judge.\n"
                "• Your bot's API: check the API key saved with this target. To change it, use "
                "Duplicate and edit on Targets.\n"
                "• A UI (browser) target that shows a login page: use Log in & save session with "
                "a test account, never a personal one."
            ),
        }

    # --- Rate limit ---
    if any(x in d for x in ("ratelimit", "rate limit", "rate_limit", "429", "too many requests")):
        return {
            "title": "Too many requests (rate limit)",
            "action": (
                "The service asked the tool to slow down (HTTP 429).\n\n"
                "• Wait a minute, then click Retry failed & continue on the run page.\n"
                "• HPC judge: many groups at once can hit the limit; wait, then retry.\n"
                "• Your own bot's API has a low limit: set 'Rate limiting — pause between "
                "bursts' when you create the target."
            ),
        }

    # --- Browser target: nothing is listening at the Page URL ---
    # Before the bot-detection rule: a stopped bot gives net::ERR_CONNECTION_REFUSED,
    # which is an address problem, not a site refusing automation.
    if any(x in d for x in ("err_connection_refused", "err_name_not_resolved",
                             "err_address_unreachable", "err_connection_timed_out")):
        return {
            "title": "Could not open your bot's page",
            "action": (
                "Nothing answered at the Page URL.\n\n"
                "• Is your bot running? Open the same address in your own browser.\n"
                + _DOCKER_ADDRESS + "\n\n"
                f"The address table is in {_AG}."
            ),
        }

    # --- Site blocked the automated browser outright (bot detection) ---
    if "could not load" in d and ("err_http_response_code_failure" in d or "err_blocked_by" in d
                                   or "err_connection_closed" in d or "net::err_" in d):
        return {
            "title": "The page refused the automated browser",
            "action": (
                "The page rejected the tool's browser before anything was typed. Your own "
                "robo-advisor should not do this, so first check the Page URL by opening it in "
                "your own browser.\n\n"
                "Test only your own group's bot, or the bot assigned to you in a red-teaming "
                "round. Never point the tool at public or commercial chatbots or websites "
                "(student guide, Rules of engagement)."
            ),
        }

    # --- Browser UI target: the page worked, the response selector didn't ---
    # Must be tested BEFORE the generic network rule below, whose "timeout"
    # keyword would otherwise claim this and blame the connection — when in
    # fact the page loaded, the prompt was typed and sent, and only the reply
    # was never matched.
    if "no new response appeared in" in d:
        return {
            "title": "The reply never matched your Response selector",
            "action": (
                "The page loaded and the message was sent, but nothing new matched the "
                "Response selector in time. Usually one of:\n\n"
                "• Your bot is slower than the wait: raise 'Wait for response — max (seconds)'.\n"
                "• The selector is wrong: run Capture from the page again and click the bot's "
                "reply.\n"
                "• The reply appeared somewhere the selector does not cover.\n\n"
                "Open the Debug block: if the reply is on the page but not under 'newly added', "
                "the selector is the problem, not the wait."
            ),
        }

    # --- Browser UI target: the browser binary is missing --------------------
    # A launch failure is NOT an ImportError, so the Patchright rule further
    # down (which matches only missing-module errors) never sees it.
    if ("executable doesn't exist" in d or "browsertype.launch" in d
            or "please run the following command to download new browsers" in d):
        return {
            "title": "The browser is not installed",
            "action": (
                "The tool's Chromium browser is missing. Without Docker, activate .venv and "
                "run:\n\n"
                "  patchright install chromium\n\n"
                "With Docker, the browser is part of the image: rebuild it with "
                "docker compose up --build. Then try again."
            ),
        }

    # --- Network / connection errors (bot or judge) ---
    if any(x in d for x in ("connectionerror", "connecterror", "connectionrefused",
                             "connection refused", "connection error", "failed to connect",
                             "name or service not known", "nodename nor servname",
                             "getaddrinfo failed", "timeout", "timed out")):
        return {
            "title": "Could not connect",
            "action": (
                "The tool could not reach the address.\n\n"
                "• Your bot: is it running? Open the same address in your own browser.\n"
                + _DOCKER_ADDRESS + "\n"
                "• The university HPC judge: it needs the campus network or NTU VPN. Connect, "
                "then click Test judge on the dashboard.\n\n"
                f"The address table is in {_AG}."
            ),
        }

    # --- Unknown / unrecognised target provider ---
    if "unknown provider" in d:
        return {
            "title": "This connection type is not supported",
            "action": (
                "This target uses a connection type this version of the tool does not support. "
                "On Targets, create it again as UI (browser), or as API with the provider "
                "Custom. Duplicate and edit keeps the name and notes for you."
            ),
        }

    # --- PyRIT version / API mismatch ---
    if ("has no" in d and "pyrit" in d) or ("pyrit" in d and "renamed" in d) or \
            ("fetch_" in d and "importerror" in d):
        return {
            "title": "Installed packages do not match this version",
            "action": (
                "A PyRIT function or class could not be found, so the installed packages differ "
                "from the ones this tool was tested with. Without Docker, activate .venv and "
                "run:\n\n"
                "  pip install -r requirements.txt\n\n"
                "With Docker, rebuild with docker compose up --build."
            ),
        }

    # --- Missing package: Patchright (the required browser engine) ---
    if ("playwright" in d or "patchright" in d) and \
            ("importerror" in d or "modulenotfounderror" in d or "no module named" in d):
        return {
            "title": "Patchright is not installed",
            "action": (
                "UI (browser) targets need Patchright and its Chromium browser. Activate .venv "
                "and run:\n\n"
                "  pip install -r requirements.txt\n"
                "  patchright install chromium\n\n"
                "Then try again (README.md, Setup)."
            ),
        }

    # --- Generic ImportError (missing package) ---
    if "importerror" in d or "modulenotfounderror" in d or "no module named" in d:
        return {
            "title": "A required Python package is not installed",
            "action": (
                "A package import failed. Without Docker, activate .venv and run:\n\n"
                "  pip install -r requirements.txt\n\n"
                "With Docker, rebuild with docker compose up --build (README.md, Setup)."
            ),
        }

    # --- No match ---
    return None
