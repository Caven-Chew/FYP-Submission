"""Turns a raw provider/driver error into a plain-English hint plus a fix.

One long lookup table; kept apart from the request handlers so both stay
readable. Add a new hint by adding a branch."""
from __future__ import annotations


# ---------- helpers ------------------------------------------------------

def parse_error_hint(detail: str) -> dict | None:
    """Map known failure strings in status_detail to a friendly action item.

    Returns a dict with keys: title, action, extra (optional second line).
    Returns None if no pattern matched (caller falls back to raw traceback only).
    """
    if not detail:
        return None
    d = detail.lower()

    # --- Circuit breaker: too many consecutive target failures ---
    if "consecutive target calls failed" in d:
        return {
            "title": "Benchmark stopped early — target unreachable",
            "action": (
                "Several target calls in a row failed, so the run stopped instead of "
                "burning through every remaining prompt against a dead target. Check:\n\n"
                "• For API targets: endpoint URL, API key, and that the service is up.\n"
                "• For UI targets: the selectors still match the page, and it isn't stuck "
                "behind a bot-check/login wall.\n\n"
                "Results collected before the stop are kept below. Fix the target and "
                "re-run the benchmark — you can also raise MAX_CONSECUTIVE_TARGET_FAILURES "
                "in .env if this is tripping on a target that's just slow, not down."
            ),
        }

    # --- Project moved machines: saved login session not found ---
    if "saved login session" in d and "not on this machine" in d:
        return {
            "title": "Saved login session missing",
            "action": (
                "This target signs in with a session recorded on another machine, and "
                "that file isn't here. The run was stopped rather than continuing "
                "signed out, because a login wall scraped as the model's reply would "
                "look like a result.\n\n"
                "• Copied the project from another computer? Copy its "
                "instance/ui_sessions/ folder across too — the app matches sessions "
                "by filename, so the paths themselves don't need fixing.\n"
                "• Otherwise re-record it: Targets → the target → Log in & save session.\n"
                "• Or clear ui_storage_state in Extra config to run signed out on purpose."
            ),
        }

    # --- Project moved machines: local file:// page not found ---
    if "points at a local page that is not on this machine" in d:
        return {
            "title": "Local page not found",
            "action": (
                "This target's ui_url is a file:// path from another machine. The app "
                "looked for the same filename in the project folder and in test/ and "
                "found nothing.\n\n"
                "• If it was one of the bundled mock pages, point ui_url at "
                "test/chat.html in this copy of the project.\n"
                "• If it was your own page, copy it into the project folder and update "
                "ui_url to match."
            ),
        }

    # --- Configuration: judge LLM missing ---
    if "judge_provider" in d and "not configured" in d:
        return {
            "title": "Judge LLM not configured",
            "action": (
                "The benchmark needs a judge LLM to score each response. For the "
                "workshop's free university HPC service, open your .env file and add:\n\n"
                "  JUDGE_PROVIDER=prof_llm\n"
                "  JUDGE_MODEL=qwen3.6-35b-a3b-fp8\n"
                "  JUDGE_API_KEY_ENV=LLM_API_KEY\n"
                "  LLM_ENDPOINT=https://<supervisor-provided-host>/v1\n"
                "  LLM_API_KEY=<supervisor-provided-key>"
            ),
        }

    if "prof llm" in d and ("llm_endpoint" in d or "llm_api_key" in d):
        return {
            "title": "HPC LLM credentials incomplete",
            "action": (
                "The free HPC judge is selected, but its endpoint or key is missing. "
                "Open your .env file and add:\n\n"
                "  LLM_ENDPOINT=https://<supervisor-provided-host>/v1\n"
                "  LLM_API_KEY=<supervisor-provided-key>\n\n"
                "The endpoint is only reachable on the campus network or VPN."
            ),
        }

    # --- Configuration: attacker LLM missing (dynamic attacks) ---
    if "attacker_provider not configured" in d or ("attacker_provider" in d and "not configured" in d):
        return {
            "title": "Attacker LLM not configured",
            "action": (
                "Dynamic attacks (Red-Teaming, Crescendo) require a separate attacker LLM "
                "for PyRIT's prompt generation. To keep the workshop on the free HPC "
                "service, add to your .env:\n\n"
                "  ATTACKER_PROVIDER=prof_llm\n"
                "  ATTACKER_MODEL=qwen3.6-35b-a3b-fp8\n"
                "  ATTACKER_API_KEY_ENV=LLM_API_KEY\n\n"
                "Alternatively, select only Static dataset attacks when creating a benchmark — "
                "those do not need an attacker LLM."
            ),
        }

    # --- Configuration: Azure endpoint missing ---
    if "azure_ai_project_endpoint is not set" in d or "azure_ai_project_endpoint" in d:
        return {
            "title": "Azure AI Project endpoint not configured",
            "action": (
                "Azure RedTeam attacks need an Azure AI Project. Add to your .env:\n\n"
                "  AZURE_AI_PROJECT_ENDPOINT=https://<your-project>.api.azureml.ms\n\n"
                "You can find this URL in the Azure AI Studio portal under your project settings. "
                "If you don't have an Azure subscription, select non-Azure attacks instead."
            ),
        }

    # --- Missing package: azure-ai-evaluation ---
    if "azure-ai-evaluation" in d or "azure.ai.evaluation" in d:
        return {
            "title": "Azure evaluation packages not installed",
            "action": (
                "Azure RedTeam attacks require extra packages. Install them with:\n\n"
                "  pip install azure-ai-evaluation azure-identity\n\n"
                "Then restart the Flask server. If you don't need Azure attacks, "
                "deselect them when creating the next benchmark."
            ),
        }

    # --- Authentication / API key errors ---
    if any(x in d for x in ("401", "unauthorized", "authentication", "invalid api key", "incorrect api key")):
        return {
            "title": "Authentication failed — API key rejected",
            "action": (
                "The target LLM (or judge/attacker LLM) rejected the API key. Check:\n\n"
                "• The environment variable name in the Target's 'API key env var' field matches "
                "the variable you set in .env (e.g. OPENAI_API_KEY).\n"
                "• The key hasn't expired or been revoked.\n"
                "• For Azure OpenAI, the endpoint URL and api_version are correct."
            ),
        }

    # --- Rate limit ---
    if any(x in d for x in ("ratelimit", "rate limit", "rate_limit", "429", "too many requests")):
        return {
            "title": "Rate limit hit",
            "action": (
                "The API returned a rate-limit error (HTTP 429). Options:\n\n"
                "• Reduce MAX_PROMPTS_PER_ATTACK in your .env (workshop default: 1) to send fewer prompts.\n"
                "• Wait a few minutes and re-run the benchmark.\n"
                "• Switch to a tier with higher rate limits, or use a local model (Ollama) "
                "that has no API rate limits."
            ),
        }

    # --- Site blocked the automated browser outright (bot detection) ---
    if "could not load" in d and ("err_http_response_code_failure" in d or "err_blocked_by" in d
                                   or "err_connection_closed" in d or "net::err_" in d):
        return {
            "title": "Site blocked the automated browser",
            "action": (
                "The target page rejected the automated request before any selector "
                "was even checked — this is bot detection (Cloudflare / DataDome / "
                "Google), not a selector or timing problem.\n\n"
                "On the target, untick 'Run browser hidden (headless)' so the run uses a "
                "visible window, then use 'Log in & save session' so it starts already "
                "signed in. Some sites (chatgpt.com among them) block automated browsers "
                "regardless — those are not benchmarkable through the browser UI."
            ),
        }

    # --- Browser UI target: the page worked, the response selector didn't ---
    # Must be tested BEFORE the generic network rule below, whose "timeout"
    # keyword would otherwise claim this and blame the connection — when in
    # fact the page loaded, the prompt was typed and sent, and only the reply
    # was never matched.
    if "no new response appeared in" in d:
        return {
            "title": "The reply never matched your response selector",
            "action": (
                "The page loaded and the prompt was sent, but nothing new matched "
                "the response selector within the wait window. Usually one of:\n\n"
                "• The model is slower than the wait — raise ui_wait_ms (a chat UI "
                "backed by a large model often needs 15000-20000).\n"
                "• The selector is wrong or has gone stale — re-run 'Capture "
                "selectors' and click the reply bubble again.\n"
                "• The reply arrived somewhere the selector doesn't cover (a "
                "different container, or inside a shadow DOM).\n\n"
                "Check the screenshot diagnosis above and the debug list below: if "
                "the reply is visible on screen but missing from 'newly added', it "
                "is the selector, not the wait."
            ),
        }

    # --- Browser UI target: the browser binary is missing --------------------
    # A launch failure is NOT an ImportError, so the Patchright rule further
    # down (which matches only missing-module errors) never sees it.
    if ("executable doesn't exist" in d or "browsertype.launch" in d
            or "please run the following command to download new browsers" in d):
        return {
            "title": "The browser binary is not installed",
            "action": (
                "Patchright is installed but its Chromium build is missing — this "
                "happens after a fresh install or a version upgrade. Download it "
                "with:\n\n"
                "  patchright install chromium\n\n"
                "Run it inside the same environment the app runs in (e.g. "
                ".venv/bin/patchright install chromium), then retry."
            ),
        }

    # --- Network / connection errors ---
    if any(x in d for x in ("connectionerror", "connecterror", "connectionrefused",
                             "connection refused", "failed to connect", "name or service not known",
                             "getaddrinfo failed", "timeout", "timed out")):
        return {
            "title": "Could not connect to the target LLM",
            "action": (
                "The benchmark runner could not reach the model endpoint. Check:\n\n"
                "• For Ollama: is `ollama serve` running? Is the endpoint URL correct "
                "(default: http://localhost:11434)?\n"
                "• For remote APIs: do you have internet access from this machine?\n"
                "• For Generic HTTP targets: verify the endpoint URL in the Target settings.\n"
                "• Check that any firewall or proxy isn't blocking outbound requests."
            ),
        }

    # --- Unknown / unrecognised provider ---
    if "unknown provider" in d:
        return {
            "title": "Unrecognised target provider",
            "action": (
                "The provider set on this target is not supported. "
                "Go to Targets and edit the target — choose one of: "
                "openai, azure_openai, anthropic, ollama, huggingface, http_generic."
            ),
        }

    # --- PyRIT version / API mismatch ---
    if ("has no" in d and "pyrit" in d) or ("pyrit" in d and "renamed" in d) or \
            ("fetch_" in d and "importerror" in d):
        return {
            "title": "PyRIT version mismatch",
            "action": (
                "A PyRIT function or class couldn't be found — the installed PyRIT version may "
                "have renamed it. Try upgrading:\n\n"
                "  pip install --upgrade pyrit\n\n"
                "If you're pinned to a specific version, check the PyRIT changelog for API changes."
            ),
        }

    # --- Missing package: Patchright (the required browser engine) ---
    if ("playwright" in d or "patchright" in d) and \
            ("importerror" in d or "modulenotfounderror" in d or "no module named" in d):
        return {
            "title": "Patchright is not installed",
            "action": (
                "Browser UI targets require Patchright — it evades the bot-detection "
                "(Cloudflare / DataDome / Google) that stock automation can't. "
                "Install it with:\n\n"
                "  pip install patchright\n"
                "  patchright install chromium\n\n"
                "The second command downloads the Chromium browser binary. "
                "Run both, then retry the benchmark."
            ),
        }

    # --- Generic ImportError (missing package) ---
    if "importerror" in d or "modulenotfounderror" in d or "no module named" in d:
        return {
            "title": "A required Python package is not installed",
            "action": (
                "The benchmark failed because a package import failed. "
                "Look at the traceback below to find the missing module name, then run:\n\n"
                "  pip install <module-name>\n\n"
                "Common ones: reportlab (PDF export), azure-ai-evaluation (Azure RedTeam), "
                "pyrit (core framework), playwright (Browser UI targets)."
            ),
        }

    # --- Objectives missing (dynamic attacks) ---
    if "no objectives available" in d or "objectives.json" in d:
        return {
            "title": "No objectives found for dynamic attack",
            "action": (
                "Dynamic attacks (Red-Teaming, Crescendo) load goals from "
                "app/data/objectives.json. That file is missing or empty, and the "
                "fallback dataset also returned nothing. Either:\n\n"
                "• Create app/data/objectives.json with at least one objective row.\n"
                "• Or run only Static dataset attacks, which don't need objectives.json."
            ),
        }

    # --- No match ---
    return None
