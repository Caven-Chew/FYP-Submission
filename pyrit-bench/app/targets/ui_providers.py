"""UI providers: targets driven through a real browser rather than an API."""
from __future__ import annotations


from .base import UIProvider
from .browser.target import BrowserUITarget
from .common import extra_of


# ----- UI providers ------------------------------------------------------

class BrowserUIProvider(UIProvider):
    slug = "ui_browser"
    label = "Browser UI (Playwright)"
    help = ("For targets that only have a web chat page. Playwright opens a real "
            "browser, types each prompt, waits for the reply, and scrapes it.")
    fields = [
        {"key": "ui_url", "label": "Page URL", "placeholder": "https://yourapp.local/chat",
         "hint": "Full URL of the chatbot page", "required": True},
        {"key": "ui_input_selector", "label": "Input selector", "placeholder": "textarea#prompt-input",
         "hint": "CSS selector for the text field where prompts are typed", "required": True},
        {"key": "ui_submit_selector", "label": "Submit selector", "placeholder": "button[type=submit]   or   Enter",
         "hint": "CSS selector for the send button, or the word Enter to press the Enter key", "required": True},
        {"key": "ui_response_selector", "label": "Response selector", "placeholder": ".message.assistant",
         "hint": "CSS selector for the assistant reply element(s); the last new match is used", "required": True},
        {"key": "ui_pre_actions", "label": "Setup steps (dismiss popups)",
         "placeholder": '[{"type":"click","selector":"button.accept-cookies"},{"type":"wait","ms":500}]',
         "hint": ("Optional JSON list of steps run before each prompt to clear cookie banners / modals: "
                  "click / key (Enter, Escape) / wait. Easiest to record these with Capture above."),
         "required": False},
        # Every wait below is entered in SECONDS, because that is the unit a
        # person actually reasons about ("give it 50 seconds to answer").
        # views_targets.parse_kv converts each _s key into the milliseconds key
        # the target reads, so nothing underneath the form changes and targets
        # saved before this keep working untouched.
        {"key": "ui_page_load_s", "label": "Page load timeout (seconds)", "placeholder": "30",
         "hint": ("How long to wait for the page to open and the message box to appear. "
                  "Nothing to do with how long a reply takes — raise this only if the site "
                  "itself is slow to come up, e.g. a cold-starting cloud app."),
         "type": "number", "default": "30"},
        {"key": "ui_llm_load_s", "label": "Wait for LLM to load (seconds)",
         "placeholder": "(same as page load timeout)",
         "hint": ("Wait for an auto-generated welcome/greeting message to fully arrive before the "
                  "first prompt is sent, so it can't be mistaken for the reply — some bots generate "
                  "the greeting live, so it can be as slow as any other reply. Blank uses the page "
                  "load timeout. 0 disables."),
         "type": "number", "default": ""},
        {"key": "ui_response_min_s", "label": "Wait for response — min (seconds)", "placeholder": "2",
         "hint": ("Once a reply appears, how long it must stop changing before it counts as "
                  "finished. This is what stops a half-written answer being captured and scored "
                  "as if it were the whole reply. 2 seconds suits most streaming chat UIs."),
         "type": "number", "default": "2"},
        {"key": "ui_response_max_s", "label": "Wait for response — max (seconds)", "placeholder": "50",
         "hint": ("The longest a single prompt may take, start to finish, before it is recorded "
                  "as a timeout. Set this comfortably above the target's slowest realistic reply: "
                  "a run sends hundreds of prompts, and a value that is too low quietly turns slow "
                  "answers into failures."),
         "type": "number", "default": "50"},
    ]

    def build_target(self, t, api_key, multi_turn: bool = False):
        # NOTE: BrowserUITarget is not a PyRIT PromptChatTarget, so multi-turn
        # strategies still reject it. Accepting the flag keeps the interface
        # uniform; making UI targets multi-turn capable is future work.
        return BrowserUITarget(extra_of(t))

    @classmethod
    def form_meta(cls):
        m = super().form_meta()
        m["fields"] = cls.fields
        return m
