"""Reply time for UI (browser) targets: from the send action to the final reply text (user request, 4 Oct 2026).

Before, a UI prompt's time also counted opening the tab, loading the page, waiting for the greeting, the
1 s submit delay, typing and the 0.8 s check that the reply had stopped changing. Now the browser target
starts the clock just before Enter or the Send click and stops it when the reply text reached its final
form; single prompts and conversation turns use that time.

Offline: a local HTML chat page (file://) that answers after a fixed delay, driven by the real
BrowserUITarget in headless Chromium. No network, no model calls.
Run from SC3098: ../.venv/bin/python -B test/test_ui_latency.py -v
"""
import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['PYTHON_DOTENV_DISABLED'] = '1'

PAGE = """<!doctype html><meta charset="utf-8"><title>bot</title>
<div id="log"><div class="bot">Hello, I am the test bot.</div></div>
<input id="msg" autofocus>
<script>
const msg = document.getElementById('msg'), log = document.getElementById('log');
msg.addEventListener('keydown', e => {
  if (e.key !== 'Enter' || !msg.value) return;
  msg.value = '';
  const bot = (text, el) => { el = el || document.createElement('div'); el.className = 'bot'; el.textContent = text; log.appendChild(el); return el; };
  if (location.hash === '#stream') {
    setTimeout(() => { const el = bot('Partial'); setTimeout(() => { el.textContent = 'Partial reply, now complete.'; }, %(rest)d); }, %(first)d);
  } else {
    setTimeout(() => bot('The complete reply.'), %(delay)d);
  }
});
</script>"""
DELAY_MS = 1500


class UiLatencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='sc3098-ui-latency-')
        cls.page = Path(cls.tmp.name) / 'bot.html'
        cls.page.write_text(PAGE % {'delay': DELAY_MS, 'first': 1000, 'rest': DELAY_MS - 1000}, encoding='utf-8')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def send(self, fragment=''):
        from app.targets.browser.target import BrowserUITarget
        target = BrowserUITarget({
            'ui_url': self.page.as_uri() + fragment, 'ui_input_selector': '#msg', 'ui_submit_selector': 'Enter',
            'ui_response_selector': '.bot', 'ui_headless': 'true', 'ui_initial_settle_ms': '300',
            'ui_response_max_ms': '10000',   # the default 1 s submit delay and 0.8 s settle check stay on
        })

        class Msg:
            conversation_id = 'c1'

            def get_value(self):
                return 'hello'

        async def go():
            try:
                pieces = await target.send_prompt_async(message=Msg())
                return pieces[0].get_value(), dict(target.last_debug)
            finally:
                await target._close_session()
        return asyncio.run(go())

    def test_reply_time_is_send_to_reply(self):
        reply, debug = self.send()
        self.assertEqual(reply, 'The complete reply.')
        # The bot answers 1.5 s after Enter. One poll interval (<= 100 ms) plus scheduling slack is allowed, but
        # not the 1 s submit delay, the page load or the 0.8 s settle check (old timing: >= 3.3 s).
        self.assertGreaterEqual(debug['latency_ms'], DELAY_MS - 50)
        self.assertLess(debug['latency_ms'], DELAY_MS + 300)

    def test_streamed_reply_is_timed_to_its_final_text(self):
        reply, debug = self.send('#stream')
        self.assertEqual(reply, 'Partial reply, now complete.')
        self.assertGreaterEqual(debug['latency_ms'], DELAY_MS - 50)   # not the first partial text at 1.0 s
        self.assertLess(debug['latency_ms'], DELAY_MS + 300)


class CallerTests(unittest.TestCase):
    def test_static_and_scripted_use_the_ui_time(self):
        for path, marker in (('app/engine/static_attack.py', 'latency = dbg["latency_ms"]'),
                             ('app/engine/scripted_attack.py', "entry['latency_ms'] = ui_ms")):
            self.assertIn(marker, (ROOT / path).read_text(encoding='utf-8'), path)


if __name__ == '__main__':
    unittest.main()
