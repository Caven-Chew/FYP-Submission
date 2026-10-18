"""Regression tests for the per-run attacker/judge model record (Task 7 prep, I-1).

Run with:  .venv/bin/python -B test/test_run_model_record.py

PyRIT is stubbed and the chat-target class only records its constructor
arguments, so this makes no network or model calls. Flask and SQLAlchemy are
real, on a throwaway database.

What it guards:

  1. A run's scores could not be traced to a judge: new benchmarks take the
     attacker and judge from .env and stored nothing, so every page and export
     read "env default". Each start now records what it resolved.
  2. The record must not become an override. It lives under the reserved
     RESOLVED_MODELS key; the role builders read only ATTACKER / JUDGE, so a
     later resume still uses the models .env names then — and the change is
     flagged instead of silently replacing the first record.
  3. Legacy runs have no record and must say so, not borrow today's .env.
  4. No key value, inline key or private endpoint URL is stored.
"""
import sys, types, os, asyncio, tempfile, json, io, unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# A throwaway database, set BEFORE any app import (see test_workshop_preset.py).
os.environ["DATABASE_URL"] = "sqlite:///" + tempfile.mktemp(suffix=".db")

# Pin every model variable BEFORE the app loads .env: load_dotenv never
# overrides a variable that is already set, even to "", so the local .env
# cannot leak into these assertions. The key values are fake markers the tests
# search for; nothing is sent anywhere.
SECRET_OPENAI = "sk-test-SECRET-openai-000"
SECRET_VENICE = "vk-test-SECRET-venice-000"
PRIVATE_URL = "https://private-llm.example.internal/v1"
for var in ("JUDGE_PROVIDER", "JUDGE_MODEL", "JUDGE_API_KEY_ENV",
            "ATTACKER_PROVIDER", "ATTACKER_MODEL", "ATTACKER_API_KEY_ENV",
            "ATTACKER_BACKUP_MODEL", "LLM_ENDPOINT", "LLM_API_KEY",
            "ANTHROPIC_API_KEY", "JUDGE_VISION_MODEL"):
    os.environ[var] = ""
os.environ.update({
    "JUDGE_PROVIDER": "openai", "JUDGE_MODEL": "gpt-judge-rec",
    "OPENAI_API_KEY": SECRET_OPENAI,
    "ATTACKER_PROVIDER": "venice", "ATTACKER_MODEL": "qwen-attacker-rec",
    "VENICE_API_KEY": SECRET_VENICE,
    "ATTACKER_BACKUP_MODEL": "venice:qwen-backup-rec",
})


# --- stub out pyrit just enough that the engine modules import ---------------
def stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


class _Any:
    def __init__(self, *a, **k): pass
    def __getattr__(self, n): return _Any()


for mod in ["pyrit", "pyrit.models", "pyrit.datasets", "pyrit.prompt_target",
            "pyrit.prompt_converter", "pyrit.score", "pyrit.memory",
            "pyrit.executor", "pyrit.executor.attack", "pyrit.common",
            "pyrit.models.conversation_reference"]:
    stub(mod)


class _Msg(_Any):
    @staticmethod
    def from_prompt(prompt=None, role=None):
        return types.SimpleNamespace(prompt=prompt, role=role)


class _ChatTarget(_Any):
    """Records what the builder asked for; never connects."""
    def __init__(self, **kw): self.kw = kw


sys.modules["pyrit.models"].Message = _Msg
sys.modules["pyrit.prompt_target"].OpenAIChatTarget = _ChatTarget
for n in ("Base64Converter", "LeetspeakConverter", "ROT13Converter", "PersuasionConverter"):
    setattr(sys.modules["pyrit.prompt_converter"], n, _Any)

# create_app() and every run prune instance/artifacts against the CURRENT
# database. On a throwaway database every real capture folder looks orphaned,
# so point the folder somewhere disposable before anything calls prune().
import app.artifacts as ARTIFACTS  # noqa: E402
ARTIFACTS.ARTIFACTS_DIR = Path(tempfile.mkdtemp()) / "artifacts"

from app import create_app  # noqa: E402
from app.attacks import ATTACKS  # noqa: E402
from app.extensions import db  # noqa: E402
from app.models import Benchmark, Prompt, PromptStatus, Target  # noqa: E402
from app.settings import RESOLVED_MODELS_KEY  # noqa: E402
from app.engine import llms, runner, static_attack  # noqa: E402

STATIC = next(a for a in ATTACKS if a.get("kind") == "static" and a.get("scoring") == "leak")


class RunModelRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config["TESTING"] = True

    def setUp(self):
        self.ctx = self.app.app_context()
        self.ctx.push()
        db.drop_all()
        db.create_all()
        self.target = Target(name="record test", provider="openai", model_name="target-m")
        db.session.add(self.target)
        db.session.commit()
        self.built_judges = []
        self.env = patch.dict(os.environ, {"JUDGE_MODEL": "gpt-judge-rec"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        llms.llm_overrides.set({})
        db.session.remove()
        self.ctx.pop()

    # ---- helpers ---------------------------------------------------------
    def _bench(self, **kw):
        b = Benchmark(name=kw.pop("name", "rec"), target_id=self.target.id,
                      attack_slugs_json=json.dumps([STATIC["slug"]]), **kw)
        db.session.add(b)
        db.session.commit()
        return b

    def _patches(self):
        real_build_judge = llms.build_judge_llm

        def spy_judge():
            j = real_build_judge()
            self.built_judges.append(j)
            return j
        return [
            patch.object(llms, "ensure_pyrit_initialized", lambda: None),
            patch.object(runner, "build_judge_llm", spy_judge),
            patch.object(runner, "build_target", return_value=object()),
            patch.object(runner, "prune", lambda *a, **k: 0),
            patch.object(runner, "ATTACK_DELAY", 0),
            patch.object(static_attack, "load_dataset_prompts", AsyncMock(return_value=["seed"])),
            patch.object(static_attack, "send_prompt", AsyncMock(return_value=("a reply", None))),
            patch.object(static_attack, "judge_response", AsyncMock(return_value=("SCORE: 1", 1, False))),
            patch.object(static_attack, "REQUEST_DELAY", 0),
        ]

    def _run(self, coro_factory):
        ps = self._patches()
        for p in ps:
            p.start()
        try:
            asyncio.run(coro_factory())
        finally:
            for p in reversed(ps):
                p.stop()
        db.session.expire_all()

    # ---- 1. a new run records what it actually used -----------------------
    def test_new_run_records_resolved_attacker_and_judge(self):
        b = self._bench()
        self._run(lambda: runner._run_async(b.id))
        b = db.session.get(Benchmark, b.id)
        self.assertEqual(b.status, "completed")
        starts = b.resolved_models
        self.assertEqual(starts[0].get("judge_method"), "severity")
        self.assertEqual(len(starts), 1)
        s = starts[0]
        self.assertEqual(s["phase"], "run")
        self.assertEqual((s["judge"]["provider"], s["judge"]["model"]), ("openai", "gpt-judge-rec"))
        self.assertEqual(s["judge"]["key_env"], "OPENAI_API_KEY")
        self.assertEqual(s["judge"]["source"], "env")
        self.assertEqual((s["attacker"]["provider"], s["attacker"]["model"]),
                         ("venice", "qwen-attacker-rec"))
        self.assertEqual((s["attacker_backup"]["provider"], s["attacker_backup"]["model"]),
                         ("venice", "qwen-backup-rec"))
        self.assertTrue(s["at_utc"].endswith("+00:00"))
        # The record names the model the builder was actually given.
        self.assertEqual(self.built_judges[0].kw["model_name"], s["judge"]["model"])

    def test_summary_shows_real_names(self):
        b = self._bench()
        self._run(lambda: runner._run_async(b.id))
        b = db.session.get(Benchmark, b.id)
        text = b.llm_config_summary
        self.assertIn("attacker: venice/qwen-attacker-rec", text)
        self.assertIn("judge: openai/gpt-judge-rec", text)
        self.assertIn("attacker backup: venice/qwen-backup-rec", text)
        self.assertNotIn("not recorded", text)
        self.assertNotIn("changed", text)
        self.assertTrue(b.models_used["recorded"])

    # ---- 2. the record is not an override ---------------------------------
    def test_role_builders_ignore_the_record_key(self):
        cfg = {RESOLVED_MODELS_KEY: [{
            "phase": "run", "judge": {"provider": "anthropic", "model": "claude-recorded"},
            "attacker": {"provider": "anthropic", "model": "claude-recorded"}}]}
        llms.llm_overrides.set(cfg)
        with patch.object(llms, "ensure_pyrit_initialized", lambda: None), \
             patch.dict(os.environ, {"JUDGE_MODEL": "gpt-judge-now"}):
            judge = llms.build_judge_llm()
            attacker = llms.build_attacker_llm()
            self.assertEqual(judge.kw["model_name"], "gpt-judge-now")
            self.assertEqual(attacker.kw["model_name"], "qwen-attacker-rec")
            self.assertEqual(llms.resolve_role("JUDGE")["source"], "env")

    def test_resume_after_env_change_appends_and_flags(self):
        b = self._bench()
        self._run(lambda: runner._run_async(b.id))
        b = db.session.get(Benchmark, b.id)
        first = dict(b.resolved_models[0])
        b.status, b.completed_attacks_json = "interrupted", "[]"
        db.session.commit()
        with patch.dict(os.environ, {"JUDGE_MODEL": "gpt-judge-changed"}):
            self._run(lambda: runner._run_async(b.id))
        b = db.session.get(Benchmark, b.id)
        starts = b.resolved_models
        self.assertEqual(starts[0].get("judge_method"), "severity")
        self.assertEqual(len(starts), 2)
        self.assertEqual(starts[0], first, "the first start's record is never overwritten")
        self.assertEqual(starts[1]["phase"], "resume")
        self.assertEqual(starts[1]["judge"]["model"], "gpt-judge-changed")
        self.assertTrue(starts[1]["differs_from_first"])
        self.assertIn("judge: openai/gpt-judge-rec, then openai/gpt-judge-changed",
                      b.llm_config_summary)
        self.assertIn("models changed between starts", b.llm_config_summary)
        self.assertEqual(self.built_judges[-1].kw["model_name"], "gpt-judge-changed",
                         "resume uses the current .env, not the record")

    def test_retry_skip_and_rerun_each_record_a_start(self):
        b = self._bench(status="completed")
        failed = Prompt(benchmark_id=b.id, attack_slug=STATIC["slug"], sequence=0,
                        attempt=0, prompt_text="p", response="",
                        status=PromptStatus.SEND_ERROR, error="timeout")
        db.session.add(failed)
        db.session.commit()
        self._run(lambda: runner._rerun_prompt(b.id, failed.id, "completed"))
        self._run(lambda: runner._recover_async(b.id))
        self._run(lambda: runner._recover_async(b.id, retry_failed=False))
        b = db.session.get(Benchmark, b.id)
        self.assertEqual([s["phase"] for s in b.resolved_models],
                         ["rerun_prompt", "retry_failed", "skip_failed"])
        self.assertFalse(any(s.get("differs_from_first") for s in b.resolved_models))
        # The first start was not a fresh run, so the summary says what it misses.
        self.assertIn("first recorded at a prompt rerun", b.llm_config_summary)
        self.assertIn("earlier starts not recorded", b.llm_config_summary)

    # ---- 3. legacy runs stay honest ----------------------------------------
    def test_legacy_run_shows_not_recorded(self):
        b = self._bench(llm_config_json="{}")
        self.assertEqual(b.llm_config_summary,
                         "attacker: env default (not recorded) · judge: env default (not recorded)")
        self.assertFalse(b.models_used["recorded"])
        self.assertNotIn("gpt-judge-rec", b.llm_config_summary, "never borrow today's .env")
        with_override = self._bench(name="legacy-override", llm_config_json=json.dumps(
            {"JUDGE": {"provider": "anthropic", "model": "claude-x"},
             "ATTACKER": {"provider": "anthropic"}}))
        self.assertIn("judge: anthropic/claude-x", with_override.llm_config_summary)
        self.assertIn("attacker: anthropic (model not recorded)", with_override.llm_config_summary)

    def test_legacy_run_resumed_later_is_labelled_partial(self):
        from datetime import datetime
        b = self._bench(status="interrupted", started_at=datetime(2026, 9, 1))
        self._run(lambda: runner._run_async(b.id))
        b = db.session.get(Benchmark, b.id)
        self.assertEqual(b.resolved_models[0]["phase"], "resume")
        self.assertIn("first recorded at a Resume", b.llm_config_summary)

    # ---- 4. no secrets ---------------------------------------------------------
    def test_no_secret_is_stored(self):
        b = self._bench(
            llm_config_json=json.dumps({"JUDGE": {"provider": "profllm", "model": "prof-m",
                                                  "api_key": "inline-SECRET-judge"}}),
            attacker_roster_json=json.dumps([{"provider": "venice", "model": "r1",
                                              "api_key": "roster-SECRET"}]))
        llms.llm_overrides.set(b.llm_config)
        with patch.dict(os.environ, {"LLM_ENDPOINT": PRIVATE_URL}), \
             patch("app.settings.ATTACKER_BACKUP_MODEL", "sk-looks-like-a-key"):
            runner._record_resolved_models(b, "run")
        db.session.commit()
        record = json.dumps(b.resolved_models)
        for secret in (SECRET_OPENAI, SECRET_VENICE, "inline-SECRET-judge", "roster-SECRET",
                       PRIVATE_URL, "sk-looks-like-a-key", "api.venice.ai", "api.openai.com"):
            self.assertNotIn(secret, record)
        self.assertNotIn('"api_key"', record)
        s = b.resolved_models[0]
        self.assertEqual((s["judge"]["provider"], s["judge"]["model"]), ("profllm", "prof-m"))
        self.assertTrue(s["judge"]["inline_key"])
        self.assertEqual(s["judge"]["endpoint_env"], "LLM_ENDPOINT")
        self.assertEqual(s["judge"]["source"], "benchmark override")
        self.assertEqual(s["attacker_roster"], ["venice/r1"])
        self.assertFalse(s["attacker_backup"]["usable"])
        self.assertIn("attacker backup: unsupported spec", b.llm_config_summary)
        self.assertIn("attacker roster: venice/r1", b.llm_config_summary)

    # ---- 5. run page, comparison, CSV and PDF all carry it -------------------
    def test_page_compare_csv_and_pdf_show_the_record(self):
        new = self._bench(name="recorded-run")
        self._run(lambda: runner._run_async(new.id))
        legacy = self._bench(name="legacy-run", llm_config_json="{}")
        client = self.app.test_client()

        page = client.get(f"/benchmarks/{new.id}").get_data(as_text=True)
        self.assertIn("judge: openai/gpt-judge-rec", page)

        compare = client.get(f"/benchmarks/compare?ids={new.id},{legacy.id}").get_data(as_text=True)
        self.assertIn("judge: openai/gpt-judge-rec", compare)
        self.assertIn("env default (not recorded)", compare)

        import csv
        rows = list(csv.reader(io.StringIO(client.get(f"/benchmarks/{new.id}/csv").get_data(as_text=True))))
        header = rows[0]
        self.assertEqual(header[24:26], ["judge_model", "models_used"])
        self.assertEqual(header[:3], ["benchmark", "target", "architecture"], "older columns keep their place")
        self.assertEqual(rows[1][header.index("judge_model")], "openai/gpt-judge-rec")
        self.assertIn("attacker: venice/qwen-attacker-rec", rows[1][header.index("models_used")])

        from pypdf import PdfReader
        pdf = client.get(f"/benchmarks/{new.id}/pdf")
        self.assertEqual(pdf.status_code, 200)
        text = " ".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf.data)).pages)
        self.assertIn("openai/gpt-judge-rec", " ".join(text.split()))

    def test_backup_spec_refactor_keeps_builder_behaviour(self):
        with patch.object(llms, "ensure_pyrit_initialized", lambda: None):
            self.assertIsNone(llms.build_converter_llm_from_spec(""))
            self.assertIsNone(llms.build_converter_llm_from_spec("unknown:model"))
            built = llms.build_converter_llm_from_spec("venice:", frame_rewrite=False)
            self.assertEqual(built.kw["model_name"], llms._ROLE_DEFAULT_MODEL["venice"])
            self.assertEqual(built.kw["api_key"], SECRET_VENICE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
