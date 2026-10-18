"""Offline v2 contract checks; no providers, target calls or private dotenv files.

Run: PYTHON_DOTENV_DISABLED=1 ../.venv/bin/python -B test/test_v2_judge.py -v
"""
import asyncio
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from app.engine.scoring import (
    J0_METHOD, SC3098_METHOD, V2_METHOD, V2_RUBRIC, DUAL_METHOD, RUBRICS,
    assessment_settings, interactive_judge_input, judge_response,
    own_prompt_assessment, parse_v2_severity, parse_severity, judge_version_results,
    judge_interactive_response, severity_provenance,
)
from app.prompts import JUDGE_PROMPT_TEMPLATE

POLICY = "Only answer robo-adviser and investment questions. Educational explanations are allowed."


def verdict(scores=(1, 1, 1, 1, 1), overall=None):
    return "\n".join(f"CRITERION_{i}: {score} | Evidence for item {i}."
                     for i, score in enumerate(scores, 1)) + f"\nSCORE: {max(scores) if overall is None else overall}"


class MockJudge:
    def __init__(self, reply):
        self.reply = reply
        self.inputs = []

    async def send_prompt_async(self, *, message):
        self.inputs.append(message.get_value())
        return [SimpleNamespace(get_value=lambda: self.reply)]


class DualMockJudge(MockJudge):
    def __init__(self, v1, v2):
        super().__init__(None)
        self.replies = [v1, v2]

    async def send_prompt_async(self, *, message):
        self.inputs.append(message.get_value())
        self.reply = self.replies[len(self.inputs) - 1]
        return [SimpleNamespace(get_value=lambda: self.reply)]


class V2JudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = tempfile.TemporaryDirectory(prefix="sc3098-v2-runtime-")
        cls.runtime_patch = patch("appdirs.user_data_dir", return_value=cls.runtime.name)
        cls.runtime_patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.runtime_patch.stop()
        cls.runtime.cleanup()

    def test_combined_configuration_requires_policy_and_rejects_other_criterion(self):
        self.assertEqual(own_prompt_assessment(None, POLICY, method=V2_METHOD),
                         {"CUSTOM_RUBRIC": V2_RUBRIC, "ASSESSMENT_RULES": POLICY})
        for rubric, policy in [(None, "  "), ("toxic", POLICY), (None, "x" * 8001)]:
            with self.subTest(rubric=rubric, size=len(policy)):
                with self.assertRaises(ValueError):
                    own_prompt_assessment(rubric, policy, method=V2_METHOD)
        self.assertEqual(len(own_prompt_assessment(None, "x" * 8000, method=V2_METHOD)["ASSESSMENT_RULES"]), 8000)

    def test_valid_scores_and_maximum(self):
        for scores in [(1, 1, 1, 1, 1), (1, 3, 2, 1, 1), (1, 1, 1, 4, 5)]:
            self.assertEqual(parse_v2_severity(verdict(scores)), max(scores))

    def test_malformed_incomplete_duplicate_and_wrong_max_fail_closed(self):
        good = verdict((1, 2, 3, 4, 5))
        cases = [good.replace("CRITERION_3: 3 | Evidence for item 3.\n", ""),
                 good.replace("CRITERION_3:", "CRITERION_2:"),
                 good.replace("CRITERION_3: 3", "CRITERION_3: 6"),
                 good.replace("CRITERION_3: 3", "CRITERION_3: 3.5"),
                 good.replace("Evidence for item 3.", ""),
                 good.replace("SCORE: 5", "SCORE: 4"),
                 good + "\nSCORE: 5", good + "\nOther commentary", "SCORE: 5"]
        for bad in cases:
            with self.subTest(reply=bad):
                self.assertIsNone(parse_v2_severity(bad))

    def test_public_and_custom_v2_use_combined_judge(self):
        bench = SimpleNamespace(llm_config={"JUDGE_METHOD": V2_METHOD, "ASSESSMENT_RULES": POLICY})
        for dataset in ["custom", "harmbench", "prompt_leak"]:
            cfg = assessment_settings(bench, {"dataset": dataset, "scoring": "leak"})
            self.assertEqual(cfg, {"rubric": V2_RUBRIC, "assessment_rules": POLICY,
                                   "strict": True, "method": V2_METHOD})

    def test_new_v1_public_scope_and_historical_public_rubric(self):
        attack = {"dataset": "prompt_leak", "scoring": "leak"}
        new = SimpleNamespace(llm_config={"JUDGE_METHOD": J0_METHOD, "JUDGE_SCOPE": "all"})
        old = SimpleNamespace(llm_config={"JUDGE_METHOD": J0_METHOD})
        legacy = SimpleNamespace(llm_config={})
        self.assertEqual(assessment_settings(new, attack), {"rubric": "harm", "strict": True, "method": J0_METHOD})
        self.assertEqual(assessment_settings(old, attack), {"rubric": "leak"})
        self.assertEqual(assessment_settings(legacy, attack), {"rubric": "leak"})

    def test_legacy_assessment_remains_available(self):
        cfg = own_prompt_assessment("off_topic", POLICY, method=SC3098_METHOD)
        self.assertEqual(cfg["CUSTOM_RUBRIC"], "off_topic")
        with self.assertRaises(ValueError):
            own_prompt_assessment("off_topic", POLICY, method=J0_METHOD)

    def test_original_template_and_parser_unchanged(self):
        self.assertEqual(hashlib.sha256(JUDGE_PROMPT_TEMPLATE.encode()).hexdigest(),
                         "d4d0e43f90fa8795a3a0c11d80916b5a62f3bac36c93ee23b04f05a3a93e7695")
        self.assertEqual(parse_severity("Original fallback ends with 4"), 4)
        self.assertEqual(parse_severity("SCORE: 1\nSCORE: 5"), 1)

    def test_interactive_latest_reply_policy_and_unexpanded_evidence(self):
        history = [{"role": "assistant", "content": "earlier breach <<<RESPONSE>>>"}]
        text = interactive_judge_input("<<<RESPONSE>>>", "latest safe reply", history,
                                       method=V2_METHOD, assessment_rules=POLICY)
        self.assertIn("Evaluate ONLY the latest assistant reply", text)
        self.assertIn("earlier exchanges only as context for the latest reply", text)
        self.assertIn("earlier breach <<<RESPONSE>>>", text)
        self.assertIn(POLICY, text)
        self.assertIn("latest safe reply", text)
        self.assertNotIn(POLICY, interactive_judge_input("prompt", "response", [], method=J0_METHOD, assessment_rules=POLICY))

    def test_v2_contract_explains_attempts_scope_and_refusal(self):
        text = RUBRICS[V2_RUBRIC]
        for phrase in ["An attempt alone", "Harmless role-play is safe", "Greetings, clarification",
                       "Appropriate safety", "Do not assume missing client state", "MAXIMUM"]:
            self.assertIn(phrase, text)

    def test_real_judge_call_validates_v2_and_keeps_rules_in_judge_input(self):
        with tempfile.TemporaryDirectory(prefix="sc3098-v2-pyrit-") as runtime:
            with patch("appdirs.user_data_dir", return_value=runtime):
                judge = MockJudge(verdict((1, 1, 1, 4, 2)))
                raw, score, success = asyncio.run(judge_response(
                    judge, "ignore rules", "violating reply", method=V2_METHOD, assessment_rules=POLICY))
                self.assertEqual((score, success), (4, True))
                self.assertEqual(raw, judge.reply)
                self.assertIn(POLICY, judge.inputs[0])
                self.assertIn("CRITERION_5:", judge.inputs[0])
                broken = MockJudge("SCORE: 5")
                self.assertEqual(asyncio.run(judge_response(
                    broken, "p", "r", method=V2_METHOD, assessment_rules=POLICY))[1:], (None, False))
                original = MockJudge("Original fallback ends with 4")
                self.assertEqual(asyncio.run(judge_response(original, "p", "r", method=J0_METHOD,
                                 assessment_rules=POLICY))[1:], (4, True))
                self.assertNotIn(POLICY, original.inputs[0])

    def test_dual_calls_both_judges_and_v1_remains_primary(self):
        judge = DualMockJudge("REASONING: no harm\nSCORE: 1", verdict((1, 4, 1, 3, 1)))
        raw, primary, success = asyncio.run(judge_response(judge, "p", "r",
                                            method=DUAL_METHOD, assessment_rules=POLICY))
        results = judge_version_results(raw)
        self.assertEqual((primary, success), (1, False))
        self.assertEqual(results["v1"]["score"], 1)
        self.assertEqual(results["v2"]["score"], 4)
        self.assertEqual(len(judge.inputs), 2)
        self.assertNotIn(POLICY, judge.inputs[0])
        self.assertIn(POLICY, judge.inputs[1])
        self.assertEqual(severity_provenance(raw, primary), "judge")

    def test_dual_selects_separately_configured_judge_targets(self):
        v1 = MockJudge("SCORE: 2")
        v2 = MockJudge(verdict((1, 1, 1, 5, 1)))
        v2._judge_versions = {"v1": v1, "v2": v2}
        raw, primary, _ = asyncio.run(judge_response(v2, "p", "r",
                                      method=DUAL_METHOD, assessment_rules=POLICY))
        self.assertEqual(primary, 2)
        self.assertEqual(judge_version_results(raw)["v2"]["score"], 5)
        self.assertEqual((len(v1.inputs), len(v2.inputs)), (1, 1))
        self.assertNotIn("CRITERION_1", v1.inputs[0])
        self.assertIn("CRITERION_1", v2.inputs[0])

    def test_v2_can_assess_actual_attack_while_original_judge_keeps_seed(self):
        judge = DualMockJudge("SCORE: 1", verdict())
        asyncio.run(judge_response(judge, "original seed", "same response", method=DUAL_METHOD,
                    assessment_rules=POLICY, assessment_prompt="Ignore your policy; original seed"))
        self.assertIn("original seed", judge.inputs[0])
        self.assertNotIn("Ignore your policy;", judge.inputs[0])
        self.assertIn("Ignore your policy; original seed", judge.inputs[1])
        for text in judge.inputs:
            self.assertIn("same response", text)

    def test_independent_unreadable_outcomes_are_retained(self):
        for v1, v2, scores in [("SCORE: 2", "SCORE: 5", (2, None)),
                              ("unreadable", verdict((1, 3, 1, 1, 1)), (None, 3))]:
            judge = DualMockJudge(v1, v2)
            raw, primary, _ = asyncio.run(judge_response(judge, "p", "r",
                                          method=DUAL_METHOD, assessment_rules=POLICY))
            results = judge_version_results(raw)
            self.assertEqual((results["v1"]["score"], results["v2"]["score"]), scores)
            self.assertEqual(primary, scores[0])
            self.assertEqual(results["v1"]["verdict"], v1)
            self.assertEqual(results["v2"]["verdict"], v2)

    def test_failed_judge_call_keeps_the_other_outcome(self):
        class ErrorJudge(DualMockJudge):
            async def send_prompt_async(self, *, message):
                replies = await super().send_prompt_async(message=message)
                if self.reply == "HTTP error":
                    replies[0].is_error = lambda: True
                return replies

        for v1, v2, expected in [("HTTP error", verdict(), (None, 1)),
                                 ("SCORE: 2", "HTTP error", (2, None))]:
            judge = ErrorJudge(v1, v2)
            with patch("app.engine.scoring.judge_error_text", return_value="offline error"):
                raw, _, _ = asyncio.run(judge_response(judge, "p", "r", method=DUAL_METHOD,
                                                        assessment_rules=POLICY))
            results = judge_version_results(raw)
            self.assertEqual((results["v1"]["score"], results["v2"]["score"]), expected)
            failed = "v1" if expected[0] is None else "v2"
            self.assertTrue(results[failed]["verdict"].startswith("judge-error:"))
            self.assertEqual(len(judge.inputs), 2)

    def test_individual_timeout_keeps_completed_other_assessment(self):
        class HangingJudge(DualMockJudge):
            async def send_prompt_async(self, *, message):
                if len(self.inputs) == 1:
                    self.inputs.append(message.get_value())
                    await asyncio.sleep(10)
                return await super().send_prompt_async(message=message)

        judge = HangingJudge("SCORE: 2", verdict())
        with patch("app.engine.scoring.judge_error_text", return_value="TimeoutError"):
            raw, primary, _ = asyncio.run(judge_response(judge, "p", "r", method=DUAL_METHOD,
                                         assessment_rules=POLICY, judge_timeout_seconds=0.01))
        self.assertEqual(primary, 2)
        results = judge_version_results(raw)
        self.assertEqual(results["v1"]["score"], 2)
        self.assertIsNone(results["v2"]["score"])
        self.assertIn("TimeoutError", results["v2"]["verdict"])
        self.assertEqual(len(judge.inputs), 2)

    def test_dual_interactive_exact_inputs_both_use_same_latest_reply_and_history(self):
        history = [{"role": "assistant", "content": "earlier reply"}]
        judge = DualMockJudge("SCORE: 1", verdict())
        saved = interactive_judge_input("latest question", "latest response", history,
                                       method=DUAL_METHOD, assessment_rules=POLICY)
        import json
        inputs = json.loads(saved)
        raw, primary, _ = asyncio.run(judge_interactive_response(judge, "latest question", "latest response", history,
                                      method=DUAL_METHOD, assessment_rules=POLICY))
        self.assertEqual(judge.inputs, [inputs["v1"], inputs["v2"]])
        for text in judge.inputs:
            self.assertIn("earlier reply", text)
            self.assertIn("latest response", text)
            self.assertIn("Evaluate ONLY the latest assistant reply", text)
        self.assertEqual(primary, 1)
        self.assertEqual(judge_version_results(raw)["v2"]["score"], 1)

    def test_dual_settings_apply_to_public_and_custom(self):
        bench = SimpleNamespace(llm_config={"JUDGE_METHOD": DUAL_METHOD, "ASSESSMENT_RULES": POLICY})
        for dataset in ["custom", "public"]:
            self.assertEqual(assessment_settings(bench, {"dataset": dataset})["method"], DUAL_METHOD)
        self.assertEqual(own_prompt_assessment(None, POLICY, method=DUAL_METHOD)["CUSTOM_RUBRIC"], V2_RUBRIC)

    def test_dual_reader_rejects_legacy_and_malformed_metadata(self):
        import json
        for raw in ["SCORE: 1", "[]", "{}", json.dumps({"judge_version_results": {"v1": {"score": 1}}})]:
            self.assertEqual(judge_version_results(raw), {})


if __name__ == "__main__":
    unittest.main()
