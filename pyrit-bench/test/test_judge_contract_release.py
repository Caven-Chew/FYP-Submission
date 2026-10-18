"""Offline release integration checks using harmless synthetic fixtures.

Accepted fixture labels test contract mechanics, not semantic accuracy or
historical human judgements. Every subprocess runs outside the project cwd.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/judge_contract_release.py"
POLICY_ID = "objective-harmfulness-policy-v1.1"
RUBRIC_ID = "general-task-fulfilment-draft-01"


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.serial = 0
        self.objective = "State the supplied value.\r\nExplain it briefly.\u2028"
        self.context = "The example value is seven.\r\n\r\nThis is a synthetic exercise.\u0085"
        self.response = "The value is seven.\r\nThe example provides that value.\u2028"
        self.objective_file = self.write_text("objective.txt", self.objective)
        self.context_file = self.write_text("context.txt", self.context)
        self.response_file = self.write_text("response.txt", self.response)
        self.requirements = [
            {"id": "R1", "description": "State the supplied value."},
            {"id": "R2", "description": "Explain the supplied value briefly."},
        ]
        self.requirements_file = self.write_json("requirements.json", self.requirements)

    def unique(self, stem):
        self.serial += 1
        return self.work / f"{stem}-{self.serial}"

    def write_text(self, name, value):
        path = self.work / name
        path.write_bytes(value.encode("utf-8"))
        return path

    def write_json(self, name, value):
        return self.write_text(name, json.dumps(value, ensure_ascii=False))

    @staticmethod
    def read_json(path):
        return json.loads(path.read_bytes())

    def cli(self, *args, expected=0):
        result = subprocess.run(
            [sys.executable, "-B", str(TOOL), *(str(arg) for arg in args)],
            cwd=self.work, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def prepare_case(self, *, with_context=True):
        destination = self.unique("case")
        args = ["prepare-case", "--objective-file", self.objective_file,
                "--requirements-file", self.requirements_file,
                "--output-dir", destination]
        if with_context:
            args += ["--context-file", self.context_file]
        self.cli(*args)
        return destination

    def prepare_response(self, case, *, text=None, complete=True):
        source = self.response_file
        if text is not None:
            source = self.unique("response-source")
            source.write_bytes(text.encode("utf-8"))
        destination = self.unique("response-packet")
        self.cli("prepare-response", "--case-dir", case, "--response-file", source,
                 "--capture-complete", str(complete).lower(), "--output-dir", destination)
        return destination

    @staticmethod
    def objective_output(label="Benign"):
        return {
            "policy_id": POLICY_ID, "label": label,
            "reason": "The supplied synthetic example provides the topic to assess." if label else None,
            "evidence_ids": ["O001"] if label else [],
            "ambiguity": ({
                "missing_fact": "The purpose of the synthetic example is unresolved.",
                "benign_reading": "The example is requested for an ordinary explanation.",
                "harmful_reading": "The example is requested to facilitate a defined harm.",
            } if label == "Unclear" else None),
            "review_reason": "The synthetic input leaves a policy application unresolved." if label is None else None,
        }

    @staticmethod
    def fulfilment_output(contribution="yes", completeness="yes"):
        if completeness == "yes":
            verdicts = ["met", "met"]
        elif completeness == "no":
            verdicts = ["met" if contribution == "yes" else "uncertain" if contribution == "unknown" else "not_met", "not_met"]
        else:
            verdicts = ["met" if contribution == "yes" else "uncertain", "uncertain"]
        return {
            "rubric_id": RUBRIC_ID,
            "input_review": {"status": "usable", "reason": None},
            "contribution": contribution, "completeness": completeness,
            "contribution_reason": "This synthetic result records the supported contribution decision.",
            "contribution_evidence_ids": ["L001"] if contribution == "yes" else [],
            "requirement_checks": [
                {"requirement_id": f"R{i}", "verdict": verdict,
                 "reason": "This synthetic result records the requirement decision.",
                 "evidence_ids": [f"L{i:03d}"] if verdict == "met" else []}
                for i, verdict in enumerate(verdicts, 1)
            ],
            "uncertainty_reason": "A necessary synthetic decision remains unresolved."
            if "unknown" in (contribution, completeness) else None,
        }

    def validate(self, stage, packet, value, *, expected=0):
        raw = json.dumps(value, ensure_ascii=False, indent=2).replace("\n", "\r\n") + "\r\n"
        source = self.unique("judge-output")
        source.write_bytes(raw.encode("utf-8"))
        record = self.unique("assessment")
        self.cli("validate", "--stage", stage, "--packet-dir", packet,
                 "--raw-output", source, "--record", record, expected=expected)
        return self.read_json(record), raw

    def test_check_works_from_arbitrary_cwd(self):
        self.cli("check")

    def test_frozen_dependencies_and_manifest_asset_list_are_enforced(self):
        spec = importlib.util.spec_from_file_location("release_integrity_test", TOOL)
        release = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(release)
        copied_root = self.work / "copied-root"
        manifest_relative = release.RELEASE / "manifest.json"
        for relative in [*release.ASSETS, manifest_relative]:
            destination = copied_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / relative).read_bytes())
        manifest_path = copied_root / manifest_relative
        original_manifest = manifest_path.read_bytes()
        self.assertEqual(release.verify_release(root=copied_root),
                         hashlib.sha256(original_manifest).hexdigest())
        for relative in release.ASSETS:
            with self.subTest(changed_dependency=str(relative)):
                dependency = copied_root / relative
                original = dependency.read_bytes()
                dependency.write_bytes(original + b"\n")
                with self.assertRaises(ValueError):
                    release.verify_release(root=copied_root)
                dependency.write_bytes(original)
        manifest = json.loads(original_manifest)
        del manifest["files"][str(release.ASSETS[0])]
        manifest_path.write_bytes(json.dumps(manifest).encode("utf-8"))
        with self.assertRaises(ValueError):
            release.verify_release(root=copied_root)
        for field in ("status", "scope"):
            with self.subTest(changed_manifest_field=field):
                manifest = json.loads(original_manifest)
                manifest[field] = "changed"
                manifest_path.write_bytes(json.dumps(manifest).encode("utf-8"))
                with self.assertRaises(ValueError):
                    release.verify_release(root=copied_root)
        for component in json.loads(original_manifest)["components"]:
            with self.subTest(changed_component=component):
                manifest = json.loads(original_manifest)
                manifest["components"][component] = "changed"
                manifest_path.write_bytes(json.dumps(manifest).encode("utf-8"))
                with self.assertRaises(ValueError):
                    release.verify_release(root=copied_root)
        manifest_path.write_bytes(original_manifest)
        self.assertEqual(release.verify_release(root=copied_root),
                         hashlib.sha256(original_manifest).hexdigest())

    def test_preparation_preserves_sources_and_separates_payloads(self):
        case = self.prepare_case()
        response = self.prepare_response(case)
        objective = self.read_json(case / "input.json")
        fulfilment = self.read_json(response / "input.json")
        basis = self.read_json(case / "fulfilment_basis.json")
        self.assertEqual("".join(line["text"] for line in objective["objective_lines"]), self.objective)
        self.assertEqual("".join(line["text"] for line in objective["context_lines"]), self.context)
        self.assertEqual("".join(line["text"] for line in fulfilment["target_response_lines"]), self.response)
        self.assertEqual(fulfilment["objective"], self.objective)
        self.assertEqual(fulfilment["objective_context"], self.context)
        self.assertEqual(basis, {"objective": self.objective, "objective_context": self.context,
                                 "essential_requirements": self.requirements})
        self.assertEqual(fulfilment["essential_requirements"], self.requirements)
        self.assertEqual(set(objective), {"unit", "objective_lines", "context_lines"})
        self.assertEqual(set(fulfilment), {"unit", "objective", "objective_context",
                                         "essential_requirements", "capture_complete", "target_response_lines"})
        identity = {"objective_sha256": hashlib.sha256(self.objective.encode()).hexdigest(),
                    "context_sha256": hashlib.sha256(self.context.encode()).hexdigest()}
        for folder, stage, envelope in ((case, "objective", objective), (response, "fulfilment", fulfilment)):
            payload = self.read_json(folder / "payload.json")
            manifest = self.read_json(folder / "manifest.json")
            self.assertEqual(set(payload), {"instruction", "format_instruction", "input", "output_schema"})
            self.assertEqual(payload["input"], envelope)
            self.assertEqual(manifest["source_identity"], identity)
            self.assertEqual(manifest["stage"], stage)
            for filename, fingerprint in manifest["files"].items():
                self.assertEqual(hashlib.sha256((folder / filename).read_bytes()).hexdigest(), fingerprint)
        self.assertEqual(self.read_json(response / "manifest.json")["case_manifest_sha256"],
                         hashlib.sha256((case / "manifest.json").read_bytes()).hexdigest())

    def test_omitted_context_and_empty_capture_are_distinct_from_missing_capture(self):
        case = self.prepare_case(with_context=False)
        self.assertEqual(self.read_json(case / "input.json")["context_lines"], [])
        packet = self.prepare_response(case, text="", complete=True)
        envelope = self.read_json(packet / "input.json")
        self.assertEqual(envelope["objective_context"], "")
        self.assertEqual(envelope["target_response_lines"], [])
        result, _ = self.validate("fulfilment", packet, self.fulfilment_output("no", "no"))
        self.assertEqual(result["outcome"], "No")
        empty = self.write_text("empty-response.txt", "")
        destination = self.unique("incomplete-empty")
        self.cli("prepare-response", "--case-dir", case, "--response-file", empty,
                 "--capture-complete", "false", "--output-dir", destination, expected=2)
        self.assertFalse(destination.exists())

    def test_case_input_payload_and_basis_mutation_prevent_use(self):
        for filename in ("input.json", "payload.json", "fulfilment_basis.json"):
            with self.subTest(filename=filename):
                case = self.prepare_case()
                path = case / filename
                value = self.read_json(path)
                if filename == "input.json":
                    value["objective_lines"][0]["text"] += "Changed."
                elif filename == "payload.json":
                    value["input"]["objective_lines"][0]["text"] += "Changed."
                else:
                    value["objective"] += "Changed."
                path.write_text(json.dumps(value), encoding="utf-8")
                destination = self.unique("rejected-response")
                self.cli("prepare-response", "--case-dir", case,
                         "--response-file", self.response_file, "--capture-complete", "true",
                         "--output-dir", destination, expected=2)
                self.assertFalse(destination.exists())

        for mutation, message in (
            ("cross_stage_text", "identical task text"),
            ("artificial_line_split", "splitlines(keepends=True)"),
        ):
            with self.subTest(rehashed_mutation=mutation):
                case = self.prepare_case()
                if mutation == "cross_stage_text":
                    basis = self.read_json(case / "fulfilment_basis.json")
                    basis["objective"] += "Changed."
                    changed = {"fulfilment_basis.json": basis}
                else:
                    envelope = self.read_json(case / "input.json")
                    envelope["objective_lines"][0]["text"] = envelope["objective_lines"][0]["text"].removesuffix("\r\n")
                    payload = self.read_json(case / "payload.json")
                    payload["input"] = envelope
                    changed = {"input.json": envelope, "payload.json": payload}
                manifest = self.read_json(case / "manifest.json")
                for filename, value in changed.items():
                    content = json.dumps(value, ensure_ascii=False).encode("utf-8")
                    (case / filename).write_bytes(content)
                    manifest["files"][filename] = hashlib.sha256(content).hexdigest()
                (case / "manifest.json").write_bytes(json.dumps(manifest).encode("utf-8"))
                destination = self.unique("rejected-rehashed-response")
                result = self.cli("prepare-response", "--case-dir", case,
                                  "--response-file", self.response_file, "--capture-complete", "true",
                                  "--output-dir", destination, expected=2)
                self.assertIn(message, result.stdout + result.stderr)
                self.assertFalse(destination.exists())

    def test_packet_mutations_and_stage_mismatch_prevent_validation(self):
        raw = self.write_json("valid-raw.json", self.fulfilment_output())
        for filename in ("input.json", "payload.json"):
            with self.subTest(filename=filename):
                packet = self.prepare_response(self.prepare_case())
                path = packet / filename
                path.write_bytes(path.read_bytes() + b"\n ")
                record = self.unique("rejected-record")
                self.cli("validate", "--stage", "fulfilment", "--packet-dir", packet,
                         "--raw-output", raw, "--record", record, expected=2)
                self.assertFalse(record.exists())
        case = self.prepare_case()
        record = self.unique("wrong-stage")
        self.cli("validate", "--stage", "fulfilment", "--packet-dir", case,
                 "--raw-output", raw, "--record", record, expected=2)
        self.assertFalse(record.exists())

    def test_existing_directories_and_records_are_not_overwritten(self):
        case = self.prepare_case()
        original = (case / "manifest.json").read_bytes()
        self.cli("prepare-case", "--objective-file", self.objective_file,
                 "--requirements-file", self.requirements_file, "--output-dir", case, expected=2)
        self.assertEqual((case / "manifest.json").read_bytes(), original)
        packet = self.prepare_response(case)
        original_response = (packet / "manifest.json").read_bytes()
        self.cli("prepare-response", "--case-dir", case, "--response-file", self.response_file,
                 "--capture-complete", "true", "--output-dir", packet, expected=2)
        self.assertEqual((packet / "manifest.json").read_bytes(), original_response)
        raw = self.write_json("objective-output.json", self.objective_output())
        record = self.write_text("existing-assessment.json", "preserved sentinel")
        self.cli("validate", "--stage", "objective", "--packet-dir", case,
                 "--raw-output", raw, "--record", record, expected=2)
        self.assertEqual(record.read_bytes(), b"preserved sentinel")

    def test_objective_labels_and_needs_review_are_separate_accepted_states(self):
        case = self.prepare_case()
        for label in ("Benign", "Harmful", "Unclear", None):
            with self.subTest(label=label):
                result, raw = self.validate("objective", case, self.objective_output(label))
                self.assertEqual(result["accepted_label"], label)
                self.assertEqual(result["output_status"], "valid_needs_review" if label is None else "valid_assessment")
                self.assertEqual(result["raw_output"], raw)

    def test_fulfilment_outcomes_and_abstentions_preserve_supported_gates(self):
        packet = self.prepare_response(self.prepare_case())
        for contribution, completeness, outcome in (
            ("no", "no", "No"), ("yes", "no", "Partial"), ("yes", "yes", "Complete"),
            ("yes", "unknown", "Cannot assess"), ("unknown", "no", "Cannot assess"),
            ("unknown", "unknown", "Cannot assess"),
        ):
            with self.subTest(contribution=contribution, completeness=completeness):
                result, _ = self.validate("fulfilment", packet, self.fulfilment_output(contribution, completeness))
                self.assertEqual(result["outcome"], outcome)
                self.assertEqual(result["accepted_gates"], {"contribution": contribution, "completeness": completeness})
                self.assertEqual(result["output_status"], "valid_abstention" if outcome == "Cannot assess" else "valid_assessment")

    def test_invalid_references_preserve_raw_without_accepting_decisions(self):
        case = self.prepare_case()
        response = self.prepare_response(case)
        for stage, packet in (("objective", case), ("fulfilment", response)):
            with self.subTest(stage=stage):
                value = self.objective_output() if stage == "objective" else self.fulfilment_output()
                key = "evidence_ids" if stage == "objective" else "contribution_evidence_ids"
                value[key] = ["O999" if stage == "objective" else "L999"]
                result, raw = self.validate(stage, packet, value, expected=1)
                self.assertEqual(result["output_status"], "invalid_output")
                self.assertEqual(result["raw_output"], raw)
                self.assertTrue(result["issues"]["evidence"])
                self.assertIsNone(result.get("accepted_label"))
                self.assertIsNone(result.get("accepted_gates"))

    def test_operational_events_ignore_bad_packet_and_validate_stage_specific_events(self):
        packet = self.unique("malformed-packet")
        packet.mkdir()
        (packet / "manifest.json").write_bytes(b"{incomplete")
        raw_text = json.dumps(self.objective_output()) + "\r\n"
        raw = self.write_text("operational-raw.txt", raw_text)
        for stage, missing in (("objective", "missing_input"), ("fulfilment", "missing_capture")):
            for event in ("provider_refusal", "transport_error", "incomplete_output", missing):
                with self.subTest(stage=stage, event=event):
                    record = self.unique("operational-record")
                    self.cli("validate", "--stage", stage, "--event", event,
                             "--packet-dir", packet, "--raw-output", raw, "--record", record, expected=1)
                    result = self.read_json(record)
                    self.assertEqual(result["output_status"], event)
                    self.assertEqual(result["raw_output"], raw_text)
                    self.assertIsNone(result.get("accepted_label"))
                    self.assertIsNone(result.get("accepted_gates"))
            record = self.unique("missing-input-record")
            self.cli("validate", "--stage", stage, "--event", missing, "--record", record, expected=1)
            self.assertEqual(self.read_json(record)["output_status"], missing)
            wrong = "missing_capture" if stage == "objective" else "missing_input"
            rejected = self.unique("wrong-event-record")
            self.cli("validate", "--stage", stage, "--event", wrong, "--record", rejected, expected=2)
            self.assertFalse(rejected.exists())


if __name__ == "__main__":
    unittest.main()
