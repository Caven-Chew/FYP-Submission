"""Offline objective-harmfulness contract draft; no clients, calls or writes."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
HELPER_PATH = BASE.parent / "fulfilment/validator.py"
HELPER_SPEC = importlib.util.spec_from_file_location("fyp_general_contract_helpers", HELPER_PATH)
if HELPER_SPEC is None or HELPER_SPEC.loader is None:
    raise ImportError("cannot load general-contract helpers")
HELPERS = importlib.util.module_from_spec(HELPER_SPEC)
HELPER_SPEC.loader.exec_module(HELPERS)

check_shape = HELPERS.check_shape
strict_loads = HELPERS.strict_loads
substantive = HELPERS.substantive
digest = HELPERS.digest

CONTRACT_ID = "objective-harmfulness-contract-draft-01"
VALIDATOR_ID = "objective-harmfulness-validator-draft-01"
POLICY_ID = "objective-harmfulness-policy-v1.1"
POLICY_INSTRUCTION_SHA256 = "3025a4944be1fba7f6279486b7373c9f189e8569b972f9dff2859193ba332e7a"
POLICY_INSTRUCTION_PATH = BASE / "policy_instruction.txt"
FORMAT_INSTRUCTION_PATH = BASE / "format_instruction.txt"
OUTPUT_SCHEMA = json.loads((BASE / "schema.json").read_text())
INPUT_SCHEMA = json.loads((BASE / "input_schema.json").read_text())
FAILURE_EVENTS = {"provider_refusal", "transport_error", "incomplete_output", "missing_input"}


def file_hash(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def split_lines(value: str) -> list[str]:
    return value.splitlines(keepends=True) if value else []


def prepare_input(objective: str, context: str = "") -> dict:
    if not isinstance(objective, str) or not isinstance(context, str):
        raise TypeError("objective and context must be strings")
    return {
        "unit": "objective",
        "objective_lines": [{"id": f"O{i:03d}", "text": text} for i, text in enumerate(split_lines(objective), 1)],
        "context_lines": [{"id": f"C{i:03d}", "text": text} for i, text in enumerate(split_lines(context), 1)],
    }


def is_single_source_line(text: str) -> bool:
    # Match prepare_input(): every boundary recognised by splitlines() starts
    # another source-line item. A trailing recognised terminator is retained
    # in its one item, including a blank line such as "\u2028".
    return len(text.splitlines(keepends=True)) <= 1


def input_issues(envelope) -> list[str]:
    errors = check_shape(envelope, INPUT_SCHEMA)
    if errors:
        return errors
    if not any(line["text"].strip() for line in envelope["objective_lines"]):
        errors.append("objective is blank")
    for prefix, lines in (("O", envelope["objective_lines"]), ("C", envelope["context_lines"])):
        for index, line in enumerate(lines, 1):
            if line["id"] != f"{prefix}{index:03d}":
                errors.append(f"{prefix} line {index}: canonical ID required")
            if not is_single_source_line(line["text"]):
                errors.append(f"{prefix} line {index}: contains more than one source line")
    return errors


def build_output_schema(envelope: dict) -> dict:
    errors = input_issues(envelope)
    if errors:
        raise ValueError("invalid input: " + "; ".join(errors))
    schema = copy.deepcopy(OUTPUT_SCHEMA)
    allowed = [line["id"] for key in ("objective_lines", "context_lines") for line in envelope[key] if line["text"].strip()]
    evidence = schema["properties"]["evidence_ids"]
    if allowed:
        evidence["items"]["enum"] = allowed
    else:
        evidence["maxItems"] = 0
    return schema


def assess(raw_output, envelope, *, event="completed", event_detail=None) -> dict:
    record = {
        "contract_id": CONTRACT_ID, "validator_id": VALIDATOR_ID, "policy_id": POLICY_ID,
        "schema_id": OUTPUT_SCHEMA["$id"], "validator_sha256": file_hash(Path(__file__)),
        "dependency_validator_sha256": file_hash(HELPER_PATH), "schema_sha256": file_hash(BASE / "schema.json"),
        "input_schema_sha256": file_hash(BASE / "input_schema.json"), "policy_instruction_sha256": file_hash(POLICY_INSTRUCTION_PATH),
        "format_instruction_sha256": file_hash(FORMAT_INSTRUCTION_PATH), "event": event, "event_detail": event_detail,
        "output_status": None, "raw_output": raw_output, "parsed_output": None, "accepted_label": None,
        "issues": {key: [] for key in ("input", "json", "schema", "logic", "evidence")},
    }
    issues = record["issues"]
    if not isinstance(event, str) or (event_detail is not None and not isinstance(event_detail, str)):
        issues["input"].append("event must be a string and event_detail a string or null")
        record["output_status"] = "invalid_input"
        return record
    if event in FAILURE_EVENTS:
        record["output_status"] = event
        return record
    if event != "completed":
        issues["input"].append("unknown execution event")
        record["output_status"] = "invalid_input"
        return record
    issues["input"] = input_issues(envelope)
    if record["policy_instruction_sha256"] != POLICY_INSTRUCTION_SHA256:
        issues["input"].append("policy instruction hash does not match frozen v1.1")
    if record["format_instruction_sha256"] is None:
        issues["input"].append("format instruction is unavailable")
    if issues["input"]:
        record["output_status"] = "invalid_input"
        return record
    bound_schema = build_output_schema(envelope)
    record["input_sha256"] = digest(json.dumps(envelope, ensure_ascii=False, sort_keys=True).encode())
    record["bound_schema_sha256"] = digest(json.dumps(bound_schema, sort_keys=True).encode())
    if not isinstance(raw_output, str):
        issues["json"].append("judge output must be a JSON text string")
    else:
        try:
            record["parsed_output"] = strict_loads(raw_output)
        except (ValueError, RecursionError) as exc:
            issues["json"].append(str(exc))
    if issues["json"]:
        record["output_status"] = "invalid_json"
        return record
    output = record["parsed_output"]
    issues["schema"] = check_shape(output, OUTPUT_SCHEMA)
    if issues["schema"]:
        record["output_status"] = "invalid_output"
        return record
    allowed = {line["id"] for key in ("objective_lines", "context_lines") for line in envelope[key] if line["text"].strip()}
    evidence_ids = output["evidence_ids"]
    if len(evidence_ids) != len(set(evidence_ids)):
        issues["evidence"].append("duplicate evidence IDs")
    for evidence_id in evidence_ids:
        if evidence_id not in allowed:
            issues["evidence"].append("disallowed, blank or non-individual evidence ID: " + evidence_id)
    label = output["label"]
    if label is None:
        if output["reason"] is not None:
            issues["logic"].append("needs_review requires reason=null")
        if output["ambiguity"] is not None:
            issues["logic"].append("needs_review requires ambiguity=null")
        if not substantive(output["review_reason"]):
            issues["logic"].append("needs_review requires a substantive review_reason")
    else:
        if not substantive(output["reason"]):
            issues["logic"].append("classified assessment requires a substantive reason")
        if not evidence_ids:
            issues["evidence"].append("classified assessment requires evidence")
        if output["review_reason"] is not None:
            issues["logic"].append("classified assessment requires review_reason=null")
        if label in {"Benign", "Harmful"} and output["ambiguity"] is not None:
            issues["logic"].append("Benign/Harmful requires ambiguity=null")
        if label == "Unclear":
            ambiguity = output["ambiguity"]
            if not isinstance(ambiguity, dict):
                issues["logic"].append("Unclear requires ambiguity details")
            else:
                for field in ("missing_fact", "benign_reading", "harmful_reading"):
                    if not substantive(ambiguity.get(field)):
                        issues["logic"].append(f"Unclear ambiguity {field} is blank or placeholder")
    if issues["logic"] or issues["evidence"]:
        record["output_status"] = "invalid_output"
        return record
    record["accepted_label"] = label
    record["output_status"] = "valid_needs_review" if label is None else "valid_assessment"
    return record
