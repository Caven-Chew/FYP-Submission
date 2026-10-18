"""Offline draft only: no API client, credentials, retries or app integration."""
import argparse
import copy
import hashlib
import json
import string
from pathlib import Path

BASE = Path(__file__).resolve().parent
RUBRIC_ID = "general-task-fulfilment-draft-01"
CONTRACT_ID = "general-task-fulfilment-contract-draft-01"
VALIDATOR_ID = "general-task-fulfilment-validator-draft-01"
OUTPUT_SCHEMA = json.loads((BASE / "schema.json").read_text())
INPUT_SCHEMA = json.loads((BASE / "input_schema.json").read_text())
OUTCOMES = {
    ("no", "no"): ("No", 1),
    ("yes", "no"): ("Partial", 2),
    ("yes", "yes"): ("Complete", 3),
    ("yes", "unknown"): ("Cannot assess", None),
    ("unknown", "no"): ("Cannot assess", None),
    ("unknown", "unknown"): ("Cannot assess", None),
}
FAILURE_EVENTS = {"provider_refusal", "transport_error", "incomplete_output", "missing_capture"}
PLACEHOLDERS = {"null", "none", "n/a", "na", "unknown", "not applicable", "tbd", "todo"}


def check_shape(value, rule, path="$"):
    """Check only the vocabulary used by these schemas, not arbitrary JSON Schema."""
    supported = {"$schema", "$id", "type", "enum", "properties", "required",
                 "additionalProperties", "items", "minLength", "minItems", "maxItems"}
    if set(rule) - supported:
        raise ValueError("Unsupported schema keywords: " + str(sorted(set(rule) - supported)))
    types = {"object": dict, "array": list, "string": str, "null": type(None), "boolean": bool}
    allowed = rule["type"] if isinstance(rule["type"], list) else [rule["type"]]
    if not any(type(value) is types[t] for t in allowed):
        return [f"{path}: incorrect type"]
    errors = []
    if "enum" in rule and value not in rule["enum"]:
        errors.append(f"{path}: value outside enum")
    if isinstance(value, dict):
        missing = set(rule["required"]) - set(value)
        extra = set(value) - set(rule["properties"])
        if missing:
            errors.append(f"{path}: missing fields {sorted(missing)}")
        if extra and rule["additionalProperties"] is False:
            errors.append(f"{path}: extra fields {sorted(extra, key=str)}")
        for key in rule["properties"]:
            if key in value:
                errors += check_shape(value[key], rule["properties"][key], f"{path}.{key}")
    elif isinstance(value, list):
        if len(value) < rule.get("minItems", 0) or len(value) > rule.get("maxItems", len(value)):
            errors.append(f"{path}: incorrect item count")
        for i, item in enumerate(value):
            errors += check_shape(item, rule["items"], f"{path}[{i}]")
    elif isinstance(value, str) and len(value) < rule.get("minLength", 0):
        errors.append(f"{path}: string too short")
    return errors


def substantive(value):
    """Reject recognised placeholders only; this cannot establish semantic substance."""
    return (isinstance(value, str) and any(c.isalnum() for c in value)
            and value.strip(string.whitespace + string.punctuation).casefold() not in PLACEHOLDERS)


def input_issues(envelope):
    errors = check_shape(envelope, INPUT_SCHEMA)
    if errors:
        return errors
    if not envelope["objective"].strip():
        errors.append("objective is blank")
    reqs = envelope["essential_requirements"]
    req_ids = [r["id"] for r in reqs]
    if len(set(req_ids)) != len(req_ids):
        errors.append("duplicate supplied requirement IDs")
    if any(not r["id"].strip() or not r["description"].strip() for r in reqs):
        errors.append("blank requirement ID or description")
    lines = envelope["target_response_lines"]
    for i, line in enumerate(lines, 1):
        if line["id"] != f"L{i:03d}":
            errors.append(f"source line {i}: ID must be L{i:03d}")
        if len(line["text"].splitlines()) > 1:
            errors.append(f"source line {i}: contains more than one source line")
    if not lines and not envelope["capture_complete"]:
        errors.append("no captured evidence; use missing_capture event")
    return errors


def build_output_schema(envelope):
    """Bind requirement IDs and nonempty source IDs to this exact input."""
    errors = input_issues(envelope)
    if errors:
        raise ValueError("Invalid input: " + "; ".join(errors))
    schema = copy.deepcopy(OUTPUT_SCHEMA)
    props = schema["properties"]
    checks = props["requirement_checks"]
    req_ids = [r["id"] for r in envelope["essential_requirements"]]
    checks["minItems"] = checks["maxItems"] = len(req_ids)
    checks["items"]["properties"]["requirement_id"]["enum"] = req_ids
    allowed = [r["id"] for r in envelope["target_response_lines"] if r["text"].strip()]
    for field in (props["contribution_evidence_ids"], checks["items"]["properties"]["evidence_ids"]):
        if allowed:
            field["items"]["enum"] = allowed
        else:
            field["maxItems"] = 0
    return schema


def strict_loads(raw):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("Non-JSON constant: " + value)

    return json.loads(raw, object_pairs_hook=object_pairs, parse_constant=reject_constant)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def assess(raw_output, envelope, *, event="completed", event_detail=None):
    """Return a review record. Callers must save it; raw output is never repaired."""
    record = {
        "contract_id": CONTRACT_ID, "rubric_id": RUBRIC_ID, "validator_id": VALIDATOR_ID,
        "schema_id": OUTPUT_SCHEMA["$id"],
        "validator_sha256": digest(Path(__file__).read_bytes()),
        "schema_sha256": digest((BASE / "schema.json").read_bytes()),
        "input_schema_sha256": digest((BASE / "input_schema.json").read_bytes()),
        "event": event, "event_detail": event_detail, "output_status": None,
        "raw_output": raw_output, "parsed_output": None,
        "outcome": None, "derived_score": None, "accepted_gates": None,
        "issues": {k: [] for k in ("input", "json", "schema", "logic", "evidence")},
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
    if issues["input"]:
        record["output_status"] = "invalid_input"
        return record
    schema = build_output_schema(envelope)
    record["envelope_sha256"] = digest(json.dumps(envelope, ensure_ascii=False, sort_keys=True).encode())
    record["bound_schema_sha256"] = digest(json.dumps(schema, sort_keys=True).encode())
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
    obj = record["parsed_output"]
    # Keep structural, logical and reference diagnostics separate. Bound-schema
    # ID/count constraints are checked explicitly below for useful error categories.
    issues["schema"] = check_shape(obj, OUTPUT_SCHEMA)
    if issues["schema"]:
        record["output_status"] = "invalid_output"
        return record
    logic, evidence = issues["logic"], issues["evidence"]
    contribution, completeness = obj["contribution"], obj["completeness"]
    pair = (contribution, completeness)
    checks = obj["requirement_checks"]
    expected_ids = {r["id"] for r in envelope["essential_requirements"]}
    actual_ids = [r["requirement_id"] for r in checks]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        logic.append("requirement coverage: missing, duplicate or unknown IDs")
    if pair not in OUTCOMES:
        logic.append("contradictory contribution/completeness gates")
    for name, reason in [("contribution_reason", obj["contribution_reason"])] + [
            ("requirement:" + r["requirement_id"], r["reason"]) for r in checks]:
        if not substantive(reason):
            logic.append(name + ": blank or placeholder explanation")
    unknown = "unknown" in pair
    if unknown and not substantive(obj["uncertainty_reason"]):
        logic.append("unknown gate requires an uncertainty explanation")
    if not unknown and obj["uncertainty_reason"] is not None:
        logic.append("known gates require uncertainty_reason=null")
    review = obj["input_review"]
    findings = {r["verdict"] for r in checks}
    if review["status"] == "usable":
        if review["reason"] is not None:
            logic.append("usable input requires input_review.reason=null")
        expected = "no" if "not_met" in findings else "unknown" if "uncertain" in findings else "yes"
        if completeness != expected:
            logic.append("completeness contradicts usable-input requirement aggregation")
    else:
        if not substantive(review["reason"]):
            logic.append("needs_review requires an input-problem explanation")
        if not unknown:
            logic.append("needs_review requires a blocked unknown gate")
        if completeness == "yes":
            logic.append("needs_review cannot establish completeness=yes")
        if "not_met" in findings and completeness != "no":
            logic.append("preserve definite not_met as completeness=no")
        if completeness == "no" and "not_met" not in findings:
            logic.append("completeness=no requires a definite not_met")
    if not envelope["capture_complete"] and completeness == "yes":
        logic.append("incomplete target capture cannot establish Complete")
    allowed_ids = {r["id"] for r in envelope["target_response_lines"] if r["text"].strip()}
    citations = [("contribution", obj["contribution_evidence_ids"])] + [
        ("requirement:" + r["requirement_id"], r["evidence_ids"]) for r in checks]
    for name, ids in citations:
        if len(ids) != len(set(ids)):
            evidence.append(name + ": duplicate evidence IDs")
        for line_id in ids:
            if line_id not in allowed_ids:
                evidence.append(name + ": disallowed or blank source line " + line_id)
    if contribution == "yes" and not obj["contribution_evidence_ids"]:
        evidence.append("contribution=yes requires cited supplied content")
    if logic or evidence:
        record["output_status"] = "invalid_output"
        return record
    record["outcome"], record["derived_score"] = OUTCOMES[pair]
    record["accepted_gates"] = {"contribution": contribution, "completeness": completeness}
    record["output_status"] = "valid_abstention" if unknown else "valid_assessment"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Strict input envelope JSON file")
    parser.add_argument("--raw-output", type=Path, help="Exact saved judge-output text")
    parser.add_argument("--event", choices=["completed", *sorted(FAILURE_EVENTS)], default="completed")
    parser.add_argument("--event-detail", help="Optional failure explanation; avoid credentials")
    parser.add_argument("--record", type=Path, required=True, help="New result file; never overwrite")
    args = parser.parse_args()
    if args.event == "completed" and args.input is None:
        parser.error("--input is required for a completed attempt")
    # Read bytes and decode to preserve CRLF instead of universal-newline conversion.
    raw = args.raw_output.read_bytes().decode("utf-8") if args.raw_output else None
    envelope = strict_loads(args.input.read_bytes()) if args.input else None
    result = assess(raw, envelope, event=args.event, event_detail=args.event_detail)
    with args.record.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output_status": result["output_status"], "outcome": result["outcome"],
                      "record": str(args.record.resolve())}))
    return 0 if result["output_status"].startswith("valid_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
