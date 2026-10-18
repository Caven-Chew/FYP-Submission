#!/usr/bin/env python3
"""Prepare and validate the frozen two-label contract offline; no model calls."""

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = Path("contracts/two_label_v1")
RELEASE = BASE
RELEASE_ID = "two-label-judge-contract-release-01"
RELEASE_STATUS = "frozen_offline_baseline"
SCOPE = "objective_only_and_static_response_fulfilment_offline_contracts"
COMPONENTS = {
    "objective_policy_id": "objective-harmfulness-policy-v1.1",
    "objective_contract_id": "objective-harmfulness-contract-draft-01",
    "fulfilment_instruction_id": "general-task-fulfilment-draft-01",
    "fulfilment_contract_id": "general-task-fulfilment-contract-draft-01",
}
OBJECTIVE = BASE / "objective"
FULFILMENT = BASE / "fulfilment"
ASSETS = [
    *(directory / name for directory in (OBJECTIVE, FULFILMENT)
      for name in ("validator.py", "input_schema.json", "schema.json")),
    OBJECTIVE / "policy_instruction.txt", OBJECTIVE / "format_instruction.txt",
    RELEASE / "fulfilment_instruction.txt", RELEASE / "fulfilment_format_instruction.txt",
    Path("tools/judge_contract_release.py"),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def json_hash(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8"))


def read_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON key: " + key)
            value[key] = item
        return value

    def reject(value):
        raise ValueError("non-JSON constant: " + value)

    return json.loads(path.read_bytes().decode("utf-8"), object_pairs_hook=pairs, parse_constant=reject)


def read_text(path):
    return path.read_bytes().decode("utf-8")


def verify_release(root=ROOT):
    path = root / RELEASE / "manifest.json"
    manifest = read_json(path)
    expected = {str(p) for p in ASSETS}
    if (manifest.get("release_id") != RELEASE_ID or manifest.get("status") != RELEASE_STATUS
            or manifest.get("scope") != SCOPE or manifest.get("components") != COMPONENTS
            or set(manifest.get("files", {})) != expected):
        raise ValueError("release identity, frozen status, scope, components or dependency list differs from this tool")
    for filename, expected_hash in manifest["files"].items():
        if sha((root / filename).read_bytes()) != expected_hash:
            raise ValueError("frozen release dependency changed: " + filename)
    return sha(path.read_bytes())


def load_component(directory, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / directory / "validator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_identity(objective, context):
    return {"objective_sha256": sha(objective.encode("utf-8")),
            "context_sha256": sha(context.encode("utf-8"))}


def require_canonical_lines(lines):
    texts = [line["text"] for line in lines]
    if texts != "".join(texts).splitlines(keepends=True):
        raise ValueError("source items must exactly match splitlines(keepends=True)")


def checked_schema(stage, envelope, components):
    schema = components[stage].build_output_schema(envelope)
    names = ("objective_lines", "context_lines") if stage == "objective" else ("target_response_lines",)
    for name in names:
        require_canonical_lines(envelope[name])
    return schema


def payload(stage, envelope, components):
    if stage == "objective":
        instruction, formatting = OBJECTIVE / "policy_instruction.txt", OBJECTIVE / "format_instruction.txt"
    else:
        instruction, formatting = RELEASE / "fulfilment_instruction.txt", RELEASE / "fulfilment_format_instruction.txt"
    return {"instruction": read_text(ROOT / instruction), "format_instruction": read_text(ROOT / formatting),
            "input": envelope, "output_schema": checked_schema(stage, envelope, components)}


def check_basis(basis, components):
    if not isinstance(basis, dict) or set(basis) != {"objective", "objective_context", "essential_requirements"}:
        raise ValueError("fulfilment basis must contain exactly objective, objective_context and essential_requirements")
    envelope = {"unit": "static_response", **basis, "capture_complete": True, "target_response_lines": []}
    checked_schema("fulfilment", envelope, components)


def write_new(path, content):
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(content)


def save_packet(directory, stage, values, identity, requirements, release_hash, **metadata):
    contents = {name: json_text(value) for name, value in values.items()}
    manifest = {"release_id": RELEASE_ID, "release_manifest_sha256": release_hash, "stage": stage,
                "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "source_identity": identity,
                "requirements_sha256": json_hash(requirements),
                "files": {name: sha(content.encode("utf-8")) for name, content in contents.items()},
                "model_calls": 0, **metadata}
    contents["manifest.json"] = json_text(manifest)
    directory.mkdir()
    for name, content in contents.items():
        write_new(directory / name, content)


def verify_packet(directory, stage, release_hash, components):
    manifest = read_json(directory / "manifest.json")
    expected = {"input.json", "payload.json"}
    if stage == "objective":
        expected.add("fulfilment_basis.json")
    if (manifest.get("stage") != stage or manifest.get("release_id") != RELEASE_ID
            or manifest.get("release_manifest_sha256") != release_hash
            or set(manifest.get("files", {})) != expected):
        raise ValueError("packet stage, release or file list does not match")
    for filename, expected_hash in manifest["files"].items():
        if sha((directory / filename).read_bytes()) != expected_hash:
            raise ValueError("packet file changed: " + filename)
    envelope = read_json(directory / "input.json")
    if read_json(directory / "payload.json") != payload(stage, envelope, components):
        raise ValueError("payload does not match the frozen instructions, input and bound schema")
    if stage == "objective":
        basis = read_json(directory / "fulfilment_basis.json")
        check_basis(basis, components)
        objective = "".join(line["text"] for line in envelope["objective_lines"])
        context = "".join(line["text"] for line in envelope["context_lines"])
        if objective != basis["objective"] or context != basis["objective_context"]:
            raise ValueError("objective and fulfilment basis do not contain identical task text")
    else:
        basis = {key: envelope[key] for key in ("objective", "objective_context", "essential_requirements")}
        objective, context = basis["objective"], basis["objective_context"]
        response = "".join(line["text"] for line in envelope["target_response_lines"])
        if manifest.get("response_sha256") != sha(response.encode("utf-8")):
            raise ValueError("response identity does not match packet text")
    if manifest.get("source_identity") != source_identity(objective, context):
        raise ValueError("source identity does not match packet text")
    if manifest.get("requirements_sha256") != json_hash(basis["essential_requirements"]):
        raise ValueError("requirements identity does not match packet")
    return envelope, basis, manifest


def prepare_case(args, release_hash, components):
    objective = read_text(args.objective_file)
    context = read_text(args.context_file) if args.context_file else ""
    requirements = read_json(args.requirements_file)
    basis = {"objective": objective, "objective_context": context, "essential_requirements": requirements}
    check_basis(basis, components)
    envelope = components["objective"].prepare_input(objective, context)
    values = {"input.json": envelope, "payload.json": payload("objective", envelope, components),
              "fulfilment_basis.json": basis}
    save_packet(args.output_dir, "objective", values, source_identity(objective, context), requirements, release_hash)


def prepare_response(args, release_hash, components):
    _, basis, manifest = verify_packet(args.case_dir, "objective", release_hash, components)
    response_bytes = args.response_file.read_bytes()
    response = response_bytes.decode("utf-8")
    envelope = {"unit": "static_response", **basis, "capture_complete": args.capture_complete == "true",
                "target_response_lines": [{"id": f"L{i:03d}", "text": line}
                                          for i, line in enumerate(response.splitlines(keepends=True), 1)]}
    values = {"input.json": envelope, "payload.json": payload("fulfilment", envelope, components)}
    save_packet(args.output_dir, "fulfilment", values, manifest["source_identity"], basis["essential_requirements"],
                release_hash, case_manifest_sha256=sha((args.case_dir / "manifest.json").read_bytes()),
                response_sha256=sha(response_bytes))


def validate(args, release_hash, components):
    component = components[args.stage]
    if args.event != "completed" and args.event not in component.FAILURE_EVENTS:
        raise ValueError("event is not supported by the selected stage")
    envelope, manifest = None, None
    if args.event == "completed":
        envelope, _, manifest = verify_packet(args.packet_dir, args.stage, release_hash, components)
    raw = read_text(args.raw_output) if args.raw_output else None
    record = component.assess(raw, envelope, event=args.event, event_detail=args.event_detail)
    record.update({"release_id": RELEASE_ID, "release_manifest_sha256": release_hash, "stage": args.stage,
                   "packet_verification": "passed" if manifest else "skipped_operational_event",
                   "packet_manifest_sha256": sha((args.packet_dir / "manifest.json").read_bytes()) if manifest else None,
                   "source_identity": manifest["source_identity"] if manifest else None,
                   "requirements_sha256": manifest["requirements_sha256"] if manifest else None})
    write_new(args.record, json_text(record))
    print(json.dumps({"output_status": record["output_status"], "record": str(args.record)}))
    return 0 if record["output_status"].startswith("valid_") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Verify every frozen executable/instruction/schema dependency")
    case = commands.add_parser("prepare-case", help="Freeze task/requirements before adding any response")
    case.add_argument("--objective-file", type=Path, required=True)
    case.add_argument("--context-file", type=Path)
    case.add_argument("--requirements-file", type=Path, required=True)
    case.add_argument("--output-dir", type=Path, required=True)
    response = commands.add_parser("prepare-response", help="Add exact saved response to verified case basis")
    response.add_argument("--case-dir", type=Path, required=True)
    response.add_argument("--response-file", type=Path, required=True)
    response.add_argument("--capture-complete", choices=["true", "false"], required=True)
    response.add_argument("--output-dir", type=Path, required=True)
    validation = commands.add_parser("validate", help="Check saved output or record a known operational failure")
    validation.add_argument("--stage", choices=["objective", "fulfilment"], required=True)
    validation.add_argument("--packet-dir", type=Path)
    validation.add_argument("--raw-output", type=Path)
    validation.add_argument("--record", type=Path, required=True)
    validation.add_argument("--event", choices=["completed", "provider_refusal", "transport_error", "incomplete_output", "missing_input", "missing_capture"], default="completed")
    validation.add_argument("--event-detail")
    args = parser.parse_args(argv)
    if args.command == "validate" and args.event == "completed" and (not args.packet_dir or not args.raw_output):
        parser.error("completed validation requires --packet-dir and --raw-output")
    try:
        release_hash = verify_release()
        if args.command == "check":
            print(f"PASS: {RELEASE_ID}; {len(ASSETS)} dependencies; manifest SHA-256 {release_hash}")
            return 0
        components = {"objective": load_component(OBJECTIVE, "release_objective"),
                      "fulfilment": load_component(FULFILMENT, "release_fulfilment")}
        if args.command == "validate":
            return validate(args, release_hash, components)
        if args.command == "prepare-case":
            prepare_case(args, release_hash, components)
        else:
            prepare_response(args, release_hash, components)
        print(f"Prepared {args.command}: {args.output_dir}")
        return 0
    except (OSError, UnicodeError, ValueError, TypeError, AttributeError, RecursionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
