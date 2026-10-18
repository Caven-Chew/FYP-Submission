"""Azure AI Evaluation RedTeam attacks."""
from __future__ import annotations

import os
import json
import time
from typing import Any
from types import SimpleNamespace

from .retry import BenchmarkAborted, ConsecutiveFailureGuard, send_prompt
from .scoring import judge_response, severity_provenance, assessment_settings, DUAL_METHOD, judge_version_results
from .llms import llm_overrides
from ..models import PromptStatus


# ---------- Azure RedTeam -------------------------------------------------

async def run_azure_redteam_attack(
    *, attack: dict, target: Any, judge: Any, guard: "ConsecutiveFailureGuard",
) -> list[dict]:
    """Run Azure AI Evaluation's RedTeam.

    Requires `azure-ai-evaluation` + `azure-identity` installed, plus
    AZURE_AI_PROJECT_ENDPOINT in .env. Raises a clear error if any of those
    are missing so the benchmark records a single failure row instead of
    crashing the whole run.

    Returns a list of {sequence, prompt_text, response, succeeded, severity,
    verdict, latency_ms, taxonomy_category, objective_id, error} per attack
    conversation Azure produced. `guard` tracks target failures during the
    scan (see `_target_cb`); Azure's `RedTeam.scan()` owns the call loop and
    may not propagate an exception raised from the callback cleanly, so this
    can't reliably abort mid-scan — the caller checks `guard.tripped` after
    this returns and stops the *next* attack instead (see `_run_attacks`).
    """
    try:
        from azure.ai.evaluation.red_team import RedTeam, RiskCategory, AttackStrategy  # type: ignore
        from azure.identity import DefaultAzureCredential  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Azure RedTeam requires 'azure-ai-evaluation' and 'azure-identity'. "
            f"Install with: pip install azure-ai-evaluation azure-identity  ({exc})"
        ) from exc

    project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    if not project_endpoint:
        raise RuntimeError(
            "AZURE_AI_PROJECT_ENDPOINT is not set. Add it to .env (see README.md) — "
            "Azure RedTeam needs an Azure AI Project to log adversarial conversations into."
        )

    # Map attack-config strings to Azure enums; skip unknown names rather than
    # crash, so newer Azure SDK releases don't break the run.
    risks = []
    for r in attack.get("risk_categories", []):
        val = getattr(RiskCategory, r, None)
        if val is not None:
            risks.append(val)
    strategies = []
    for s in attack.get("attack_strategies", ["Baseline"]):
        val = getattr(AttackStrategy, s, None)
        if val is not None:
            strategies.append(val)
    if not risks:
        raise RuntimeError(
            f"No valid risk_categories on attack {attack['slug']!r} — "
            "check Azure RiskCategory enum names."
        )

    # Target callback — Azure calls this for each adversarial prompt.
    async def _target_cb(query: str) -> str:
        text, err = await send_prompt(target, query)
        try:
            guard.record(err)
        except BenchmarkAborted:
            # Swallow here — see the docstring above for why we can't safely
            # raise out of Azure's own scan loop from this callback.
            # guard.tripped stays true; the caller checks it once we return.
            pass
        if err:
            # Azure drives its own scan loop and hands back only a string per
            # prompt, so there is no per-row column to put a capture ref in.
            # Name it inline instead, so a failure in the results table can
            # still be traced to its screenshot on disk.
            ref = (getattr(target, "last_debug", None) or {}).get("capture_ref")
            return f"[target-error] {err}" + (f" (capture: {ref})" if ref else "")
        return text or ""

    credential = DefaultAzureCredential()
    red_team = RedTeam(
        azure_ai_project=project_endpoint,
        credential=credential,
        risk_categories=risks,
        num_objectives=int(attack.get("num_objectives", 5)),
    )

    t0 = time.perf_counter()
    try:
        scan_result = await red_team.scan(
            target=_target_cb,
            attack_strategies=strategies,
            scan_name=f"{attack['slug']}-{int(t0)}",
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Azure RedTeam scan failed: {type(exc).__name__}: {exc}") from exc
    total_latency = int((time.perf_counter() - t0) * 1000)

    # Azure result shape varies by SDK version. Normalise to a list of
    # {prompt, response, risk_category, attack_strategy, success}.
    attack_details = []
    if isinstance(scan_result, dict):
        attack_details = (
            scan_result.get("attack_details")
            or scan_result.get("conversations")
            or scan_result.get("results")
            or []
        )
    elif hasattr(scan_result, "attack_details"):
        attack_details = getattr(scan_result, "attack_details") or []

    tax = attack.get("taxonomy_category")
    results: list[dict] = []

    for i, entry in enumerate(attack_details):
        # Pull whatever the SDK gave us in a tolerant way.
        if isinstance(entry, dict):
            convo = entry.get("conversation") or entry.get("messages") or []
            user_msgs = [m for m in convo if (m.get("role") if isinstance(m, dict) else "") == "user"]
            asst_msgs = [m for m in convo if (m.get("role") if isinstance(m, dict) else "") == "assistant"]
            prompt_text = (user_msgs[-1].get("content") if user_msgs else entry.get("query") or entry.get("prompt", "")) or ""
            response = (asst_msgs[-1].get("content") if asst_msgs else entry.get("response", "")) or ""
            risk_cat = entry.get("risk_category") or entry.get("risk", "")
            strat = entry.get("attack_strategy") or entry.get("strategy", "")
            success_field = entry.get("attack_success", entry.get("success"))
        else:
            prompt_text = str(getattr(entry, "prompt", "") or getattr(entry, "query", ""))
            response = str(getattr(entry, "response", ""))
            risk_cat = str(getattr(entry, "risk_category", ""))
            strat = str(getattr(entry, "attack_strategy", ""))
            success_field = getattr(entry, "attack_success", None)

        # Trust Azure's evaluator if present; otherwise re-judge with our LLM.
        # Severity has to come from ONE place or the axis is not comparable
        # across attacks — the whole point of running Azure's strategies
        # alongside the static ones is to rank them against each other. Azure
        # only reports a binary pass/fail, and the old code turned that into a
        # 4 or a 1, which then pooled with real judge scores in the histogram
        # and the mean. So always ask our judge, and keep Azure's own call as
        # evidence in the verdict rather than as a number.
        cfg = llm_overrides.get()
        assessment = (assessment_settings(SimpleNamespace(llm_config=cfg), attack)
                      if cfg.get('JUDGE_METHOD') == DUAL_METHOD else {})
        verdict, severity, succeeded = await judge_response(judge, prompt_text, response, **assessment)
        # Computed from the ORIGINAL judge verdict, before the azure: prefix is
        # appended below — so 'fallback' still reflects how the judge's own reply
        # was parsed.
        severity_source = severity_provenance(verdict, severity)
        if success_field is not None:
            azure_succeeded = (bool(success_field)
                               and not str(success_field).lower().startswith("unsuccess"))
            if cfg.get('JUDGE_METHOD') == DUAL_METHOD and judge_version_results(verdict):
                payload = json.loads(verdict)
                payload['azure_result'] = str(success_field)
                verdict = json.dumps(payload, ensure_ascii=False)
            else:
                verdict = f"azure:{success_field} | {verdict}"
            if severity is None and cfg.get('JUDGE_METHOD') != DUAL_METHOD:
                # The judge was unavailable or its reply was unparseable. Fall
                # back to Azure's verdict so a real finding is not silently
                # recorded as a failure — but still do not invent a severity.
                succeeded = azure_succeeded

        if cfg.get('JUDGE_METHOD') == DUAL_METHOD and judge_version_results(verdict):
            payload = json.loads(verdict)
            payload.update(azure_risk=risk_cat, azure_strategy=strat)
            saved_verdict = json.dumps(payload, ensure_ascii=False)
        else:
            saved_verdict = f"{verdict} | risk={risk_cat} | strat={strat}"
        results.append({
            "sequence": i,
            "prompt_text": prompt_text,
            "response": response,
            "error": None,
            "succeeded": succeeded,
            "severity": severity,
            "severity_source": severity_source,
            "verdict": saved_verdict,
            "status": PromptStatus.JUDGE_ERROR if cfg.get("JUDGE_METHOD") == DUAL_METHOD and severity is None else PromptStatus.OK,
            # Azure reports one duration for the whole scan and never per
            # prompt. Attribute that single real measurement to the first row
            # and leave the rest null rather than dividing it up and presenting
            # every slice as if it had been timed.
            "latency_ms": total_latency if not results else None,
            "latency_measured": not results,
            "taxonomy_category": tax,
            "objective_id": f"az-{risk_cat}-{strat}-{i}",
        })

    if not results:
        # Azure scan returned nothing — surface as one failure row.
        results.append({
            "sequence": 0, "prompt_text": "(no Azure attack_details returned)",
            "response": "", "error": "Azure RedTeam returned no attack_details",
            "succeeded": False, "severity": None, "verdict": "empty",
            "status": PromptStatus.SETUP_ERROR,
            "severity_source": "none", "latency_measured": False,
            "latency_ms": total_latency, "taxonomy_category": tax,
            "objective_id": "az-empty",
        })
    return results  # caller persists these, then checks guard.tripped
