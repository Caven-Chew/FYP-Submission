"""Scripted conversations: per-reply scoring for new runs, whole-script scoring for saved legacy runs.

Each script runs in a fresh conversation. Turns are sent through the same
request the target's connection test used: API targets carry the earlier turns
in their saved body (``HttpTarget.send_turn_async``), and browser targets keep
one chat page per conversation. Every exchange is committed as it happens, so
an interruption never turns an unfinished conversation into a defensive success.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid

from ..activity import set_activity
from ..extensions import db
from ..models import ConversationRetry, Prompt, PromptStatus
from ..settings import REQUEST_DELAY
from ..targets.throttle import throttle_slept_ms
from .control import wait_while_paused
from .pending_scores import PendingScores, close_after_reply, prepare_reply_for_close
from .scoring import (assessment_settings, detect_canary, judge_response, looks_like_refusal,
                      severity_provenance, judge_interactive_response, interactive_judge_input,
                      SC3098_METHOD)


class _Turn:
    """Minimal message for targets that key their chat session on conversation_id."""

    def __init__(self, text: str, conversation_id: str):
        self._text = text
        self.conversation_id = conversation_id

    def get_value(self) -> str:
        return self._text


async def _send_turn(target, prompt: str, history: list[dict], conversation_id: str) -> str:
    if hasattr(target, "send_turn_async"):
        return await target.send_turn_async(prompt, history)
    from pyrit.prompt_target import PromptTarget
    if isinstance(target, PromptTarget):
        from pyrit.models import Message
        message = Message.from_prompt(prompt=prompt, role="user")
        message.get_piece().conversation_id = conversation_id
    else:
        message = _Turn(prompt, conversation_id)
    pieces = await target.send_prompt_async(message=message)
    return pieces[0].get_value() if pieces else ""


def judge_views(transcript: list[dict]) -> tuple[str, str]:
    """User turns and assistant replies in their own judge slots, paired by turn number."""
    users = "\n\n".join(f"Turn {e['turn']} (user): {e['prompt']}" for e in transcript)
    replies = "\n\n".join(f"Turn {e['turn']} (assistant reply): {e['response']}" for e in transcript)
    return users, replies


async def run_scripted_attack(bench, attack, target, judge, limit, guard, *,
                              resume=False, retry_failed=True, retry_prompt=None):
    if bench.llm_config.get("SCRIPTED_SCORE_UNIT") == "turn":
        return await _run_scored_turns(bench, attack, target, judge, guard,
                                      resume=resume, retry_failed=retry_failed, retry_prompt=retry_prompt)
    # Historical runs retain their saved whole-conversation scoring unit.
    # Students' own cases always run in full; the per-attack cap is for datasets.
    scripts = bench.custom_conversations
    if not scripts:
        raise ValueError("This benchmark has no saved multi-turn conversations.")
    assessment = assessment_settings(bench, attack)
    existing = {p.sequence: p for p in Prompt.query.filter_by(
        benchmark_id=bench.id, attack_slug=attack['slug']).all()
        if p.status != PromptStatus.SETUP_ERROR}

    for index, turns in enumerate(scripts):
        if retry_prompt is not None and index != retry_prompt.sequence:
            continue
        old = existing.get(index) if resume else retry_prompt
        # A conversation left INCOMPLETE was cut off by a restart: it is
        # unfinished work, so Resume (retry_failed=False) finishes it too.
        if (old is not None and resume and old.status != PromptStatus.INCOMPLETE
                and (not retry_failed or not old.can_rerun)):
            continue
        await wait_while_paused(bench.id)
        if old is not None and old.can_rescore:
            # Every turn was captured; only the judge failed. Judge the stored
            # transcript again instead of replaying the conversation, which
            # would replace the captured replies (review finding #3).
            set_activity(bench.id, 'judging', f'Re-scoring stored conversation {index + 1}/{len(scripts)}')
            stored = old.transcript
            users, replies = judge_views(stored)
            old.verdict, old.severity, old.succeeded = await judge_response(judge, users, replies, **assessment)
            old.severity_source = severity_provenance(old.verdict, old.severity)
            old.status = (PromptStatus.JUDGE_ERROR if old.severity is None else
                          PromptStatus.RECOVERED if any(e.get('status') == PromptStatus.RECOVERED for e in stored)
                          else PromptStatus.OK)
            db.session.commit()
            continue
        if old is not None:
            db.session.add(ConversationRetry(
                benchmark_id=bench.id, attack_slug=attack['slug'],
                objective_id=old.objective_id or f"script-{index}",
                original_rows_json=json.dumps([
                    {column.name: getattr(old, column.name) for column in Prompt.__table__.columns}
                ], default=str)))
            row = old
        else:
            row = Prompt(benchmark_id=bench.id, attack_slug=attack['slug'], sequence=index,
                         attempt=0, attempt_group=f"{attack['slug']}#{index}",
                         objective_id=f"script-{index}", taxonomy_category=attack.get('taxonomy_category'))
            db.session.add(row)
        row.prompt_text = '\n---\n'.join(turns)
        row.response = ''
        row.transcript_json = '[]'
        row.status = PromptStatus.INCOMPLETE
        row.rubric = 'harm' if assessment.get('method') == 'dual_v1_v2' else assessment['rubric']
        row.severity = None
        row.severity_source = 'none'
        row.succeeded = False
        row.refused = False
        row.error = None
        row.verdict = None
        row.capture_ref = None
        row.latency_ms = 0
        row.latency_measured = True
        row.canary_detected = None
        row.note = ('Conversation rerun from the beginning in a fresh chat.' if old else None)
        db.session.commit()
        transcript = []
        history: list[dict] = []
        conversation_id = str(uuid.uuid4())
        try:
            for turn_no, turn in enumerate(turns, 1):
                await wait_while_paused(bench.id)
                if transcript and REQUEST_DELAY:
                    await asyncio.sleep(REQUEST_DELAY)
                    await wait_while_paused(bench.id)
                set_activity(bench.id, 'sending',
                             f"Conversation {index + 1}/{len(scripts)}: turn {turn_no}/{len(turns)}")
                entry = {'turn': turn_no, 'prompt': turn, 'response': '',
                         'conversation_id': conversation_id, 'status': PromptStatus.SEND_ERROR}
                started = time.perf_counter()
                slept = throttle_slept_ms(target)
                try:
                    reply = (await _send_turn(target, turn, history, conversation_id) or '').strip()
                    if not reply:
                        raise ValueError('No complete assistant response captured for this turn.')
                    entry['response'] = reply
                    debug = getattr(target, 'last_debug', None) or {}
                    entry['status'] = PromptStatus.RECOVERED if debug.get('recovery_note') else PromptStatus.OK
                    if debug.get('recovery_note'):
                        entry['note'] = debug['recovery_note']
                except Exception as exc:
                    # Keep the root cause short: the first line, not a wrapped HTML body.
                    entry['error'] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"
                    raise
                finally:
                    entry['latency_ms'] = max(0, int((time.perf_counter() - started) * 1000
                                                    - (throttle_slept_ms(target) - slept)))
                    ui_ms = (getattr(target, 'last_debug', None) or {}).get('latency_ms')
                    if entry['response'] and ui_ms is not None:
                        entry['latency_ms'] = ui_ms  # UI target: send action to final reply text (4 Oct 2026)
                    transcript.append(entry)
                    row.transcript_json = json.dumps(transcript, ensure_ascii=False)
                    row.response = '\n\n'.join(e['response'] for e in transcript if e['response'])
                    row.latency_ms = sum(e['latency_ms'] for e in transcript)
                    row.canary_detected = detect_canary(bench, [e['response'] for e in transcript])
                    row.capture_ref = (getattr(target, 'last_debug', None) or {}).get('capture_ref')
                    db.session.commit()
                history += [{'role': 'user', 'content': turn}, {'role': 'assistant', 'content': entry['response']}]
        except Exception as exc:
            row.status = PromptStatus.SEND_ERROR
            failed = transcript[-1] if transcript else {}
            row.error = (f"Turn {failed.get('turn', '?')} of {len(turns)}: "
                         + (failed.get('error') or f'{type(exc).__name__}: {exc}'))
            db.session.commit()
            guard.record(row.error)
            continue

        set_activity(bench.id, 'judging', f'Scoring full conversation {index + 1}/{len(scripts)}')
        users, replies = judge_views(transcript)
        row.verdict, row.severity, row.succeeded = await judge_response(judge, users, replies, **assessment)
        row.severity_source = severity_provenance(row.verdict, row.severity)
        row.status = (PromptStatus.JUDGE_ERROR if row.severity is None else
                      PromptStatus.RECOVERED if any(e['status'] == PromptStatus.RECOVERED for e in transcript)
                      else PromptStatus.OK)
        row.refused = any(looks_like_refusal(e['response']) for e in transcript)
        db.session.commit()
        guard.record(None)
    return True


async def _score_turn(bench, row, judge, assessment, history):
    """Score only this saved reply; prior exchanges provide context, not evidence of new success."""
    from flask import current_app
    options = {key: assessment[key] for key in ("method", "rubric", "assessment_rules") if key in assessment}
    options.setdefault("method", SC3098_METHOD)
    exact = interactive_judge_input(row.prompt_text, row.response, history, **options)
    row.judge_input_json = json.dumps({"version": "scripted-turn-1", "method": options["method"],
                                      "assessment": options, "history": history,
                                      "prompt": row.prompt_text, "response": row.response,
                                      "input": exact}, ensure_ascii=False)
    row.status = PromptStatus.SCORING
    db.session.commit()
    set_activity(bench.id, "judging", f"Scoring {row.objective_id}, turn {row.transcript[0]['turn']}")
    if len(exact) > current_app.config.get("INTERACTIVE_JUDGE_MAX_CHARS", 120000):
        row.verdict, row.severity, row.succeeded = "Judge context limit exceeded; reply remains unscored.", None, False
    else:
        timeout = float(current_app.config.get("INTERACTIVE_JUDGE_TIMEOUT_SECONDS", 120))
        try:
            row.verdict, row.severity, row.succeeded = await asyncio.wait_for(
                judge_interactive_response(judge, row.prompt_text, row.response, history,
                                           judge_timeout_seconds=timeout, **options),
                timeout * (2 if options['method'] == 'dual_v1_v2' else 1) + 5)
        except Exception as exc:
            row.verdict, row.severity, row.succeeded = f"judge-error: {type(exc).__name__}", None, False
    row.severity_source = severity_provenance(row.verdict, row.severity)
    row.status = (PromptStatus.JUDGE_ERROR if row.severity is None else
                  PromptStatus.RECOVERED if row.transcript[0].get("status") == PromptStatus.RECOVERED
                  else PromptStatus.OK)
    row.refused = looks_like_refusal(row.response)
    db.session.commit()


async def _run_scored_turns(bench, attack, target, judge, guard, *, resume, retry_failed, retry_prompt, pending_scores=None):
    if pending_scores is None:
        async with PendingScores() as scores:
            return await _run_scored_turns(bench, attack, target, judge, guard,
                resume=resume, retry_failed=retry_failed, retry_prompt=retry_prompt, pending_scores=scores)
    scripts = bench.custom_conversations
    if not scripts:
        raise ValueError("This benchmark has no saved scripted conversations.")
    assessment = assessment_settings(bench, attack)
    offset = 0
    for index, turns in enumerate(scripts):
        first_sequence = offset
        offset += len(turns)
        objective_id = f"script-{index}"
        if retry_prompt is not None and retry_prompt.objective_id != objective_id:
            continue
        rows = Prompt.query.filter_by(benchmark_id=bench.id, attack_slug=attack['slug'],
                                      objective_id=objective_id).order_by(Prompt.sequence).all()
        if retry_prompt is not None and retry_prompt.can_rescore:
            snapshot = json.loads(retry_prompt.judge_input_json)
            await _score_turn(bench, retry_prompt, judge, snapshot['assessment'], snapshot['history'])
            continue
        if resume and rows:
            failed_send = any(row.status == PromptStatus.SEND_ERROR for row in rows)
            interrupted = (any(row.status == PromptStatus.INCOMPLETE for row in rows)
                           or (len(rows) < len(turns) and not failed_send))
            if not retry_failed and failed_send and not interrupted:
                continue
            if len(rows) == len(turns) and all((row.response or '').strip() for row in rows):
                # Captured responses are immutable on a judge retry. Also recover
                # a reply committed immediately before the process stopped scoring.
                for row in rows:
                    if row.status == PromptStatus.INCOMPLETE or (retry_failed and row.can_rescore):
                        snapshot = json.loads(row.judge_input_json)
                        await _score_turn(bench, row, judge, snapshot['assessment'], snapshot['history'])
                continue
            if not interrupted and not retry_failed:
                continue
        if rows:
            # A lost/failed target chat cannot be resumed reliably. Preserve the
            # old group, replay this script in a fresh chat and keep other scripts.
            db.session.add(ConversationRetry(benchmark_id=bench.id, attack_slug=attack['slug'],
                objective_id=objective_id, original_rows_json=json.dumps([
                    {column.name: getattr(row, column.name) for column in Prompt.__table__.columns}
                    for row in rows], default=str)))
            for row in rows:
                db.session.delete(row)
            db.session.commit()
        history = []
        conversation_id = str(uuid.uuid4())
        for turn_no, turn in enumerate(turns, 1):
            await wait_while_paused(bench.id)
            if turn_no > 1 and REQUEST_DELAY:
                await asyncio.sleep(REQUEST_DELAY)
                await wait_while_paused(bench.id)
            row = Prompt(benchmark_id=bench.id, attack_slug=attack['slug'],
                         sequence=first_sequence + turn_no - 1, attempt=0,
                         attempt_group=f"{attack['slug']}#{index}", objective_id=objective_id,
                         taxonomy_category=attack.get('taxonomy_category'), prompt_text=turn,
                         response='', status=PromptStatus.INCOMPLETE, severity_source='none',
                         rubric='harm' if assessment.get('method') == 'dual_v1_v2' else assessment['rubric'],
                         latency_measured=True, note='Conversation replayed in a fresh chat.' if rows else None)
            entry = {'turn': turn_no, 'prompt': turn, 'response': '',
                     'conversation_id': conversation_id, 'status': PromptStatus.INCOMPLETE}
            row.transcript_json = json.dumps([entry], ensure_ascii=False)
            # Persist context before sending, so a captured reply can be scored
            # after an interruption without reconstructing or replaying history.
            options = {key: assessment[key] for key in ('method', 'rubric', 'assessment_rules') if key in assessment}
            row.judge_input_json = json.dumps({'history': history, 'assessment': options}, ensure_ascii=False)
            db.session.add(row)
            db.session.commit()
            set_activity(bench.id, 'sending', f"Conversation {index + 1}/{len(scripts)}: turn {turn_no}/{len(turns)}")
            started, slept = time.perf_counter(), throttle_slept_ms(target)
            failed = False
            try:
                prepare_reply_for_close(target, closing=turn_no == len(turns))
                row.response = (await _send_turn(target, turn, history, conversation_id) or '').strip()
                if not row.response:
                    raise ValueError('No complete assistant response captured for this turn.')
                debug = getattr(target, 'last_debug', None) or {}
                entry['status'] = PromptStatus.RECOVERED if debug.get('recovery_note') else PromptStatus.OK
                row.capture_ref = debug.get('capture_ref')
                if debug.get('recovery_note'):
                    row.note = debug['recovery_note']
            except Exception as exc:
                failed = True
                entry['status'] = row.status = PromptStatus.SEND_ERROR
                row.error = f"Turn {turn_no} of {len(turns)}: {type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"
                entry['error'] = row.error
            finally:
                row.latency_ms = max(0, int((time.perf_counter() - started) * 1000 - (throttle_slept_ms(target) - slept)))
                ui_ms = (getattr(target, 'last_debug', None) or {}).get('latency_ms')
                if row.response and ui_ms is not None:
                    row.latency_ms = ui_ms
                entry.update(response=row.response, latency_ms=row.latency_ms)
                row.transcript_json = json.dumps([entry], ensure_ascii=False)
                row.canary_detected = detect_canary(bench, [row.response])
                db.session.commit()
            if failed:
                await close_after_reply(target)
                guard.record(row.error)
                break
            row.status = PromptStatus.SCORING
            db.session.commit()
            # Close only the final turn's page; earlier turns share this chat.
            closing = asyncio.create_task(close_after_reply(target)) if turn_no == len(turns) else None
            try:
                await pending_scores.submit(lambda row=row, history=history:
                    _score_turn(bench, row, judge, assessment, history))
            finally:
                if closing is not None:
                    await closing
            history = history + [{'role': 'user', 'content': turn}, {'role': 'assistant', 'content': row.response}]
            guard.record(None)
    return True
