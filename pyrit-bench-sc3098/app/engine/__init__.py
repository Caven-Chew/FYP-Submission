"""The benchmark engine — everything that runs an attack and scores a reply.

Read it in this order:

  runner.py          the entry point: start_benchmark() spawns the thread that
                     walks the selected attacks and writes Prompt rows
  static_attack.py   one-shot dataset prompts (optionally converted)
  dynamic_attack.py  multi-turn PyRIT orchestrators driven by an attacker LLM
  azure_attack.py    Azure AI Evaluation's RedTeam sweep

Supporting modules:

  llms.py            builds the judge / attacker / converter models
  datasets.py        where prompts and objectives come from
  converters.py      the prompt transforms (poetry, many-shot, translate, ...)
  scoring.py         the LLM judge, severity parsing, refusal heuristics
  conversations.py   reading conversations back out of PyRIT memory
  retry.py           send-with-backoff and the consecutive-failure abort guard
  connectivity.py    the "test this target" button

Prompt text lives in app/prompts.py; tunable numbers live in app/settings.py.
"""
from __future__ import annotations

from ..settings import ATTEMPTS_PER_PROMPT, SUCCESS_SEVERITY
from .connectivity import test_target
from .control import (
    BenchmarkCancelled, is_cancelled, is_paused, request_cancel,
    request_pause, request_resume,
)
from .converters import build_converter, build_converters
from .datasets import (
    load_dataset_prompts, load_structured_objectives, load_taxonomy_slugs
)
from .llms import build_attacker_llm, build_judge_llm
from .retry import BenchmarkAborted
from .runner import cancel_and_join, start_benchmark
from .scoring import judge_response

__all__ = [
    "ATTEMPTS_PER_PROMPT", "SUCCESS_SEVERITY", "BenchmarkAborted", "BenchmarkCancelled",
    "build_attacker_llm", "build_converter", "build_converters", "build_judge_llm",
    "cancel_and_join", "is_cancelled", "is_paused", "judge_response",
    "load_dataset_prompts", "load_structured_objectives",
    "load_taxonomy_slugs", "request_cancel", "request_pause", "request_resume",
    "start_benchmark", "test_target",
]
