"""Where prompts and objectives come from: PyRIT datasets and the local JSON files."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..prompts import DEFAULT_TRUE_DESC


# app/data/, not app/engine/data/. This was `.parent` until 2026-08-30, which
# pointed at a directory that has never existed: every loader below silently
# returned empty, so dynamic attacks fell back to a PyRIT dataset with the
# generic true_description instead of the 36 structured objectives, and the
# many-shot converter lost its local examples. Nothing errored — it just
# measured something other than what the catalog said it did.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------- PyRIT datasets ------------------------------------------------

async def load_dataset_prompts(dataset_slug: str, limit: int) -> list[str]:
    """Return up to `limit` prompt strings from a PyRIT dataset.

    Resilient across PyRIT versions: tries the stable public
    `pyrit.datasets.fetch_*_dataset()` function first, then falls back to the
    version-specific private provider class. The returned dataset is normalised
    via its .seeds / .prompts list of objects carrying a .value / .prompt.
    """
    # A "local:<name>" slug reads app/data/local_prompts/<name>.json instead of
    # calling PyRIT. Used for attack families no public dataset covers — the
    # system-prompt extraction set is the first — so the catalog can carry them
    # without a second code path through the runner.
    if dataset_slug.startswith("local:"):
        return load_local_prompts(dataset_slug.split(":", 1)[1], limit)

    ds = await _fetch_dataset(dataset_slug)
    seeds = getattr(ds, "seeds", None) or getattr(ds, "prompts", None) or list(ds)
    out: list[str] = []
    for seed in seeds:
        text = getattr(seed, "value", None) or getattr(seed, "prompt", None) or str(seed)
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


# slug -> public pyrit.datasets fetch function name (stable API, tried first).
# Add a new static dataset by adding its slug here AND in _fetch_dataset_private.
_PUBLIC_DATASET_FN = {
    "harmbench": "fetch_harmbench_dataset",
    "advbench": "fetch_adv_bench_dataset",
    "xstest": "fetch_xstest_dataset",
    "forbidden_questions": "fetch_forbidden_questions_dataset",
}


async def _fetch_dataset(slug: str) -> Any:
    """Resolve a PyRIT dataset, tolerant of API changes between releases.

    Order: public ``fetch_*_dataset()`` (sync or async) → private provider class
    (current 0.9 layout). Both are guarded; if neither yields a dataset we raise
    an ImportError naming what was tried, so the benchmark records one clear row
    instead of crashing the run.
    """
    import inspect

    errors: list[str] = []

    # ── Public API (forward-compatible) ──────────────────────────────────────
    fn_name = _PUBLIC_DATASET_FN.get(slug)
    if fn_name:
        try:
            import pyrit.datasets as _pd
            fn = getattr(_pd, fn_name, None)
            if fn is not None:
                res = fn()
                if inspect.isawaitable(res):
                    res = await res
                if res is not None:
                    return res
        except Exception as e:  # noqa: BLE001
            errors.append(f"public {fn_name}: {type(e).__name__}: {e}")

    # ── Private provider classes (current PyRIT 0.9 layout) ───────────────────
    try:
        return await _fetch_dataset_private(slug)
    except Exception as e:  # noqa: BLE001
        errors.append(f"private provider: {type(e).__name__}: {e}")

    raise ImportError(
        f"Could not load dataset {slug!r} via any known PyRIT API. Tried: "
        + " | ".join(errors)
    )


async def _fetch_dataset_private(slug: str) -> Any:
    """Fallback using PyRIT's version-specific private provider classes."""
    from pathlib import Path as _Path

    if slug == "harmbench":
        from pyrit.datasets.seed_datasets.remote.harmbench_dataset import _HarmBenchDataset  # type: ignore
        provider: Any = _HarmBenchDataset()
    elif slug == "advbench":
        # AdvBench ships as a local .prompt YAML file inside the PyRIT package.
        from pyrit.datasets.seed_datasets.local.local_dataset_loader import _LocalDatasetLoader  # type: ignore
        import pyrit.datasets as _pd
        local_path = _Path(_pd.__file__).parent / "seed_datasets" / "local" / "adv_bench.prompt"
        if not local_path.exists():
            raise ImportError(f"AdvBench local file not found at {local_path}")
        provider = _LocalDatasetLoader(file_path=local_path)
    elif slug == "xstest":
        from pyrit.datasets.seed_datasets.remote.xstest_dataset import _XSTestDataset  # type: ignore
        provider = _XSTestDataset()
    elif slug == "forbidden_questions":
        return await _fetch_forbidden_questions()
    else:
        raise ValueError(f"Unknown dataset slug: {slug!r}")

    # PyRIT 0.14 renamed fetch_dataset -> fetch_dataset_async (the old name is
    # kept as a deprecated alias). Prefer the new name; fall back for <0.14.
    fetch = getattr(provider, "fetch_dataset_async", None) or provider.fetch_dataset
    return await fetch(cache=True)


async def _fetch_forbidden_questions() -> Any:
    """Load the Forbidden Questions dataset, working around a PyRIT 0.14.0 bug.

    ``_ForbiddenQuestionsDataset.fetch_dataset_async`` builds each SeedPrompt
    with ``harm_categories=item["content_policy_name"]`` — a single string
    (e.g. "Illegal Activity") — but ``SeedPrompt.harm_categories`` is typed
    ``list[str]``, so pydantic raises a ValidationError on the very first row.
    Re-fetch the raw HuggingFace rows via the provider's own loader helper and
    build the SeedDataset ourselves with harm_categories correctly wrapped in
    a list. Safe to remove once upstream fixes this (see
    https://github.com/Azure/PyRIT — forbidden_questions_dataset.py).
    """
    from pyrit.datasets.seed_datasets.remote.forbidden_questions_dataset import _ForbiddenQuestionsDataset  # type: ignore
    from pyrit.models import SeedDataset, SeedPrompt  # type: ignore

    provider = _ForbiddenQuestionsDataset()
    data = await provider._fetch_from_huggingface_async(
        dataset_name=provider.source, config="default", split="train", cache=True,
    )
    authors = ["Xinyue Shen", "Zeyuan Chen", "Michael Backes", "Yun Shen", "Yang Zhang"]
    description = (
        'This is the Forbidden Question Set dataset proposed in the ACM CCS 2024 paper '
        '"Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts '
        "on Large Language Models."
    )
    seed_prompts = [
        SeedPrompt(
            value=item["question"],
            data_type="text",
            dataset_name=provider.dataset_name,
            authors=authors,
            harm_categories=[item["content_policy_name"]],
            source="https://huggingface.co/datasets/TrustAIRLab/forbidden_question_set",
            description=description,
        )
        for item in data
    ]
    return SeedDataset(seeds=seed_prompts, dataset_name=provider.dataset_name)


def load_local_prompts(name: str, limit: int) -> list[str]:
    """Return up to `limit` prompts from app/data/local_prompts/<name>.json.

    The file is {"_about": ..., "prompts": [{"id", "style", "prompt"}, ...]}.
    Unlike the PyRIT loaders this raises when the file is missing or malformed:
    a local file that has been renamed or mis-shaped is a packaging fault, and a
    silent empty list would report the attack as 0% rather than as broken.
    """
    path = DATA_DIR / "local_prompts" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Local prompt set not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"Local prompt set {path} is not readable JSON: {e}") from e
    rows = data.get("prompts", []) if isinstance(data, dict) else data
    out = [
        (r.get("prompt") if isinstance(r, dict) else r)
        for r in rows
    ]
    out = [t for t in out if isinstance(t, str) and t.strip()]
    if not out:
        raise ValueError(f"Local prompt set {path} contains no usable prompts.")
    return out[:limit]


def load_structured_objectives() -> list[dict]:
    """Load the structured objective/true_description rows from app/data/objectives.json.

    Each row is {id, category, taxonomy_category, objective, true_description}.
    Returns [] if the file is missing or malformed — the caller decides whether
    to fall back to a PyRIT dataset.
    """
    path = DATA_DIR / "objectives.json"
    if not path.exists():
        logging.getLogger(__name__).warning(
            "objectives.json not found at %s — dynamic attacks will fall back to a "
            "PyRIT dataset with the generic true_description.", path)
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("objectives", []) if isinstance(data, dict) else data
        return [r for r in rows if isinstance(r, dict) and r.get("objective")]
    except (OSError, json.JSONDecodeError):
        return []


def load_taxonomy_slugs() -> set[str]:
    """Slugs of taxonomy categories defined in app/data/taxonomy.json (for validation)."""
    path = DATA_DIR / "taxonomy.json"
    if not path.exists():
        logging.getLogger(__name__).warning(
            "taxonomy.json not found at %s — category validation is inert.", path)
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {c["slug"] for c in data.get("categories", []) if c.get("slug")}
    except (OSError, json.JSONDecodeError):
        return set()


def sample_objectives(rows: list[dict], limit: int) -> list[dict]:
    """Pick up to `limit` objectives spread across taxonomy categories.

    The naive `rows[:limit]` biases a capped run toward whatever happens to sit
    at the top of objectives.json — after the set was expanded that is entirely
    the original violence/illegal rows, so categories added later (misinfo,
    privacy, ...) would never be tested at the default cap. Instead this does a
    round-robin over taxonomy_category: one from each category, then a second
    from each, and so on. A capped run therefore covers every category before it
    doubles up on any, which is what makes the per-taxonomy rollup meaningful.

    Deterministic (preserves file order within each category, round-robins
    categories in first-seen order) so re-running a benchmark hits the same
    objectives — reproducibility matters for a report.
    """
    if limit is None or limit <= 0 or limit >= len(rows):
        return rows

    buckets: dict[str, list[dict]] = {}
    order: list[str] = []
    for r in rows:
        cat = r.get("taxonomy_category") or "_uncategorised"
        if cat not in buckets:
            buckets[cat] = []
            order.append(cat)
        buckets[cat].append(r)

    picked: list[dict] = []
    idx = 0
    while len(picked) < limit:
        progressed = False
        for cat in order:
            bucket = buckets[cat]
            if idx < len(bucket):
                picked.append(bucket[idx])
                progressed = True
                if len(picked) >= limit:
                    break
        if not progressed:
            break
        idx += 1
    return picked


async def resolve_objectives(
    attack: dict, limit: int, exclude_categories: set[str] | None = None,
) -> list[dict]:
    """Return a list of {id, objective, true_description, taxonomy_category} rows.

    Honours `objectives_source` on the attack:
      - "json" (default) : load app/data/objectives.json; fall back to the
                           PyRIT dataset if the file is empty / missing.
      - "dataset"        : use the legacy PyRIT dataset path only.

    `exclude_categories` drops objectives whose `category` (the human-facing
    label — "Self-Harm", "Sexual") is listed, before sampling, so the remaining
    categories still get the full round-robin spread rather than a set with
    holes in it. Used by the workshop preset, where a room of students should
    not have those objectives put on a projector. Matched case-insensitively.
    The dataset fallback below cannot be filtered this way — a PyRIT dataset
    carries no category labels — so an excluded run that falls through to it
    returns nothing rather than quietly returning unfiltered prompts.
    """
    source = attack.get("objectives_source", "json")
    excl = {c.strip().lower() for c in (exclude_categories or set()) if c.strip()}
    if source == "json":
        rows = load_structured_objectives()
        if excl:
            rows = [r for r in rows if (r.get("category") or "").lower() not in excl]
        if rows:
            return sample_objectives(rows, limit)
        if excl:
            return []
    # Fallback: synthesise rows from a PyRIT dataset using the generic true_description.
    dataset = attack.get("objectives_dataset")
    if not dataset:
        return []
    prompts = await load_dataset_prompts(dataset, limit)
    return [
        {
            "id": f"ds-{i}",
            "objective": p,
            "true_description": DEFAULT_TRUE_DESC,
            "taxonomy_category": attack.get("taxonomy_category"),
        }
        for i, p in enumerate(prompts)
    ]
