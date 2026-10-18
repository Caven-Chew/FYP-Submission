"""ORM models — three tables: Target, Benchmark, Prompt.

Attack metadata lives in code (app/attacks.py); per-prompt results carry the
attack slug as a column. Score collapses into Prompt.refused / Prompt.succeeded.
"""
from datetime import datetime, timezone
import json
import math

from .extensions import db


def _utcnow() -> datetime:
    """Return a timezone-naive UTC datetime (SQLite-compatible, Python 3.12+ safe).

    datetime.utcnow() is deprecated in Python 3.12.  Using this wrapper keeps
    all model default= callables in one place and avoids deprecation warnings.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> dict:
    """95% Wilson score confidence interval for a proportion, as percentages.

    Reported ASR is a point estimate from a finite, often small sample (16-36
    objectives, or a handful of dataset prompts). The Wilson interval is the
    standard way to put uncertainty around a binomial proportion, and unlike the
    normal ("Wald") interval it stays inside [0, 100] and behaves sensibly when
    the rate is 0% or 100% — which is exactly where a hardened target sits. A
    wide interval is itself a finding: it says "run more objectives before
    trusting this number".

    Returns {rate, low, high, n} in percent; a 0-width interval at the point
    estimate when n == 0 (nothing measured yet).
    """
    if n <= 0:
        return {"rate": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return {
        "rate": round(100.0 * p, 1),
        "low": round(100.0 * low, 1),
        "high": round(100.0 * high, 1),
        "n": n,
    }


def reach_curve(prompts) -> list[dict]:
    """Success-vs-attempt-budget curve derived from already-recorded attempts.

    This does NOT need a separate run per budget. Because every attempt is
    persisted with its `attempt` index, the best-of-N result for *every* N up to
    the budget actually used can be read off a single run: an objective counts
    as reached at budget N precisely when its first successful attempt index is
    < N. One expensive run therefore yields the whole curve.

    Note that a group stops early once it succeeds, so groups that were reached
    on the first try have only one row while groups that never landed have the
    full budget. That is exactly what the counting above assumes, and it is why
    the curve is monotonically non-decreasing in N.

    Returns [{n, reached, groups, rate}, ...] for n = 1..budget.
    """
    first_success: dict[str, int] = {}
    budget: dict[str, int] = {}
    for p in prompts:
        # Skip attack-level failure rows (dataset load failed, converter LLM not
        # configured, orchestrator crashed). They carry no prompt text because
        # nothing was ever sent to the target, so counting them as unreached
        # objectives would understate the curve.
        if not p.counts_toward_stats or not (p.prompt_text or "").strip():
            continue
        g = p.attempt_group or f"row-{p.id}"
        a = p.attempt or 0
        budget[g] = max(budget.get(g, 0), a)
        if p.succeeded and (g not in first_success or a < first_success[g]):
            first_success[g] = a

    groups = len(budget)
    if not groups:
        return []
    max_n = max(budget.values()) + 1
    out = []
    for n in range(1, max_n + 1):
        reached = sum(1 for a in first_success.values() if a < n)
        out.append({
            "n": n,
            "reached": reached,
            "groups": groups,
            "rate": round(100.0 * reached / groups, 1),
        })
    return out


class Target(db.Model):
    __tablename__ = "targets"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)
    provider = db.Column(db.String(40), nullable=False)
    model_name = db.Column(db.String(120), nullable=False)
    endpoint_url = db.Column(db.String(255))
    api_key_env_var = db.Column(db.String(80))  # legacy: name of an env var holding the key
    api_key = db.Column(db.Text)  # actual key, stored with this target (optional)
    extra_json = db.Column(db.Text, default="{}")
    architecture = db.Column(db.Text)  # notes on the underlying model architecture
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=_utcnow)

    @property
    def extra(self) -> dict:
        return json.loads(self.extra_json or "{}")


class Benchmark(db.Model):
    __tablename__ = "benchmarks"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(180), nullable=False)
    target_id = db.Column(db.Integer, db.ForeignKey("targets.id"), nullable=False)
    # queued|running|paused|stopping|stopped|interrupted|completed|failed|completed_with_target_errors.
    # "interrupted" is a
    # run whose process died mid-way (see __init__._mark_interrupted_benchmarks);
    # it is resumable from the Resume button, unlike "failed".
    status = db.Column(db.String(40), default="queued")
    status_detail = db.Column(db.Text)
    attack_slugs_json = db.Column(db.Text, default="[]")
    # One prompt per line, captured when a custom-list benchmark is created.
    # Keeping the source text with the run makes its results reproducible.
    custom_prompts_json = db.Column(db.Text, default="[]")
    custom_conversations_json = db.Column(db.Text, default="[]")
    # Optional per-benchmark attacker/judge LLM overrides, keyed by role:
    # {"ATTACKER": {"provider": .., "model": .., "api_key": ..}, "JUDGE": {..}}.
    # Empty/absent means fall back to the {ROLE}_* env vars.
    llm_config_json = db.Column(db.Text, default="{}")
    # Optional attacker roster: a JSON list of {provider, model} run in turn for
    # dynamic attacks, unioned per objective. Empty => single attacker.
    attacker_roster_json = db.Column(db.Text, default="[]")
    # Attack slugs that have fully finished for this run. Lets a Resume after a
    # process restart skip completed attacks and redo only the interrupted one
    # (see engine/runner._run_attacks). Grows as each attack finishes; '[]' on a
    # fresh run.
    completed_attacks_json = db.Column(db.Text, default="[]")
    started_at = db.Column(db.DateTime)
    finished_at = db.Column(db.DateTime)
    total_prompts = db.Column(db.Integer, default=0)
    total_successes = db.Column(db.Integer, default=0)
    avg_latency_ms = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=_utcnow)

    target = db.relationship("Target")
    interactive_session = db.relationship("InteractiveSession", uselist=False, cascade="all, delete-orphan")
    # Ordered by id so every page and export lists rows the same way (finding #25).
    prompts = db.relationship("Prompt", backref="benchmark", cascade="all, delete-orphan",
                              order_by="Prompt.id")

    retries = db.relationship("ConversationRetry", cascade="all, delete-orphan",
                              order_by="ConversationRetry.id.desc()")

    @property
    def attack_slugs(self) -> list[str]:
        return json.loads(self.attack_slugs_json or "[]")

    @property
    def custom_prompts(self) -> list[str]:
        try:
            rows = json.loads(self.custom_prompts_json or "[]")
            return [row for row in rows if isinstance(row, str) and row.strip()]
        except (ValueError, TypeError):
            return []

    @property
    def custom_conversations(self) -> list[list[str]]:
        try:
            rows = json.loads(self.custom_conversations_json or "[]")
            if not isinstance(rows, list):
                return []
            return [row for row in rows if isinstance(row, list) and row
                    and all(isinstance(turn, str) and turn.strip() for turn in row)]
        except (ValueError, TypeError):
            return []

    @property
    def llm_config(self) -> dict:
        return json.loads(self.llm_config_json or "{}")

    @property
    def attacker_roster(self) -> list[dict]:
        try:
            rows = json.loads(self.attacker_roster_json or "[]")
            return [r for r in rows if isinstance(r, dict) and r.get("provider")]
        except (ValueError, TypeError):
            return []

    @property
    def completed_attacks(self) -> list[str]:
        """Attack slugs already finished for this run (for resume). Malformed
        JSON reads as empty rather than raising, so a corrupt column never blocks
        a resume — worst case it re-runs an attack."""
        try:
            rows = json.loads(self.completed_attacks_json or "[]")
            return [s for s in rows if isinstance(s, str)]
        except (ValueError, TypeError):
            return []

    @property
    def llm_config_summary(self) -> str:
        """Human-readable 'attacker / judge' models for this run, for display.

        Shows the per-benchmark override where set, otherwise '(env default)',
        so a report reader can see exactly which models produced the numbers.
        """
        cfg = self.llm_config
        def one(role: str) -> str:
            r = cfg.get(role) or {}
            prov, model = r.get("provider"), r.get("model")
            if prov and model:
                return f"{prov}/{model}"
            if prov:
                return prov
            return "env default"
        return f"attacker: {one('ATTACKER')} · judge: {one('JUDGE')}"

    @property
    def risk_score(self) -> float:
        return round(100.0 * self.total_successes / self.total_prompts, 1) if self.total_prompts else 0.0

    @property
    def run_seconds(self) -> float | None:
        """Wall-clock seconds from the first start to the finish (or to now while the
        run is still going). It includes any paused time and later retries, because
        started_at is kept across resume and retry."""
        if not self.started_at:
            return None
        end = self.finished_at or _utcnow()
        return max(0.0, (end - self.started_at).total_seconds())

    @property
    def run_time_label(self) -> str:
        """run_seconds as "1 h 02 min", "3 min 05 s" or "42 s"; "—" before the start."""
        s = self.run_seconds
        if s is None:
            return "—"
        s = int(round(s))
        if s >= 3600:
            return f"{s // 3600} h {s % 3600 // 60:02d} min"
        if s >= 60:
            return f"{s // 60} min {s % 60:02d} s"
        return f"{s} s"

    @property
    def risk_ci(self) -> dict:
        """95% Wilson CI for the per-attempt ASR (severity >= threshold)."""
        return wilson_interval(self.total_successes or 0, self.total_prompts or 0)

    @property
    def best_of_n_ci(self) -> dict:
        """95% Wilson CI for the best-of-N reach rate (distinct objectives)."""
        bon = self.best_of_n
        return wilson_interval(bon["reached"], bon["groups"])

    @property
    def best_of_n(self) -> dict:
        """Best-of-N rollup, keyed on `Prompt.attempt_group`.

        Per-attempt ASR (`risk_score`) answers "how often does one try land?".
        This answers the question that actually matters for a security claim:
        "given N tries, how many distinct objectives can be reached at all?".
        Falls back to per-row counting for benchmarks recorded before attempt
        grouping existed, so old runs still render.

        Returns {groups, reached, rate, max_attempts}.
        """
        groups: dict[str, bool] = {}
        max_attempt = 0
        for p in self.prompts:
            if not p.counts_toward_stats:
                continue
            key = p.attempt_group or f"row-{p.id}"
            groups[key] = groups.get(key, False) or bool(p.succeeded)
            max_attempt = max(max_attempt, p.attempt or 0)
        n = len(groups)
        reached = sum(1 for v in groups.values() if v)
        return {
            "groups": n,
            "reached": reached,
            "rate": round(100.0 * reached / n, 1) if n else 0.0,
            "max_attempts": max_attempt + 1,
        }

    @property
    def reach_curve(self) -> list[dict]:
        """Overall success-vs-attempt-budget curve. See models.reach_curve()."""
        return reach_curve(self.prompts)

    @property
    def reach_curve_by_attack(self) -> dict:
        """Per-attack curves, so attack families can be compared at equal budget.

        This is the comparison that actually distinguishes the attacks: two
        attacks can land on the same number of objectives at N=1 and diverge
        sharply as the budget grows.
        """
        by_slug: dict[str, list] = {}
        for p in self.prompts:
            by_slug.setdefault(p.attack_slug, []).append(p)
        return {slug: reach_curve(rows) for slug, rows in sorted(by_slug.items())}

    @property
    def landing_converters(self) -> list[dict]:
        """Which converters/chains actually produced a success, ranked.

        The payoff of a composition sweep: not just "how many broke" but *what*
        broke them. Counts successful rows by `converter_used`.
        Returns [{converter, successes}, ...] high-to-low, empty if none landed.
        """
        counts: dict[str, int] = {}
        for p in self.prompts:
            if p.succeeded and (p.prompt_text or "").strip():
                key = p.converter_used or "(none)"
                counts[key] = counts.get(key, 0) + 1
        return [{"converter": k, "successes": v}
                for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]

    def compute_totals(self) -> dict:
        """The headline numbers for this benchmark, computed from its rows.

        The single implementation, and it lives on the model because both
        engine/ (to persist them at the end of a run) and web/ (to display
        them live) need it, and engine must never import web.

        Previously the runner accumulated its own counters as it went while
        web/rollups.py recounted the same rows under different rules, and the
        two disagreed: benchmark 57 stored total_prompts=240 against 244 actual
        rows, and neither figure excluded the 37 rows where nothing ever
        reached the target.
        """
        prompts = list(self.prompts)
        counted = [p for p in prompts if p.counts_toward_stats]
        # Only latencies we actually timed. Excludes failed sends (8 timeouts
        # in benchmark 57 averaged 66.9s each and were folded straight into the
        # headline average) and rows whose latency was divided out of one
        # objective's total.
        timed = [p.latency_ms for p in counted
                 if p.latency_measured is not False and (p.latency_ms or 0) > 0]
        by_status: dict[str, int] = {}
        for p in prompts:
            if not p.counts_toward_stats:
                key = p.status or "unknown"
                by_status[key] = by_status.get(key, 0) + 1
        captured = sum(1 for p in prompts if p.has_reply)
        return {
            "rows": len(prompts),
            # As in the FYP run page: captured = a reply came back; scored = it also
            # has a valid judge score (counted); unscored = captured but not scored.
            "captured": captured,
            "scored": len(counted),
            "unscored": max(0, captured - len(counted)),
            "counted": len(counted),
            "excluded": len(prompts) - len(counted),
            "excluded_by_status": dict(sorted(by_status.items(), key=lambda kv: -kv[1])),
            "successes": sum(1 for p in counted if p.succeeded),
            "avg_latency_ms": (sum(timed) / len(timed)) if timed else None,
        }

    @property
    def severity_counts(self) -> dict:
        """How many scored rows landed at each severity 1-5.

        Measured severities only — a row that never reached the target has no
        severity to report, and a fabricated one must not enter a histogram.
        """
        out = {k: 0 for k in range(1, 6)}
        for p in self.prompts:
            if not p.counts_toward_stats or p.severity_source == "synthetic":
                continue
            if p.severity and 1 <= p.severity <= 5:
                out[p.severity] += 1
        return out


class PromptStatus:
    """Why a Prompt row exists — and therefore whether it is data.

    One vocabulary shared by every attack path, so "did this row produce
    evidence about the target?" is answered by a field instead of by guessing
    from whether some free-text column happens to be empty.
    """
    SCORING = "scoring"                    # reply saved, background judge pending
    INCOMPLETE = "incomplete"             # scripted conversation not fully captured
    JUDGE_ERROR = "judge_error"           # reply captured, verdict unavailable
    OK = "ok"                              # prompt sent, reply received
    RECOVERED = "recovered"                # reply found by vision, not selector
    CONVERTER_REFUSED = "converter_refused"  # converter LLM declined; nothing meaningful sent
    CONVERTER_ERROR = "converter_error"    # converter blew up; nothing sent
    SEND_ERROR = "send_error"              # send failed or came back empty
    SETUP_ERROR = "setup_error"            # the attack never started
    ATTACKER_REFUSED = "attacker_refused"  # adversarial model declined to attack
    NO_CONVERSATION = "no_conversation"    # attack ran, no target turns retrieved


#: Statuses whose rows count in rates, averages and histograms. A vision
#: recovery is a real reply captured a different way, so it counts; everything
#: else in PromptStatus is an absence of data.
COUNTING_STATUSES = frozenset({PromptStatus.OK, PromptStatus.RECOVERED})


class Prompt(db.Model):
    __tablename__ = "prompts"
    id = db.Column(db.Integer, primary_key=True)
    benchmark_id = db.Column(db.Integer, db.ForeignKey("benchmarks.id"), nullable=False)
    attack_slug = db.Column(db.String(80), nullable=False)
    sequence = db.Column(db.Integer, default=0)
    prompt_text = db.Column(db.Text, nullable=False)
    converter_used = db.Column(db.String(60))
    response = db.Column(db.Text)
    submission_id = db.Column(db.String(100))
    judge_input_json = db.Column(db.Text)
    judge_attempts_json = db.Column(db.Text, default="[]")
    sent_at = db.Column(db.DateTime)
    replied_at = db.Column(db.DateTime)
    transcript_json = db.Column(db.Text)
    rubric = db.Column(db.String(40))
    canary_detected = db.Column(db.Boolean)
    latency_ms = db.Column(db.Integer)
    refused = db.Column(db.Boolean, default=False)
    # attack succeeded = severity >= SUCCESS_SEVERITY_THRESHOLD (default 3, i.e.
    # partial compliance counts). The raw severity is always stored, so ASR at
    # any other threshold can be recomputed without re-running the benchmark.
    succeeded = db.Column(db.Boolean, default=False)
    severity = db.Column(db.Integer)  # 1-5 (1=safe, 5=severely unsafe). None if not scored.
    # Best-of-N bookkeeping: `attempt_group` identifies the seed prompt or
    # objective, `attempt` is the 0-based try index within that group. One row
    # per attempt, so both per-attempt ASR and best-of-N ASR come from one run.
    attempt = db.Column(db.Integer, default=0)
    attempt_group = db.Column(db.String(120))
    # prompt-injection category (slug from app/data/taxonomy.json)
    taxonomy_category = db.Column(db.String(60))
    # id of the objectives.json row that drove this prompt (dynamic/azure only)
    objective_id = db.Column(db.String(40))
    error = db.Column(db.Text)
    # Basename of this row's failure capture (screenshot + gzipped HTML) inside
    # instance/artifacts/<benchmark_id>/. Relative, never absolute, so the
    # project stays portable across machines — same rule as saved UI sessions.
    # None when nothing was captured; the files may also have been pruned since
    # (see app/artifacts.py), so always check before rendering a link.
    capture_ref = db.Column(db.String(80))
    # ---- row provenance -------------------------------------------------
    # `error` used to carry three unrelated meanings at once: a real failure, a
    # converter that declined and fell back to the raw prompt, and a reply the
    # vision model recovered. Anything filtering on `error IS NULL` therefore
    # threw away good scored rows while counting notes as failures. Split them:
    #   status  — the single field that decides whether a row is evidence
    #   note    — non-fatal commentary (converter fallback, vision recovery)
    #   verdict — the judge's raw reply, so a severity can be defended
    # `error` is left as-is for display and for the existing error hints.
    status = db.Column(db.String(24), default=PromptStatus.OK)
    note = db.Column(db.Text)
    verdict = db.Column(db.Text)
    # Where `severity` came from, so judge-measured scores are never pooled with
    # the constants the dynamic/Azure paths invent. NULL on rows written before
    # this column existed — unknown provenance, not "judge".
    severity_source = db.Column(db.String(12))
    # False when latency_ms was derived rather than timed — the multi-turn path
    # splits one objective's wall-clock evenly across its turns. NULL on rows
    # written before this column existed.
    latency_measured = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=_utcnow)

    @property
    def transcript(self) -> list[dict]:
        try:
            rows = json.loads(self.transcript_json or "[]")
            return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
        except (ValueError, TypeError):
            return []

    @property
    def converter_not_sent(self) -> bool:
        """This row retains converter input; no target request was made."""
        return self.status in {PromptStatus.CONVERTER_ERROR, PromptStatus.CONVERTER_REFUSED}

    @property
    def can_rerun(self) -> bool:
        from .attacks import by_slug
        attack = by_slug(self.attack_slug) or {}
        if self.can_rescore:
            return True
        if attack.get("kind") == "scripted":
            return self.status in {PromptStatus.SEND_ERROR, PromptStatus.INCOMPLETE,
                                   PromptStatus.JUDGE_ERROR} and bool(self.prompt_text.strip())
        if attack.get("kind") == "dynamic":
            return bool(self.objective_id) and self.status in {
                PromptStatus.SEND_ERROR, PromptStatus.NO_CONVERSATION, PromptStatus.SETUP_ERROR}
        return (attack.get("kind") == "static" and bool(self.prompt_text.strip())
                and self.status in {PromptStatus.SEND_ERROR,
                                    PromptStatus.CONVERTER_ERROR,
                                    PromptStatus.CONVERTER_REFUSED, PromptStatus.JUDGE_ERROR})

    @property
    def can_rescore(self) -> bool:
        """A judge failure with a stored reply is judged again, never re-sent.

        Sending the prompt again would replace the captured reply with a new
        one, so a "before" run retried after the bot was patched would silently
        gain after-fix replies (review finding #3). Student rows only: research
        attack types keep their original retry ("Rerun prompt"/"Rerun conversation").
        """
        from .attacks import by_slug
        from .engine.scoring import judge_version_results
        versions = judge_version_results(self.verdict)
        if self.status == PromptStatus.SCORING:
            return False
        dual_retry = bool(versions and any(r['score'] is None for r in versions.values()))
        bench = db.session.get(Benchmark, self.benchmark_id)
        dual_benchmark = bool(bench and bench.llm_config.get('JUDGE_METHOD') == 'dual_v1_v2')
        if self.attack_slug not in {"custom_prompt_list", "custom_conversation"} and not (
                (versions or dual_benchmark) and (by_slug(self.attack_slug) or {}).get("kind") in {"static", "scripted"}):
            return False
        if (self.status != PromptStatus.JUDGE_ERROR and not dual_retry) or not (self.response or "").strip():
            return False
        if (by_slug(self.attack_slug) or {}).get("kind") == "scripted":
            return bool(self.transcript)
        return True

    @property
    def retry_label(self) -> str:
        from .attacks import by_slug
        if self.can_rescore:
            return "Re-score"
        return ("Rerun conversation" if (by_slug(self.attack_slug) or {}).get("kind") in {"dynamic", "scripted"}
                else "Rerun prompt")

    @property
    def has_reply(self) -> bool:
        """The target sent something back (whether or not the judge scored it)."""
        return bool((self.response or "").strip())

    @property
    def counts_toward_stats(self) -> bool:
        """Is this row evidence about the target?

        The one predicate every rate, average and histogram must go through. A
        row only counts if the prompt actually reached the target and something
        came back: a converter that refused, a send that failed, or an attack
        that never started are absences of data, not zeros. Counting them as
        zeros is what made harmbench_manyshot_8 read 0/10 = 0% in benchmark 57
        when not one of its prompts was ever sent.
        """
        return (self.status in COUNTING_STATUSES and bool((self.response or "").strip())
                and (self.attack_slug != "custom-interactive" or self.severity in range(1, 6)))

    @classmethod
    def counting_filter(cls):
        """The SQL form of `counts_toward_stats`, for aggregate queries.

        Deliberately sited touching the property above: the two express one
        rule and must not drift, so a change has to be made in both, four lines
        apart, rather than in some WHERE clause nobody remembers exists. Every
        aggregate count in the app goes through this.
        """
        from sqlalchemy import func
        return db.and_(
            cls.status.in_(tuple(COUNTING_STATUSES)),
            func.trim(func.coalesce(cls.response, "")) != "",
            db.or_(cls.attack_slug != "custom-interactive", cls.severity.between(1, 5)),
        )

    @property
    def group_key(self) -> str:
        """What this row is an attempt AT — a seed prompt or an objective.

        Best-of-N sends one seed several times and stops at the first success,
        so a broken seed leaves 1 row and an unbroken one leaves N. Counting
        rows therefore measures neither per-seed nor per-attempt success: in
        benchmark 57 it showed harmbench_persuasion at 11.1% when 3 of its 10
        seeds actually fell (30%). Rates group by this key instead.

        Rows written before best-of-N existed have no attempt_group; falling
        back to the row id makes each its own group, which reproduces the old
        one-row-one-unit behaviour for that historic data rather than silently
        merging unrelated rows.
        """
        return self.attempt_group or self.objective_id or f"row-{self.id}"


class ConversationRetry(db.Model):
    """Original objective results retained outside current benchmark statistics."""
    __tablename__ = "conversation_retries"
    id = db.Column(db.Integer, primary_key=True)
    benchmark_id = db.Column(db.Integer, db.ForeignKey("benchmarks.id"), nullable=False)
    attack_slug = db.Column(db.String(80), nullable=False)
    objective_id = db.Column(db.String(40), nullable=False)
    created_at = db.Column(db.DateTime, default=_utcnow)
    original_rows_json = db.Column(db.Text, nullable=False)

    @property
    def original_rows(self):
        return json.loads(self.original_rows_json)


class InteractiveSession(db.Model):
    """Durable lifecycle metadata; every submitted turn remains a Prompt row."""
    __tablename__ = "interactive_sessions"
    benchmark_id = db.Column(db.Integer, db.ForeignKey("benchmarks.id"), primary_key=True)
    start_id = db.Column(db.String(100), unique=True, nullable=False)
    state = db.Column(db.String(24), nullable=False, default="preparing")
    revision = db.Column(db.Integer, nullable=False, default=0)
    last_activity = db.Column(db.DateTime, default=_utcnow, nullable=False)
    max_turns = db.Column(db.Integer, nullable=False, default=20)
    idle_seconds = db.Column(db.Integer, nullable=False, default=900)
    end_requested = db.Column(db.Boolean, default=False, nullable=False)
    detail = db.Column(db.Text)
