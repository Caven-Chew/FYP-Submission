"""Attack catalog.

Each attack is one of three kinds:

  - kind="static"        : prompts come from a PyRIT dataset (HarmBench, etc.),
                           optionally passed through a converter. `converter`
                           takes one slug or a comma-separated chain applied in
                           order, e.g. "tense,poetry".
  - kind="dynamic"       : prompts come from a PyRIT orchestrator (Red-Teaming /
                           Crescendo / TAP / PAIR) driven by an attacker LLM,
                           fed the structured objectives in
                           app/data/objectives.json. `orchestrator` picks which:
                           "red_teaming" and "crescendo" take `max_turns`;
                           "tap" and "pair" take `tree_width` / `tree_depth`
                           (and TAP also `branching_factor`).
  (kind="azure_redteam" existed but its catalog entries were removed 2026-08-18
   — the azure-ai-evaluation[redteam] extra pins pyrit==0.11.0, incompatible with
   the pyrit==0.14.0 TAP/PAIR need. The runner branch + engine/azure_attack.py
   are kept inert; re-add catalog entries if a compatible azure release lands.)

`taxonomy_category` is the slug from app/data/taxonomy.json used to tag every
Prompt row an attack produces, so reports can group by it.

`sweep` (optional, static kind) is a list of converter chains; each prompt is
tried against every chain and counts as broken if any lands, with the chain that
worked recorded in `converter_used`. When set it supersedes `converter`/`attempts`.

`attempts` (optional, any kind) is the best-of-N budget for this attack: the
prompt/objective is retried up to that many times and counts as reached if any
attempt lands. Defaults to ATTEMPTS_PER_PROMPT in .env (which itself defaults
to 1). Cost scales roughly linearly with it.

Add a new attack by appending a dict — the UI picks it up automatically.
"""

ATTACKS = [
    # ----- Static dataset attacks (use PyRIT datasets directly) -----
    {
        "slug": "harmbench",
        "tier": 1,
        "name": "HarmBench harmful behaviors",
        "category": "Static dataset",
        "taxonomy_category": "direct_harmful",
        "kind": "static",
        "dataset": "harmbench",
        "converter": None,
    },
    {
        "slug": "advbench",
        "tier": 1,
        "name": "AdvBench harmful prompts",
        "category": "Static dataset",
        "taxonomy_category": "direct_harmful",
        "kind": "static",
        "dataset": "advbench",
        "converter": None,
    },
    {
        "slug": "xstest",
        "tier": 1,
        "name": "XSTest exaggerated-safety probes",
        "category": "Static dataset",
        "taxonomy_category": "context_framing",
        "kind": "static",
        "dataset": "xstest",
        "converter": None,
    },
    {
        "slug": "forbidden_questions",
        "tier": 1,
        "name": "Forbidden Questions dataset",
        "category": "Static dataset",
        "taxonomy_category": "direct_harmful",
        "kind": "static",
        "dataset": "forbidden_questions",
        "converter": None,
    },

    # ----- Encoded / obfuscated attacks (static prompts + converter) -----
    {
        "slug": "harmbench_base64",
        "tier": 1,
        "name": "HarmBench encoded as Base64",
        "category": "Encoding",
        "taxonomy_category": "encoding_obfuscation",
        "kind": "static",
        "dataset": "harmbench",
        "converter": "base64",
    },
    {
        "slug": "harmbench_leetspeak",
        "tier": 1,
        "name": "HarmBench obfuscated as leetspeak",
        "category": "Encoding",
        "taxonomy_category": "encoding_obfuscation",
        "kind": "static",
        "dataset": "harmbench",
        "converter": "leetspeak",
    },
    {
        "slug": "harmbench_rot13",
        "tier": 1,
        "name": "HarmBench ROT13-encoded",
        "category": "Encoding",
        "taxonomy_category": "encoding_obfuscation",
        "kind": "static",
        "dataset": "harmbench",
        "converter": "rot13",
    },

    # ----- Semantic reframing (static prompts + LLM-driven converter) -----
    # These need an LLM to do the rewriting: ATTACKER_PROVIDER (preferred) or
    # JUDGE_PROVIDER must be set in .env. Unlike the encoding converters above,
    # they preserve the request in readable form and attack how it is framed —
    # the family that still lands on safety-hardened 2026-era models.
    #
    # `attempts` sets a best-of-N budget: the rewrite is stochastic, so each
    # attempt is a different poem/framing and a seed only counts as failed once
    # every attempt has failed. Raise it to trade API spend for a stronger,
    # lower-variance signal; sweep it to produce an ASR-vs-budget curve.
    {
        "slug": "harmbench_poetry",
        "tier": 2,
        "name": "HarmBench rewritten as adversarial poetry",
        "category": "Semantic reframing",
        "taxonomy_category": "semantic_reframing",
        "kind": "static",
        "dataset": "harmbench",
        "converter": "poetry",
        "attempts": 3,
    },
    {
        "slug": "harmbench_persuasion",
        "tier": 2,
        "name": "HarmBench with expert-endorsement persuasion framing",
        "category": "Semantic reframing",
        "taxonomy_category": "semantic_reframing",
        "kind": "static",
        "dataset": "harmbench",
        "converter": "persuasion",
        "attempts": 3,
    },
    {
        "slug": "harmbench_tense",
        "tier": 2,
        "name": "HarmBench reframed in the past tense",
        "category": "Semantic reframing",
        "taxonomy_category": "semantic_reframing",
        "kind": "static",
        "dataset": "harmbench",
        "converter": "tense",
        "attempts": 3,
    },
    # Composed chain. Vendors patch single transforms; combinations are far less
    # covered, so a chain usually beats either link on its own. This is the
    # comparison worth putting in the report: tense vs poetry vs tense+poetry.
    {
        "slug": "harmbench_tense_poetry",
        "tier": 3,
        "name": "HarmBench past tense + poetry (composed chain)",
        "category": "Semantic reframing",
        "taxonomy_category": "semantic_reframing",
        "kind": "static",
        "dataset": "harmbench",
        "converter": "tense,poetry",
        "attempts": 3,
    },

    {
        "slug": "harmbench_math",
        "tier": 2,
        "name": "HarmBench recast as a symbolic maths problem",
        "category": "Semantic reframing",
        "taxonomy_category": "semantic_reframing",
        "kind": "static",
        "dataset": "harmbench",
        "converter": "math",
        "attempts": 3,
    },
    {
        "slug": "harmbench_lowresource",
        "tier": 2,
        "name": "HarmBench translated into a low-resource language",
        "category": "Semantic reframing",
        "taxonomy_category": "semantic_reframing",
        "kind": "static",
        "dataset": "harmbench",
        # Override the language with TRANSLATE_LANGUAGE in .env, or write it
        # inline here as "translate:Swahili".
        "converter": "translate",
        "attempts": 3,
    },
    {
        "slug": "harmbench_persuasion_tense",
        "tier": 3,
        "name": "HarmBench logical-appeal framing + past tense (composed chain)",
        "category": "Semantic reframing",
        "taxonomy_category": "semantic_reframing",
        "kind": "static",
        "dataset": "harmbench",
        "converter": "persuasion:logical_appeal,tense",
        "attempts": 3,
    },
    {
        "slug": "harmbench_codechameleon",
        "tier": 2,
        "name": "HarmBench as a CodeChameleon decode-then-obey puzzle",
        "category": "Encoding",
        "taxonomy_category": "encoding_obfuscation",
        "kind": "static",
        "dataset": "harmbench",
        # Unlike base64/rot13 the model must reason to recover the request,
        # which is why this survives filters that catch flat encodings. No LLM
        # needed for the transform itself.
        "converter": "codechameleon:binary_tree",
        "attempts": 1,
    },
    {
        "slug": "harmbench_variation_bon",
        "tier": 2,
        "name": "HarmBench rephrased — best-of-N random search",
        "category": "Semantic reframing",
        "taxonomy_category": "semantic_reframing",
        "kind": "static",
        "dataset": "harmbench",
        # The rephrase is a weak attack on its own. The point is the budget:
        # each attempt is a genuinely different prompt, so this measures how far
        # plain random search gets you and acts as the control every other
        # best-of-N result should be compared against.
        "converter": "variation",
        "attempts": 10,
    },

    # ----- Converter-composition sweep -----
    # Rather than committing to one transform, try a battery of chains against
    # each prompt and count it broken if ANY lands — then read off which chain
    # did it. The consistent 2026 finding is that single transforms are patched
    # but stacked ones are not, so the winning chain is often one you would not
    # have picked by hand. `converter_used` on each row records the chain tried,
    # and the reach curve becomes "objectives broken vs number of chains tried".
    #
    # COST: len(sweep) sends per prompt. The LLM-backed chains also cost a
    # converter call each. Start with MAX_PROMPTS_PER_ATTACK low.
    {
        "slug": "harmbench_composition_sweep",
        "tier": 3,
        "name": "HarmBench — converter composition sweep",
        "category": "Composition sweep",
        "taxonomy_category": "semantic_reframing",
        "kind": "static",
        "dataset": "harmbench",
        "sweep": [
            "tense",
            "poetry",
            "persuasion:expert_endorsement",
            "tense,poetry",
            "persuasion:logical_appeal,tense",
            "tense,poetry,persuasion:expert_endorsement",
            "translate,poetry",
            "manyshot:32",
        ],
    },

    # ----- Many-shot jailbreak (in-context learning attack) -----
    # Attacks a different mechanism from every other entry: it floods the context
    # with fabricated compliant turns rather than rephrasing the request, so it
    # can land on a model that resists all single-prompt tricks. Swept at three
    # shot counts because effectiveness scales with the number of demonstrations
    # — the count where it starts landing is itself a finding. No LLM needed.
    {
        "slug": "harmbench_manyshot_8",
        "tier": 3,
        "name": "HarmBench many-shot jailbreak (8 shots)",
        "category": "Many-shot",
        "taxonomy_category": "multi_turn_escalation",
        "kind": "static",
        "dataset": "harmbench",
        "converter": "manyshot:8",
        "attempts": 1,
    },
    {
        "slug": "harmbench_manyshot_32",
        "tier": 3,
        "name": "HarmBench many-shot jailbreak (32 shots)",
        "category": "Many-shot",
        "taxonomy_category": "multi_turn_escalation",
        "kind": "static",
        "dataset": "harmbench",
        "converter": "manyshot:32",
        "attempts": 1,
    },
    {
        "slug": "harmbench_manyshot_poetry",
        "tier": 3,
        "name": "HarmBench many-shot + poetry (composed)",
        "category": "Many-shot",
        "taxonomy_category": "multi_turn_escalation",
        "kind": "static",
        "dataset": "harmbench",
        # Stack the in-context attack with a phrasing attack on the final turn.
        "converter": "manyshot:32,poetry",
        "attempts": 3,
    },

    # ----- System-prompt extraction (local prompt set, leak-scored) -----
    # dataset="local:<name>" reads app/data/local_prompts/<name>.json instead of
    # calling PyRIT — no public dataset covers this family. scoring="leak" swaps
    # the 1-5 harm rubric for the 1-5 DISCLOSURE rubric: these prompts request
    # nothing harmful, so a harm judge scores a complete verbatim leak a 1 and
    # the attack would report 0% against a target that folded every time.
    {
        "slug": "sysprompt_extraction",
        "tier": 1,
        "name": "System-prompt extraction",
        "category": "System-prompt extraction",
        "taxonomy_category": "instruction_override",
        "kind": "static",
        "dataset": "local:system_prompt_extraction",
        "converter": None,
        "scoring": "leak",
    },
    {
        "slug": "sysprompt_extraction_persuasion",
        "tier": 2,
        "name": "System-prompt extraction (persuasive reframing)",
        "category": "System-prompt extraction",
        "taxonomy_category": "instruction_override",
        "kind": "static",
        "dataset": "local:system_prompt_extraction",
        # Same probes rewritten by the attacker LLM as an expert/authority
        # appeal. Reads against the plain set above: identical objectives, one
        # transform apart, which is the whole tier-1-vs-tier-2 comparison in a
        # family that produces no harmful output.
        "converter": "persuasion:expert_endorsement",
        "scoring": "leak",
    },
    {
        "slug": "sysprompt_extraction_base64",
        "tier": 1,
        "name": "System-prompt extraction (base64)",
        "category": "System-prompt extraction",
        "taxonomy_category": "instruction_override",
        "kind": "static",
        "dataset": "local:system_prompt_extraction",
        "converter": "base64",
        "scoring": "leak",
    },

    # ----- Dynamic / orchestrator-driven attacks -----
    # objectives_source="json" loads app/data/objectives.json so each objective
    # carries its own true_description; falls back to objectives_dataset only if
    # the json is missing.
    {
        "slug": "redteam_dynamic",
        "tier": 4,
        "name": "Red-Teaming orchestrator (structured objectives + attacker LLM)",
        "category": "Dynamic (attacker LLM)",
        "taxonomy_category": "role_play_jailbreak",
        "kind": "dynamic",
        "orchestrator": "red_teaming",
        "objectives_source": "json",
        "objectives_dataset": "harmbench",  # fallback
        "max_turns": 5,
    },
    {
        "slug": "crescendo_dynamic",
        "tier": 4,
        "name": "Crescendo multi-turn escalation",
        "category": "Dynamic (attacker LLM)",
        "taxonomy_category": "multi_turn_escalation",
        "kind": "dynamic",
        "orchestrator": "crescendo",
        "objectives_source": "json",
        "objectives_dataset": "harmbench",
        # The published Crescendo protocol escalates over ~10 turns; stopping at
        # 7 truncates the run just as it reaches the turns that tend to land.
        "max_turns": 10,
    },

    # ----- Search-based orchestrators (TAP / PAIR) -----
    # Crescendo and RedTeaming commit to a single line of attack: if the first
    # few turns go the wrong way, the remaining turns are spent on a dead end.
    # TAP instead expands several refinements per round, scores them all, keeps
    # the best `tree_width` and prunes the rest — so budget follows whichever
    # branch is actually making progress. It is the strongest automated
    # black-box attack PyRIT ships. PAIR is the same algorithm with the tree
    # collapsed to parallel single-line refinement streams; it is cheaper and a
    # useful control — the TAP-vs-PAIR gap isolates what the branching itself
    # buys you, which is a clean comparison for the report.
    #
    # COST WARNING. Target calls per objective are roughly
    #     tree_width x branching_factor x tree_depth
    # plus a scoring call per node. At the defaults below that is up to ~30
    # target calls and ~30 judge calls PER OBJECTIVE — around 30x a Crescendo
    # run. Start with MAX_PROMPTS_PER_ATTACK=2 and scale up once you have
    # measured what one objective actually costs you.
    {
        "slug": "tap_dynamic",
        "tier": 4,
        "name": "Tree of Attacks with Pruning (branching search)",
        "category": "Dynamic (attacker LLM)",
        "taxonomy_category": "multi_turn_escalation",
        "kind": "dynamic",
        "orchestrator": "tap",
        "objectives_source": "json",
        "objectives_dataset": "harmbench",
        "tree_width": 3,        # branches kept per round after pruning
        "tree_depth": 5,        # refinement rounds
        "branching_factor": 2,  # refinements generated per surviving branch
        "on_topic_checking": True,  # prune branches that drifted off-objective
        # Pinned to 1: TAP already searches internally, so an outer best-of-N
        # would multiply an already expensive run. Widen the tree instead.
        "attempts": 1,
    },
    {
        "slug": "pair_dynamic",
        "tier": 4,
        "name": "PAIR (parallel iterative refinement)",
        "category": "Dynamic (attacker LLM)",
        "taxonomy_category": "multi_turn_escalation",
        "kind": "dynamic",
        "orchestrator": "pair",
        "objectives_source": "json",
        "objectives_dataset": "harmbench",
        "tree_width": 3,   # N in the PAIR paper: parallel refinement streams
        "tree_depth": 5,   # K in the PAIR paper: refinements per stream
        "attempts": 1,
    },

    # Azure AI Evaluation RedTeam attacks were removed 2026-08-18: the
    # azure-ai-evaluation[redteam] extra hard-pins pyrit==0.11.0, which is
    # incompatible with the pyrit==0.14.0 that TAP/PAIR and dynamic scoring
    # require. Their coverage (Baseline + base64/rot13/leetspeak/unicode +
    # jailbreak templates) is already provided by the local converters and the
    # TAP/PAIR/Crescendo/RedTeaming dynamic attacks. The engine module
    # engine/azure_attack.py is kept (inert) so its Stage-2 regression tests
    # still run; re-add catalog entries here if a compatible azure release lands.
]


# ---------- potency tiers -------------------------------------------------
#
# Each attack carries a `tier` (1-4) ranking how hard it pushes on a
# safety-hardened model. This is an *attack sophistication* ladder, not a
# statement about how harmful the objective is — every attack chases the same
# objectives; they differ in how far they bend the request out of the shape
# safety training was tuned on. It also mirrors the escalation story the report
# tells: baselines that should fail, single transforms that start to land,
# composed/in-context attacks, and adaptive attacker-LLM search.
#
#   1  Baseline           direct dataset prompts + flat encodings. Expected to
#                         score ~0% on a 2026-era aligned model; the control.
#   2  Single transform   one semantic/reasoning reframe (poetry, tense,
#                         persuasion, math, low-resource, CodeChameleon, BoN).
#   3  Composed / in-ctx  stacked converter chains + many-shot; single
#                         transforms are patched, combinations far less so.
#   4  Adaptive           attacker-LLM orchestrators that search and escalate
#                         (Red-Teaming, Crescendo, TAP, PAIR). Strongest.
#
# `slug` in TIER_META gives the short badge text; ordering here is the order
# tiers render in the UI. An attack with no `tier` (e.g. a newly appended dict
# that forgot one) falls into FALLBACK_TIER so it can never silently vanish from
# the form.
TIER_META = {
    1: {"slug": "Tier 1", "label": "Baseline",
        "desc": "Direct dataset prompts and flat encodings. Expected to fail on "
                "an aligned model — the control every stronger attack is read against."},
    2: {"slug": "Tier 2", "label": "Single-transform reframing",
        "desc": "One semantic or reasoning reframe that keeps the request legible "
                "but moves it off the shapes safety training was tuned on."},
    3: {"slug": "Tier 3", "label": "Composed & in-context",
        "desc": "Stacked converter chains and many-shot. Vendors patch single "
                "transforms; combinations are far less covered."},
    4: {"slug": "Tier 4", "label": "Adaptive multi-turn (attacker LLM)",
        "desc": "Attacker-LLM orchestrators that search and escalate over turns "
                "(Red-Teaming, Crescendo, TAP, PAIR). The strongest attacks here."},
}
FALLBACK_TIER = 2


def by_tier() -> list[tuple[int, dict, list]]:
    """Group ATTACKS by potency tier for display.

    Returns an ordered list of (tier_number, tier_meta, attacks) following the
    order of TIER_META, skipping tiers that have no attacks. Any attack whose
    `tier` is missing or unknown is bucketed into FALLBACK_TIER so appending a
    dict without a tier can never drop it from the form.
    """
    buckets: dict[int, list] = {t: [] for t in TIER_META}
    for a in ATTACKS:
        t = a.get("tier")
        if t not in buckets:
            t = FALLBACK_TIER
        buckets[t].append(a)
    return [(t, TIER_META[t], buckets[t]) for t in TIER_META if buckets[t]]


# ---------- presets -------------------------------------------------------
#
# A preset is a named, one-click configuration of the New Benchmark form: which
# attacks are ticked, how many prompts each one sends, and which objective
# categories are withheld. It exists because the default form runs the whole
# catalog at ten prompts an attack, which is the right default for a research
# run and the wrong one for a room of thirty students on shared wifi with a
# thirty-minute lab.
#
# The Workshop UI starts with this set locked. A facilitator can unlock the
# complete catalog for the current browser session. Two fields carry through
# to the run itself rather than just pre-ticking boxes —
#   prompts_per_attack   -> llm_config["MAX_PROMPTS_PER_ATTACK"], read by the
#                           runner in place of the .env cap for this benchmark
#                           only, so the workshop does not require editing .env
#                           on the day and then remembering to put it back.
#   exclude_objective_categories
#                        -> llm_config["EXCLUDE_OBJECTIVE_CATEGORIES"], applied
#                           in datasets.resolve_objectives. Only bites on
#                           dynamic attacks (static datasets carry no category
#                           labels), which the workshop preset excludes anyway —
#                           it is there so ticking one by hand on the day still
#                           does the right thing.
PRESETS = {
    "workshop": {
        "label": "Workshop",
        "summary": "One-click configuration for the student workshop lab: "
                   "1 prompt per attack, four attack families, self-harm and "
                   "sexual objectives withheld. Runs in about five minutes.",
        "prompts_per_attack": 1,
        # Four families, one attack each, chosen so the result screen shows a
        # mechanism per row rather than three flavours of the same one:
        #   baseline        — what alignment does reliably stop, the control
        #   encoding        — the filter-vs-model teaching point
        #   reframing       — the highest-yield single-turn transform
        #   extraction      — the centrepiece; every participant has a system
        #                     prompt and every one assumes it is private
        # Deliberately absent: the dynamic orchestrators (too slow for a
        # thirty-minute lab, and non-deterministic enough that two neighbours
        # comparing numbers would draw the wrong conclusion), the sweeps and
        # many-shot (judge-token cost), and the composed chains.
        "attack_slugs": [
            "harmbench",
            "harmbench_base64",
            "harmbench_persuasion",
            "sysprompt_extraction",
        ],
        # Withheld from a teaching room with students in it. Nothing is lost
        # pedagogically — the mechanisms are identical whatever the objective
        # is about — and it should be said out loud rather than done quietly,
        # because "we removed these and here is why" is itself part of the
        # lesson about scope.
        "exclude_objective_categories": ["Self-Harm", "Sexual"],
    },
}


def preset(name: str) -> dict | None:
    return PRESETS.get(name)


def by_slug(slug: str) -> dict | None:
    return next((a for a in ATTACKS if a["slug"] == slug), None)


def categories() -> list[str]:
    seen, out = set(), []
    for a in ATTACKS:
        if a["category"] not in seen:
            seen.add(a["category"]); out.append(a["category"])
    return out
