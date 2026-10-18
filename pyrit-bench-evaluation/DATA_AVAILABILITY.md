# Data availability and reproduction boundaries

| Area | Included | Separate input or limitation |
|---|---|---|
| S1 fixed-response comparison | Text-free population metadata, projected prediction labels, expected full analysis and preparation/analysis code | Obtain the pinned StrongREJECT human-rating CSV separately and generate preparation outside Git. Numeric reference ratings/raw corpus are not redistributed. |
| Historical thresholds/attempts | Aggregate results, source tool and synthetic SQLite tests | Production database is withheld; original source selection cannot be rerun from aggregates alone. |
| Controlled practice-bot benchmark | Aggregate results, design, offline analysis code and synthetic tests | Exact original loader needs the reviewed full response capture to re-hash text and rerun marker detection. That corpus is not included; aggregates do not recreate it. |
| Reason review | Tally code and synthetic tests | Original row-level review/evidence remains outside this repository. |
| Workshop/course feedback | Analysis methods and synthetic form tests/configuration | Private respondent exports, timestamps, free text and row-level derivatives are excluded. |
| Application verification | Source tests in pyrit-bench | Historical evidence is separate from rerunning those tests on this snapshot. |

The normalization projection drops stage-level metadata from prediction JSONL files and preserves IDs, validity and decisions used by analysis. This is an explicitly derived analysis input, not a claim that the original files' bytes are identical.

The original StrongREJECT human CSV is https://osf.io/download/jwmqe/ (SHA-256 7a5928a4f09b4cbfce274001873ccf095510d5a990f8be52e4323d4c069fbad2). Source repository: https://github.com/dsbowen/strong_reject. The upstream prompt repository's MIT statement does not, by itself, establish a blanket licence for the separate human corpus or inherited prompt sources. Download under the source's terms rather than publishing a copied corpus from this repository.

No full-report reproduction claim is made. S1 statistics can be recomputed with the separately obtained numeric reference. Historical SQL selection, original response marker re-detection and participant-level feedback cannot be independently rerun from the included aggregates.
