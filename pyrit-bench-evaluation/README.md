# PyRIT Bench evaluation

Saved-data analysis tools and selected evidence supporting the FYP's measurement comparisons. This folder complements the [pyrit-bench application](../pyrit-bench/README.md) and [practice-chatbots targets](../practice-chatbots/README.md).

## Environment and offline tests

Use Python 3.12. Core S1/threshold calculations use the standard library; optional charts and spreadsheet methods use requirements.txt.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -B tools/run_submission_tests.py
```

Tests use synthetic input and no hosted models. Two original-corpus tests skip when that separately supplied corpus is absent. This repository supports saved-data analysis; live collection/judging services are not included.

## Reproduce the S1 numeric analysis

Obtain the pinned CSV from the original source under its terms (see DATA_AVAILABILITY.md). Keep both the download and generated preparation OUTSIDE this Git repository, because preparation also emits raw objective/reply text:

```sh
python -B tools/study_s1_prepare_public_human.py prepare --source-csv /path/outside/repository/strongreject-human.csv --expected-sha256 7a5928a4f09b4cbfce274001873ccf095510d5a990f8be52e4323d4c069fbad2 --seed fyp-s1r2-public-human-2026-09-28-v1 --dev-objectives 5 --subsample 300 --repeat 40 --out-dir /path/outside/repository/s1-preparation
python -B tools/reproduce_s1.py --prep-dir /path/outside/repository/s1-preparation --out-dir /path/outside/repository/s1-results
```

The numeric inputs must hash to population.jsonl = 34b1d10e739328cd392fe9e6314a9af3d330b486b316983c65480fb4f0a46ea1 and private/external_reference.jsonl = 3cf114e96ec15a950683045d8273516235bc8c2eab3c9452047ebe62260cae01. Category CSVs are optional for these two files; category-bearing outputs will differ without them. The wrapper retains the original cuts, contrasts, resample count and seed, then compares all selected statistical structures with the saved analysis while ignoring explicitly changed file-path/hash metadata.

Expected main population: 970 replies, 219 primary-reference positives, 751 negatives. J2 is a separate 300-response subset. Missing/invalid outputs remain penalised as in the original analysis. No new model call is needed.

## Other findings

See DATA_AVAILABILITY.md before interpreting the included historical and controlled-benchmark aggregates. Synthetic tests establish implementation behaviour; they do not replace withheld original observations. No private workshop/student data, source database or raw browser capture is included. See [NOTICE.md](NOTICE.md) for attribution.
