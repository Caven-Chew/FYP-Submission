# PyRIT Bench

A Flask application for configuring chatbot targets, running adversarial assessments and reviewing execution and judging results. Uses Microsoft PyRIT 0.14.0.

## Setup

Use Python 3.12 from the repository root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
patchright install chromium
cp .env.example .env
python -m app.run
```

Configure your own model endpoint, key and session key in `.env`. Open http://127.0.0.1:5001. Local storage is created under `instance/`. Assess only targets you are authorised to test. Model calls require your own access.

## Scoring

Severity (1–5) is the default. `two_label` separately assesses objective harmfulness and reply fulfilment; `dual` retains both with severity primary. Select the method per run or with `JUDGE_METHOD`.

`contracts/two_label_v1/` contains required judge instructions, schemas and validators. The packaged integrity manifest uses the repository's portable paths. It differs from the study release manifest; contract instructions, schemas and validation rules are retained. Existing runs with an incompatible manifest hash must not be silently treated as compatible.

## Offline tests

On macOS/Linux, after creating `.venv` above:

```sh
python -B tools/judge_contract_release.py check
python -B test/test_judge_contract_release.py
python -B tools/run_all_isolated_suites.py --jobs 4 --with-resume-check
```

Twelve application suites use disposable storage and block external connections. The resume check uses a local mock server. These checks do not call hosted models. Windows users can activate `.venv\Scripts\activate` for app setup; the aggregate test runner assumes `.venv/bin/python`.

## Optional backends

In-process HuggingFace targets additionally need `pip install 'pyrit[huggingface]==0.14.0'` and suitable model access/hardware. Azure RedTeam is disabled in the catalogue; its optional SDKs are not included in the base requirements.

The [practice-chatbots](../practice-chatbots/README.md) folder supplies study targets; [pyrit-bench-evaluation](../pyrit-bench-evaluation/README.md) contains numerical analyses. See [NOTICE.md](NOTICE.md) for attribution. Credentials, databases and captured conversations are not included.
