# PyRIT Bench for SC3098

Classroom application with interactive and scripted assessment workflows, teaching attack examples and two educational judge views. The final research application is in [pyrit-bench](../pyrit-bench/README.md).

## Setup

Use Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/patchright install chromium
cp .env.example .env
.venv/bin/python -m app.run
```

Fill your own session key and model endpoint/key in `.env`. Keep `DEFAULT_JUDGE=v1` or `v2`. Open http://127.0.0.1:5002. Model-backed assessments require valid access; local storage is created under `instance/`.

Alternatively, configure `.env` and run `docker compose up --build`. Compose exposes loopback ports 5002 (app) and 6080 (noVNC). Container construction/execution has not been verified by the native Python tests.

## Configuration and troubleshooting

Use this repository's `.env.example` for supported options. If a model request fails, check the endpoint, model identifier, key, network access and context limits. Browser targets require the Chromium installation above. In Docker, open http://127.0.0.1:6080/vnc.html and select Connect to see the capture browser. API request bodies can use the supported `{history}` placeholder for conversation history. The [practice-chatbots](../practice-chatbots/README.md) folder provides BrewBot/RoboBot; any independently configured target may be used.

## Offline tests

```sh
.venv/bin/python -B test/run_offline_suites.py --output /tmp/sc3098-check-NEW
```

The 19 suites use disposable copies, mocked models and disabled dotenv. Even `test_live_pipeline.py` is a mocked offline test. One interactive conversation assertion has shown an intermittent timing failure; a focused rerun passed. No model-service availability is implied.

`app/data/gdp_checklist.json` contains fictional teaching fixtures. [NOTICE.md](NOTICE.md) records upstream attribution. Credentials, student applications and feedback are excluded.
