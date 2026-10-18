# Practice chatbots

BrewBot and RoboBot are the two practice targets used by the controlled FYP benchmark. Both share bot.py, one Flask process, the interface and dependencies, so they share this folder. 

## Run locally

Use Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python bot.py
```

Open http://127.0.0.1:5056. BREWBOT_OFFLINE=true enables a scripted BrewBot demonstration; RoboBot Strict is deterministic. RoboBot's other modes need a configured model. To exercise those modes, set BREWBOT_OFFLINE=false and provide an OpenAI-compatible base URL, model and your own key (BREWBOT_* or LLM_ENDPOINT/LLM_API_KEY). No shared classroom credentials are included. The original loader supports a parent .env fallback; create a local .env to keep the intended configuration beside the app.

Alternatively, after creating the local .env, run `docker compose up --build`. The packaged Compose file reads this repository's .env rather than a parent classroom file. Docker execution is optional and is not claimed as tested by a native Python regression run.

BrewBot exposes /api/chat; RoboBot exposes /api/robo. Fictional records, prices, planted disclosure markers and instruction-bearing notes are deliberate fixtures. Do not interpret the scripted demonstration as an LLM safety evaluation. No real student/customer records are included.

## Offline tests

```sh
python -B -m unittest discover -s tests -v
```

The twenty tests use mocks and Flask's test client; one UI regression uses Node when available. No hosted model call is needed.

See [NOTICE.md](NOTICE.md) for attribution. Study analyses are in [pyrit-bench-evaluation](../pyrit-bench-evaluation/README.md).
