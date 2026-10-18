# PyRIT Bench workshop

Workshop application, its BrewBot target and the final workshop slides.

**The application retains its original expiry: normal startup is refused after 7 September 2026.** Shared model access and facilitator unlocks are not supplied.

## Setup and offline tests

Use Python 3.12 with separate application and bot environments:

```sh
python3.12 -m venv pyrit/.venv
pyrit/.venv/bin/python -m pip install -r pyrit/requirements.txt
python3.12 -m venv brewbot/.venv
brewbot/.venv/bin/python -m pip install -r brewbot/requirements.txt
pyrit/.venv/bin/python -B test/run_offline_checks.py --bot-python brewbot/.venv/bin/python
```

The runner checks expiry, five app pages and twenty bot tests in disposable copies. It disables dotenv, removes model/unlock configuration and blocks network calls. A pre-expiry date is mocked only inside the test; the app's expiry guard is unchanged.

For the BrewBot scripted demonstration:

```sh
cd brewbot
.venv/bin/python bot.py --offline --host 127.0.0.1
```

Open http://127.0.0.1:5055. Dependencies are needed even in offline mode. Online use requires your own endpoint, model and key; use `.env.example` as a template. Scripted behaviour is a teaching demonstration, not a model-safety result.

The app entry point is `python -m app.run` from `pyrit/` on port 5050, subject to expiry. Browser targets additionally need `patchright install chromium`. Azure RedTeam has optional SDK requirements outside the base installation.

## Slides and downloads

The final deck is [slides/workshop.pptx](slides/workshop.pptx). Its download links and survey QR codes are retained.

[Workshop downloads on SharePoint](https://entuedu-my.sharepoint.com/:f:/r/personal/chew0453_e_ntu_edu_sg/Documents/LLM%20Security%20Workshop?d=w16c0399978ec4ff18e274d143488b929&csf=1&web=1&e=ohgoGn) contain the large macOS/Windows builds and original shared files. These remain on SharePoint. Access was checked from the owner's signed-in account; examiner access still needs checking.

See [NOTICE.md](NOTICE.md) for attribution. Credentials, compiled builds, local databases and participant responses are excluded.
