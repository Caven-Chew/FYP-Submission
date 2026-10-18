"""Development entry point.

    python -m app.run          (from the project root)

`python app/run.py` also works — the guard below puts the project root on
sys.path so the `app` package is importable either way.
"""
import os
import sys

if __package__ in (None, ""):  # started as a plain script, not with -m
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app

app = create_app()

if __name__ == "__main__":
    # Debug mode enables the Werkzeug debugger (arbitrary code execution), so
    # keep it off by default. Set FLASK_DEBUG=1 in .env only for local dev.
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    # Keep development consistent with the packaged workshop app. macOS often
    # reserves Flask's old port 5000 for AirPlay; BrewBot uses 5055.
    port = int(os.getenv("PORT", "5050"))
    app.run(host="127.0.0.1", port=port, debug=debug, use_reloader=False)
