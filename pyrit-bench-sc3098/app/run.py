"""Development entry point.

    python -m app.run          (from the project root)

`python app/run.py` also works — the guard below puts the project root on
sys.path so the `app` package is importable either way.
"""
import logging
import os
import sys

if __package__ in (None, ""):  # started as a plain script, not with -m
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_served_app

class DockerAddressFilter(logging.Filter):
    """Show the Compose host URL while preserving warnings and request/error logs."""

    def filter(self, record):
        lines = record.getMessage().splitlines()
        if any(line.startswith(" * Running on ") for line in lines):
            lines = [line for line in lines if not line.startswith(" * Running on ")]
            lines.append(" * Open PyRIT-Bench in your browser: " + os.environ["APP_PUBLIC_URL"])
            record.msg = "\n".join(lines)
            record.args = ()
        return True


# Both judge versions use the configured default display.
app = create_served_app()

if __name__ == "__main__":
    # Debug mode enables the Werkzeug debugger (arbitrary code execution), so
    # keep it off by default. Set FLASK_DEBUG=1 in .env only for local dev.
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    # macOS runs AirPlay Receiver on port 5000, which answers requests instead
    # of Flask. Set PORT in .env (e.g. 5001) rather than fighting it.
    port = int(os.getenv("PORT", "5000"))
    host = os.getenv("HOST", "127.0.0.1")
    # Only Compose knows the port exposed on the student's laptop.
    if os.getenv("IN_DOCKER") == "1" and os.getenv("APP_PUBLIC_URL"):
        logging.getLogger("werkzeug").addFilter(DockerAddressFilter())
    app.run(host=host, port=port, debug=debug, use_reloader=False)
