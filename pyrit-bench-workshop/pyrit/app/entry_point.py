"""PyInstaller entry point.

Builds run against this file instead of app/run.py so packaging concerns (no
reloader, no debug mode, auto-opening a browser tab) stay separate from the
normal dev entry point. Referenced by build.spec at the project root.
"""
import os
import sys
import threading
import time
import webbrowser

# Allow this entry point to be run directly from the project root as well as
# being launched by PyInstaller.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app   # importing this loads .env via app.config

HOST = "127.0.0.1"
# Use the same workshop-safe default on every platform. macOS commonly reserves
# Flask's traditional port 5000 for AirPlay Receiver; BrewBot uses 5055.
PORT = int(os.getenv("PORT", "5050"))


def _open_browser() -> None:
    time.sleep(1.2)
    webbrowser.open(f"http://{HOST}:{PORT}/")


def main() -> None:
    """Start the packaged local server."""
    app = create_app()
    threading.Thread(target=_open_browser, daemon=True).start()
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
