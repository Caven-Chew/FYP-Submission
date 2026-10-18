"""Application configuration.

Lives inside the app package, but everything it points at stays at the project
root: ROOT is this file's grandparent, so `.env` and `instance/app.db` sit next
to README.md rather than inside app/.
"""
from pathlib import Path
import os
import sys
from dotenv import load_dotenv

# When bundled by PyInstaller, __file__ resolves inside the extracted/bundled
# app folder, which is either a temp dir (--onefile) or an internal subfolder
# (--onedir) that isn't meant to hold user data. Point ROOT at the folder
# containing the .exe instead, so .env and instance/ live in a stable,
# writable location next to the packaged app.
if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).resolve().parent
else:
    ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
(ROOT / "instance").mkdir(exist_ok=True)


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me")
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{(ROOT / 'instance' / 'app.db').as_posix()}",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_PROMPTS_PER_ATTACK = int(os.getenv("MAX_PROMPTS_PER_ATTACK", "1"))
