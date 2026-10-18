"""The single Flask blueprint every view module registers on."""
from __future__ import annotations

from flask import Blueprint


bp = Blueprint("main", __name__)
