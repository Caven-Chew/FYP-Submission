"""The web layer — one Flask blueprint, split by page.

  blueprint.py         the shared `bp` every view registers on
  security.py          CSRF token issue + check
  views_dashboard.py   /
  views_targets.py     /targets...
  views_benchmarks.py  /benchmarks...

Shared by those views:

  charts.py       inline SVG charts (no JS, no chart library)
  rollups.py      Benchmark rows -> the summary numbers a page displays
  report_pdf.py   the supervisor-facing PDF report
  error_hints.py  raw provider error -> plain-English hint + fix
  text.py         small text helpers
"""
from __future__ import annotations

from .blueprint import bp

# Importing the view modules is what registers their routes on `bp`.
from . import security          # noqa: F401,E402
from . import views_dashboard   # noqa: F401,E402
from . import views_targets     # noqa: F401,E402
from . import views_benchmarks  # noqa: F401,E402

__all__ = ["bp"]
