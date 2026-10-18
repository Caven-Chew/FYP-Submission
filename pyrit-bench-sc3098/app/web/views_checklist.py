"""Old Checklist D links. The Checklist D attacks were removed from the app (user decision,
3 Oct 2026): the course's example prompts for each item are in the slides and the user guide."""
from flask import redirect, url_for

from .blueprint import bp


@bp.route('/checklist')
def checklist():
    return redirect(url_for('main.benchmarks_new'))
