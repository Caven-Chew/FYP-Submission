#!/usr/bin/env python3
"""Descriptive analysis of the SC3098 "Reading PyRIT-Bench results" Google Form export.

Implements rules written before any response was read:
reports/final_review/sc3098_survey/ANALYSIS_PLAN.md (§2, §4-§9, §11) and the
"Version 2 Google Form" rules in interpretation_activity/FACILITATOR_GUIDE.md.
Column wording comes from analysis_tool/column_map.json (form version 10).

Offline: no network or model call. The export is read only. Writes summary.md,
results.json and free_text_for_coding.md into --out-dir. Free-text answers go only to
free_text_for_coding.md; no output contains a timestamp from the export.

Run (see tools/README.md):
    .venv/bin/python -B tools/sc3098_form_analysis.py --input EXPORT.xlsx --invited N --out-dir DIR
Tests:
    .venv/bin/python -B -m unittest discover -s tools -p 'test_sc3098_form_analysis.py' -v
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import statistics
import sys
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from survey_io import sha, write_json  # noqa: E402  (September tool helpers, reused)

ROOT = Path(__file__).resolve().parents[1]
SURVEY = ROOT / 'methods' / 'course-feedback'
DEFAULT_MAP = SURVEY / 'column_map.json'
PLAN = SURVEY / 'ANALYSIS_PLAN.md'
GUIDE = SURVEY / 'FACILITATOR_GUIDE.md'
SEPTEMBER = ROOT / 'reports' / 'final_review' / 'workshop_survey_r6'
SGT = timezone(timedelta(hours=8))
OUTPUTS = ('summary.md', 'results.json', 'free_text_for_coding.md')
LEVELS = ('exact', 'code', 'prefix')
# Fallback matching compares the first min(K, length of either text) normalised characters, which must be
# at least MIN long: grid row labels K=25, MIN=15; question titles (item code removed) K=60, MIN=30.
ROW_PREFIX, ROW_MIN = 25, 15
STEM_PREFIX, STEM_MIN = 60, 30

# ANALYSIS_PLAN.md §2, column "The paper must not say", quoted verbatim (a test checks this).
MAY_NOT_SAY = [
    'That PyRIT-Bench is "usable" or "easy" in general, or easier than another tool',
    'That tasks were observed or verified',
    'That students "understood" or "learned"; that this is a knowledge or learning-gain test',
    'A SUS grade or percentile as if the sample were representative',
    'That changes between the two versions caused any difference',
    'That RQ1 is answered, unless a participant route is confirmed (report §6.3); even then it is self-administered, not an observed task study',
]
# Form-specific limits, paraphrased from FACILITATOR_GUIDE.md (version-2 rules) and ANALYSIS_PLAN.md §2, §5, §6, §9.
FORM_LIMITS = [
    'n counts submissions, not verified students: one response per person cannot be enforced without sign-in, so duplicates cannot be ruled out; no submission was merged or removed by timestamp.',
    'There is no consent question; every non-excluded submission is treated as agreement to take part (user decision, 9 October 2026).',
    'Google Forms does not record timing or assistance, so case results carry no "without answer-relevant help" condition.',
    'The case excerpts state parts of the scoring context, so intended answers show interpretation with that context provided, not recall of the app or learning. Case C is a labelled illustration.',
    'The developer ran the activity and awarded class-activity points, so answers may lean favourable or cautious; answers were given after the lab.',
    'SUS: small convenience sample; the score is not compared with published benchmarks and is not a percentage of tasks completed.',
    'P1 describes the group only; it is not cross-tabulated with any other item (identification risk in small groups).',
    'September comparisons are descriptive counts with each group\'s n: different students, course, setting and app version; no significance test and no causal claim.',
]


class FormAnalysisError(Exception):
    """A refusal with a message for the user (exit status 2)."""


# ---------------------------------------------------------------- text helpers

_FOLD = str.maketrans({'‘': "'", '’': "'", '‚': "'", '‛': "'", '′': "'",
                       '“': '"', '”': '"', '„': '"', '‟': '"', '″': '"',
                       '‐': '-', '‑': '-', '‒': '-', '–': '-', '—': '-', '−': '-'})
_CODE = re.compile(r'^\s*([A-Za-z]\d+)\s*[.):]\s*')


def norm(text):
    """Normalise a header or value for matching: NFKC, straight quotes, plain hyphens, single spaces, case-folded."""
    text = unicodedata.normalize('NFKC', str(text)).translate(_FOLD)
    return re.sub(r'\s+', ' ', text).strip().casefold()


def norm_option(text):
    return norm(text).rstrip(' .')


def split_header(header):
    """'Title [Row]' -> ('Title', 'Row'); other headers -> (header, None)."""
    match = re.match(r'^(.*?)\s*\[([^\[\]]*)\]\s*$', header, re.S)
    return (match.group(1), match.group(2)) if match else (header, None)


def code_of(stem):
    match = _CODE.match(stem)
    return match.group(1).upper() if match else None


def without_code(stem):
    return _CODE.sub('', stem, count=1)


def cell_text(value):
    """Cell value as text; None for blank. Integer-valued floats lose the '.0' (xlsx numbers)."""
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, datetime):
        value = value.isoformat()
    text = str(value)
    return text if text.strip() else None


# ---------------------------------------------------------------- reading

def read_table(path):
    """Return (headers, rows, info); rows are (spreadsheet_row_number, cells) with the header as row 1."""
    path = Path(path)
    if not path.is_file():
        raise FormAnalysisError(f'Input file not found: {path}')
    suffix = path.suffix.lower()
    if suffix == '.csv':
        with path.open(newline='', encoding='utf-8-sig') as stream:
            records = [list(r) for r in csv.reader(stream)]
        info = {'format': 'csv', 'sheet': None, 'sheets': None}
    elif suffix in ('.xlsx', '.xlsm'):
        import openpyxl
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            sheet = workbook.active
            records = [list(r) for r in sheet.iter_rows(values_only=True)]
            info = {'format': 'xlsx', 'sheet': sheet.title, 'sheets': list(workbook.sheetnames)}
        finally:
            workbook.close()
    else:
        raise FormAnalysisError(f'Unsupported input type "{path.suffix}": use the Google Sheets export as .xlsx or .csv.')
    if not records or all(cell_text(v) is None for v in records[0]):
        raise FormAnalysisError(f'{path.name} has no header row: it is empty. Nothing to analyse.')
    width = max(len(r) for r in records)
    headers = [cell_text(v) or '' for v in records[0]] + [''] * (width - len(records[0]))
    rows, empty = [], 0
    for number, record in enumerate(records[1:], start=2):
        cells = [cell_text(v) for v in record] + [None] * (width - len(record))
        if all(v is None for v in cells):
            empty += 1
        else:
            rows.append((number, cells))
    info['empty_rows_ignored'] = empty
    return headers, rows, info


# ---------------------------------------------------------------- column matching

def same_start(a, b, k, minimum):
    """True when the first min(k, len(a), len(b)) characters agree and that is at least `minimum`."""
    n = min(k, len(a), len(b))
    return n >= minimum and a[:n] == b[:n]


def _rows_match(item_rows, col_row):
    target = norm(col_row)
    return any(norm(row) == target or same_start(norm(row), target, ROW_PREFIX, ROW_MIN) for row in item_rows)


def _level_ok(level, item, col):
    grid = item.get('row') is not None
    if grid != (col['row'] is not None):
        return False
    variants = [(item['header'], item.get('row'))] + [split_header(a) for a in item.get('aliases', [])]
    if level == 'exact':
        return any(norm(stem) == norm(col['stem']) and (row is None) == (col['row'] is None)
                   and (row is None or norm(row) == norm(col['row'])) for stem, row in variants)
    rows = [row for _, row in variants if row is not None]
    if grid and not _rows_match(rows, col['row']):
        return False
    if level == 'code':
        return item.get('code') is not None and col['code'] == item['code'].upper()
    return any(same_start(norm(without_code(stem)), norm(without_code(col['stem'])), STEM_PREFIX, STEM_MIN)
               for stem, _ in variants)


def match_columns(headers, rows, column_map):
    """Match export columns to map keys. Returns (assigned {key: column index}, report)."""
    columns = []
    for index, header in enumerate(headers):
        stem, row = split_header(header)
        columns.append({'index': index, 'header': header, 'stem': stem, 'row': row, 'code': code_of(stem)})
    items = column_map['items']
    by_key = {item['key']: item for item in items}
    pending, used, assigned, levels, ambiguous = list(items), set(), {}, {}, {}
    for level in LEVELS:
        claims = {}
        for item in pending:
            candidates = [c for c in columns if c['index'] not in used and _level_ok(level, item, c)]
            if candidates:
                claims[item['key']] = candidates
        demand = Counter(c['index'] for candidates in claims.values() for c in candidates)
        for key, candidates in claims.items():
            if len(candidates) == 1 and demand[candidates[0]['index']] == 1:
                assigned[key], levels[key] = candidates[0]['index'], level
            else:
                ambiguous[key] = {'level': level, 'headers': [c['header'] for c in candidates]}
        used.update(assigned.values())
        pending = [item for item in pending if item['key'] not in claims]

    def non_blank(index):
        return sum(cells[index] is not None for _, cells in rows)

    matched = []
    for item in items:
        if item['key'] in assigned:
            header = headers[assigned[item['key']]]
            recorded = item['header'] + (f" [{item['row']}]" if item.get('row') else '')
            matched.append({'key': item['key'], 'level': levels[item['key']], 'header': header,
                            'differs_from_record': norm(header) != norm(recorded)})
    stale_codes = {rule['code'].upper() for rule in column_map.get('stale_columns', []) if rule.get('grid')}
    stale, unmatched = [], []
    for col in columns:
        if col['index'] in used:
            continue
        entry = {'header': col['header'] or '(no header)', 'column_number': col['index'] + 1, 'non_blank_n': non_blank(col['index'])}
        (stale if col['code'] in stale_codes and col['row'] is not None else unmatched).append(entry)
    report = {'keys_n': len(items), 'matched_n': len(matched), 'matched': matched,
              'ambiguous': [{'key': k, **v} for k, v in ambiguous.items()],
              'missing': [k for k in by_key if k not in assigned and k not in ambiguous],
              'stale': stale, 'unmatched': unmatched}
    return assigned, report


# ---------------------------------------------------------------- counting

def option_lookup(item):
    lookup = {norm_option(o): o for o in item['options']}
    for canonical, aliases in item.get('option_aliases', {}).items():
        for alias in aliases:
            lookup[norm_option(alias)] = canonical
    return lookup


def count_choice(item, values):
    """Counts per listed option; blanks and unexpected values are kept separate, never merged."""
    lookup = option_lookup(item)
    counts = {o: 0 for o in item['options']}
    blank, other = 0, Counter()
    for value in values:
        if value is None:
            blank += 1
        elif norm_option(value) in lookup:
            counts[lookup[norm_option(value)]] += 1
        else:
            other[value.strip()] += 1
    result = {'n': len(values), 'answered_n': len(values) - blank, 'blank_n': blank, 'counts': counts}
    if item.get('other_allowed'):
        result['other_free_text_n'] = sum(other.values())  # text itself goes only to free_text_for_coding.md
    else:
        # Unexpected values may be reworded options or, after an unrecorded form edit, free text:
        # count them here; analyse() moves the values themselves to free_text_for_coding.md only.
        result['other_value_n'] = sum(other.values())
        result['other_values'] = dict(sorted(other.items()))
    if item.get('rating_options'):
        result['rating_n'] = sum(counts[o] for o in item['rating_options'])
        result['non_rating_counts'] = {o: counts[o] for o in item['options'] if o not in item['rating_options']}
    if item.get('top_two'):
        result['top_two_n'] = sum(counts[o] for o in item['top_two'])
    if item.get('intended'):
        result['intended_n'] = counts[item['intended']]
        result['not_sure_n'] = counts[item['not_sure']]
        result['other_listed_counts'] = {o: counts[o] for o in item['options'] if o not in (item['intended'], item['not_sure'])}
        result['other_listed_n'] = sum(result['other_listed_counts'].values())
    return result


def sus_value(value, aliases):
    """1-5 from a grid answer such as '1 Strongly disagree' or '4'; None for blank; 'invalid' otherwise."""
    if value is None:
        return None
    match = re.match(r'\s*([1-5])(?!\d)', value)
    if match:
        return int(match.group(1))
    return aliases.get(norm_option(value), 'invalid')


def sus_score(answers):
    """SUS 0-100 from ten answers 1-5 in statement order: odd (answer - 1), even (5 - answer), sum x 2.5."""
    if len(answers) != 10 or any(a not in (1, 2, 3, 4, 5) for a in answers):
        raise ValueError('SUS needs ten answers, each 1-5')
    return 2.5 * sum((a - 1) if i % 2 == 0 else (5 - a) for i, a in enumerate(answers))


def analyse_sus(sus_items, assigned, rows, aliases):
    keys = [item['key'] for item in sorted(sus_items, key=lambda i: i['sus_item'])]
    missing = [k for k in keys if k not in assigned]
    if len(keys) != 10 or missing:
        return {'computed': False, 'reason': f'SUS not computed: O1 rows not found in the export: {", ".join(missing) or "map does not list ten rows"}'}
    alias_map = {norm_option(k): v for k, v in aliases.items()}
    per_row = {k: {'counts': {str(v): 0 for v in range(1, 6)}, 'blank_n': 0, 'other_value_n': 0, 'other_values': Counter()} for k in keys}
    scores, partly, none_answered, with_invalid, straight = [], 0, 0, 0, 0
    for _, cells in rows:
        parsed = []
        for key in keys:
            raw = cells[assigned[key]]
            value = sus_value(raw, alias_map)
            parsed.append(value)
            if value is None:
                per_row[key]['blank_n'] += 1
            elif value == 'invalid':
                per_row[key]['other_value_n'] += 1
                per_row[key]['other_values'][raw.strip()] += 1
            else:
                per_row[key]['counts'][str(value)] += 1
        with_invalid += 'invalid' in parsed
        if all(isinstance(v, int) for v in parsed):
            scores.append(sus_score(parsed))
            straight += len(set(parsed)) == 1 and parsed[0] != 3
        elif all(v is None for v in parsed):
            none_answered += 1
        else:
            partly += 1
    for entry in per_row.values():
        entry['other_values'] = dict(sorted(entry['other_values'].items()))
    result = {'computed': True, 'n': len(rows), 'complete_n': len(scores), 'incomplete_n': len(rows) - len(scores),
              'incomplete_partly_answered_n': partly, 'incomplete_no_rows_answered_n': none_answered,
              'rows_with_unreadable_value_n': with_invalid, 'straight_line_not_neutral_n': straight,
              'scores_sorted': sorted(scores), 'per_row': per_row}
    if scores:
        result.update(median=statistics.median(scores), minimum=min(scores), maximum=max(scores),
                      mean=round(statistics.fmean(scores), 2))
    return result


def analyse_cases(items, assigned, rows):
    choice = {item['case']: item for item in items if item.get('case') and item['type'] == 'choice'}
    reason = {item['case']: item for item in items if item.get('case') and item['type'] == 'text'}
    cases = sorted(choice)
    missing = [choice[c]['key'] for c in cases if choice[c]['key'] not in assigned]
    if missing:
        return {'computed': False, 'reason': f'All-case counts not computed: choice columns not found: {", ".join(missing)}'}
    lookups = {c: option_lookup(choice[c]) for c in cases}
    answered_all = all_intended = 0
    for _, cells in rows:
        values = [cells[assigned[choice[c]['key']]] for c in cases]
        if all(v is not None for v in values):
            answered_all += 1
            all_intended += all(lookups[c].get(norm_option(v)) == choice[c]['intended'] for c, v in zip(cases, values))
    result = {'computed': True, 'cases': cases, 'n': len(rows), 'answered_all_choices_n': answered_all,
              'all_intended_n': all_intended, 'all_intended_denominator': 'answered_all_choices_n',
              'not_all_choices_answered_n': len(rows) - answered_all}
    reason_keys = [reason[c]['key'] for c in cases if c in reason]
    if len(reason_keys) == len(cases) and all(k in assigned for k in reason_keys):
        result['answered_all_choices_and_reasons_n'] = sum(
            all(cells[assigned[choice[c]['key']]] is not None and cells[assigned[reason[c]['key']]] is not None for c in cases)
            for _, cells in rows)
    else:
        result['answered_all_choices_and_reasons_n'] = None
    return result


def september_counts(column_map):
    """Overall (scope 'all') counts of the matched September items from the frozen evidence; None if unavailable."""
    path = SEPTEMBER / 'summary.json'
    if not path.is_file():
        return None
    digest = sha(path)
    recorded = None
    sums = SEPTEMBER / 'SHA256SUMS.txt'
    if sums.is_file():
        for line in sums.read_text().splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[1].strip() == 'summary.json':
                recorded = parts[0]
    entries = {e['item']: e for e in json.loads(path.read_text()) if e['scope'] == 'all'}
    items = {}
    for item in column_map['items']:
        sept = item.get('september', {}).get('item')
        if sept in entries and item['type'] == 'choice':
            e = entries[sept]
            items[item['key']] = {'september_item': sept, 'population_n': e['population_n'], 'blank_n': e['blank_n'], 'counts': e['counts']}
    return {'file': str(path.relative_to(ROOT)), 'sha256': digest, 'matches_SHA256SUMS': digest == recorded, 'scope': 'all', 'items': items}


# ---------------------------------------------------------------- arguments

def parse_exclusions(specs):
    """'ROW:REASON' strings -> {row: reason}. ROW is the spreadsheet row number (header = row 1)."""
    out = {}
    for spec in specs or []:
        row, sep, reason = spec.partition(':')
        if not sep or not row.strip().isdigit() or not reason.strip():
            raise FormAnalysisError(f'--exclude "{spec}": use ROW:REASON, for example --exclude "2:dummy test entry by the organiser". ROW is the spreadsheet row number (the header is row 1).')
        number = int(row)
        if number < 2:
            raise FormAnalysisError(f'--exclude "{spec}": row 1 is the header; the first response is row 2.')
        if number in out:
            raise FormAnalysisError(f'--exclude names row {number} twice.')
        out[number] = reason.strip()
    return out


# ---------------------------------------------------------------- analysis

def analyse(input_path, column_map_path=DEFAULT_MAP, invited=None, exclusions=None, now=None):
    """Return (results, free_text). Raises FormAnalysisError for a refusal."""
    input_path, column_map_path = Path(input_path), Path(column_map_path)
    column_map = json.loads(column_map_path.read_text())
    headers, data_rows, info = read_table(input_path)
    if not data_rows:
        raise FormAnalysisError(f'{input_path.name} has 0 response rows (header only). Nothing to analyse: export again after the form has responses.')
    exclusions = exclusions or {}
    present = {number for number, _ in data_rows}
    absent = sorted(set(exclusions) - present)
    if absent:
        raise FormAnalysisError(f'--exclude names row(s) {absent}, which are not response rows in {input_path.name} (response rows: {min(present)}-{max(present)}).')
    if invited is not None and invited < 1:
        raise FormAnalysisError('--invited must be a positive number of Lab 7 participants.')
    rows = [(number, cells) for number, cells in data_rows if number not in exclusions]
    if not rows:
        raise FormAnalysisError('No response rows remain after the --exclude rows are removed. Nothing to analyse.')
    assigned, columns = match_columns(headers, rows, column_map)
    items = column_map['items']
    by_key = {item['key']: item for item in items}

    item_results, unexpected = {}, {}
    for item in items:
        if item['type'] in ('timestamp', 'sus'):
            continue
        if item['key'] not in assigned:
            item_results[item['key']] = {'found': False}
            continue
        values = [cells[assigned[item['key']]] for _, cells in rows]
        if item['type'] in ('choice', 'grid_choice'):
            counted = count_choice(item, values)
            if counted.get('other_values'):
                unexpected[item['key']] = counted['other_values']
            counted.pop('other_values', None)
            item_results[item['key']] = {'found': True, **counted}
        else:
            blank = sum(v is None for v in values)
            item_results[item['key']] = {'found': True, 'n': len(values), 'answered_n': len(values) - blank, 'blank_n': blank}

    sus = analyse_sus([i for i in items if i['type'] == 'sus'], assigned, rows, column_map.get('sus_value_aliases', {}))
    for key, entry in sus.get('per_row', {}).items():
        if entry['other_values']:
            unexpected[key] = entry['other_values']
        entry.pop('other_values')
    cases = analyse_cases(items, assigned, rows)

    conflicts = {'sus_straight_line_not_neutral_n': sus.get('straight_line_not_neutral_n')}
    d1, d2 = by_key.get('D1'), by_key.get('D2')
    if d1 and d2 and d1['key'] in assigned and d2['key'] in assigned:
        lookup, no_review = option_lookup(d1), set(d1.get('no_review_options', []))
        conflicts['d2_answered_after_d1_no_review_n'] = sum(
            lookup.get(norm_option(cells[assigned['D1']] or '')) in no_review and cells[assigned['D2']] is not None
            for _, cells in rows)
    else:
        conflicts['d2_answered_after_d1_no_review_n'] = None

    free_text = {}
    for item in items:
        if item['key'] not in assigned:
            continue
        index = assigned[item['key']]
        if item['type'] == 'text':
            free_text[item['key']] = [(n, cells[index]) for n, cells in rows if cells[index] is not None]
        elif item.get('other_allowed'):
            lookup = option_lookup(item)
            free_text[item['key'] + ' Other'] = [(n, cells[index]) for n, cells in rows
                                                 if cells[index] is not None and norm_option(cells[index]) not in lookup]

    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(microsecond=0)
    response_rate = None
    if invited is not None:
        response_rate = {'submissions_n': len(rows), 'invited_n': invited,
                         'note': 'Submissions divided by Lab 7 participants; uniqueness of submissions cannot be verified.'}
    results = {
        'tool': 'tools/sc3098_form_analysis.py',
        'form': column_map.get('form'),
        'run_time_utc': now.isoformat().replace('+00:00', 'Z'),
        'run_time_sgt': now.astimezone(SGT).isoformat(),
        'input': {'file': input_path.name, 'sha256': sha(input_path), **info, 'columns_n': len(headers),
                  'response_rows_n': len(data_rows)},
        'column_map': {'file': column_map_path.name, 'sha256': sha(column_map_path)},
        'rules': {name: {'file': str(path.relative_to(ROOT)), 'sha256': sha(path) if path.is_file() else None}
                  for name, path in (('analysis_plan', PLAN), ('facilitator_guide', GUIDE))},
        'exclusions': [{'sheet_row': n, 'reason': r} for n, r in sorted(exclusions.items())],
        'retained_n': len(rows),
        'n_is': 'number of retained submissions, not a verified number of students',
        'response_rate': response_rate,
        'columns': columns,
        'items': item_results,
        'sus': sus,
        'cases': cases,
        'conflicts': conflicts,
        'free_text_answers_n': {k: len(v) for k, v in free_text.items()},  # counts only; text is never stored here
        'september': september_counts(column_map),
    }
    results['unexpected_values_n'] = {k: sum(v.values()) for k, v in unexpected.items()}
    return results, {'answers': free_text, 'unexpected': unexpected}


# ---------------------------------------------------------------- rendering

def _table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' if i == 0 else '---:' for i in range(len(header))) + '|']
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return lines + ['']


def _md(text):
    return str(text).replace('|', '\\|')


def _choice_lines(item, res):
    if not res.get('found'):
        return [f"Column not found in the export: no counts (not zero). See section 2.", '']
    rows = [[_md(o), res['counts'][o]] for o in item['options']]
    if 'other_free_text_n' in res:
        rows.append(['Other (free text; in free_text_for_coding.md)', res['other_free_text_n']])
    else:
        rows.append(['Other value (not a listed option)', res['other_value_n']])
    rows.append(['Blank', res['blank_n']])
    lines = _table(['Answer', f"Count (n = {res['n']})"], rows)
    if res.get('other_value_n'):
        lines += [f"{res['other_value_n']} answer(s) matched no listed option. The values are listed in `free_text_for_coding.md` (they may be reworded options or free text). If one is a reworded option, add it to `option_aliases` in the column map and record the change as a deviation (plan §11).", '']
    if 'rating_n' in res:
        lines += [f"Rating answers (the five scale points): {res['rating_n']}; answered {res['answered_n']}; blank {res['blank_n']}. "
                  + '; '.join(f'"{_md(o)}" {c}' for o, c in res['non_rating_counts'].items()) + ' (reported separately, not low ratings).', '']
    return lines


def render_summary(results, column_map):
    by_key = {item['key']: item for item in column_map['items']}
    items, n = results['items'], results['retained_n']
    out = ['# SC3098 form — descriptive summary', '',
           f"Form: {results['form']}. Generated by `{results['tool']}` at {results['run_time_sgt']} (SGT; {results['run_time_utc']} UTC).",
           f"Input: `{results['input']['file']}` ({results['input']['format']}" + (f", sheet \"{results['input']['sheet']}\"" if results['input']['sheet'] else '') + f"), SHA-256 `{results['input']['sha256']}`.",
           f"Rules applied, written before any response: `ANALYSIS_PLAN.md` (version 2 with §11 revisions; SHA-256 `{(results['rules']['analysis_plan']['sha256'] or 'missing')[:16]}…`) and the version-2 Google Form rules in `FACILITATOR_GUIDE.md` (SHA-256 `{(results['rules']['facilitator_guide']['sha256'] or 'missing')[:16]}…`). Column map `{results['column_map']['file']}` (SHA-256 `{results['column_map']['sha256'][:16]}…`).",
           '',
           'Counts only. **n is the number of retained submissions, not a verified number of students.** A blank is no answer: it is never counted as wrong or negative and never replaced. Free-text answers are not in this file; they are in `free_text_for_coding.md` for manual coding, and the report uses paraphrases and respondent counts only (plan §8).', '']
    if results['input'].get('sheets') and len(results['input']['sheets']) > 1:
        out += [f"Note: the workbook has {len(results['input']['sheets'])} sheets; the active sheet \"{results['input']['sheet']}\" was read.", '']

    out += ['## 1. Submissions', '']
    out += _table(['Measure', 'Count'], [
        ['Response rows in the export', results['input']['response_rows_n']],
        ['Excluded by the user (`--exclude`)', len(results['exclusions'])],
        ['**Retained submissions (n)**', n]])
    out += [f"- Excluded spreadsheet row {e['sheet_row']}: {_md(e['reason'])}" for e in results['exclusions']] or ['No row was excluded.']
    out += ['',
            f"Fully empty spreadsheet rows ignored (not submissions): {results['input']['empty_rows_ignored']}. No other exclusion: no removal for speed, inconsistency, unexpected answers or similar timestamps (plan §4, §5).", '']
    rate = results['response_rate']
    if rate:
        pct = 100 * rate['submissions_n'] / rate['invited_n']
        out += [f"Response rate: {rate['submissions_n']} submissions / {rate['invited_n']} Lab 7 participants ({rate['submissions_n']}/{rate['invited_n']} = {pct:.1f}%). Uniqueness of submissions cannot be verified." +
                (' Submissions exceed the participant count: duplicates or non-Lab-7 respondents cannot be ruled out.' if rate['submissions_n'] > rate['invited_n'] else ''), '']
    else:
        out += ['Response rate: not computed (no `--invited` count given).', '']

    cols = results['columns']
    out += ['## 2. Columns matched / unmatched', '',
            f"Keys matched: {cols['matched_n']} of {cols['keys_n']}. Match levels: exact = recorded version-10 header; code = same item code and (for grid rows) row label; prefix = same start of the question title without its code (first {STEM_PREFIX} normalised characters, at least {STEM_MIN}). A header that differs from the record suggests a wording edit after version 10: check it and record it as a deviation (plan §11).", '']
    out += _table(['Key', 'Match', 'Header differs from record'], [[m['key'], m['level'], 'yes' if m['differs_from_record'] else 'no'] for m in cols['matched']])
    differing = [m for m in cols['matched'] if m['differs_from_record']]
    if differing:
        out += ['Headers that differ from the version-10 record:', ''] + [f"- {m['key']}: \"{_md(m['header'])}\"" for m in differing] + ['']
    out += [f"Keys not found (not analysed; no zero counts are reported for them): {', '.join(cols['missing']) or 'none'}.", '']
    if cols['ambiguous']:
        out += ['Ambiguous keys (several columns fit; not analysed — add the exact header to `aliases` in the column map):', '']
        out += [f"- {a['key']} ({a['level']} level): " + '; '.join(f'"{_md(h)}"' for h in a['headers']) for a in cols['ambiguous']] + ['']
    else:
        out += ['Ambiguous keys: none.', '']
    if cols['stale']:
        out += ['Stale P1 grid columns from versions 4–5 (never analysed; the guide says to ignore them when empty):', '']
        out += [f"- column {s['column_number']} \"{_md(s['header'])}\": non-blank {s['non_blank_n']}" + (' — **not empty: check before reporting**' if s['non_blank_n'] else '') for s in cols['stale']] + ['']
    if cols['unmatched']:
        out += ['Unmatched columns (contents not shown or analysed):', '']
        out += [f"- column {u['column_number']} \"{_md(u['header'])}\": non-blank {u['non_blank_n']}" for u in cols['unmatched']] + ['']
    else:
        out += ['Unmatched columns: none.', '']

    out += ['## 3. P1 — prior use of automated chatbot-testing tools', '']
    out += _choice_lines(by_key['P1'], items['P1'])
    out += ['P1 describes the group only. It is not used to explain differences between students and is not cross-tabulated with any item.', '']

    out += ['## 4. Cases A–C', '',
            f'Per case: intended answer, other listed answers, "Not sure", other value and blank (plan §5 rules for interpretation items; intended answers from the facilitator guide). n = {n}.', '']
    case_items = [i for i in column_map['items'] if i.get('case') and i['type'] == 'choice']
    rows = []
    for item in case_items:
        res = items[item['key']]
        if not res.get('found'):
            rows.append([f"{item['case']} ({item['key']})", 'not found', '', '', '', '', ''])
            continue
        rows.append([f"{item['case']} ({item['key']})", res['intended_n'], res['other_listed_n'], res['not_sure_n'], res['other_value_n'], res['blank_n'], res['answered_n']])
    out += _table(['Case', 'Intended', 'Other listed answer', 'Not sure', 'Other value', 'Blank', 'Answered (incl. Not sure)'], rows)
    for item in case_items:
        res = items[item['key']]
        if not res.get('found'):
            continue
        out += [f"**{item['key']}** (intended: \"{_md(item['intended'])}\")", '']
        out += _choice_lines(item, res)
    reason_rows = []
    for item in [i for i in column_map['items'] if i.get('case') and i['type'] == 'text']:
        res = items[item['key']]
        reason_rows.append([item['key'], res.get('answered_n', 'not found'), res.get('blank_n', '')])
    out += ['One-sentence reasons (counts only; the text is in `free_text_for_coding.md`):', '']
    out += _table(['Reason item', 'Answered', 'Blank'], reason_rows)
    c = results['cases']
    if c['computed']:
        out += _table(['All three cases', 'Count'], [
            ['Answered all three choices (Not sure and other values count as answered)', c['answered_all_choices_n']],
            [f"… of whom chose all three intended answers (denominator {c['answered_all_choices_n']})", c['all_intended_n']],
            ['Did not answer all three choices', c['not_all_choices_answered_n']],
            ['Answered all three choices and all three reasons', c['answered_all_choices_and_reasons_n'] if c['answered_all_choices_and_reasons_n'] is not None else 'not computed (a reason column is missing)']])
    else:
        out += [c['reason'], '']
    out += ['Reasons are scored by hand with the facilitator guide (supported / unsupported / blank / unresolved). This tool does not score them, so the joint outcome (intended choice and supported reason) is not computed here. No individual is graded or identified.', '']

    out += ['## 5. Lab work (D1, D2)', '', 'Self-reports, not observed behaviour.', '']
    out += _choice_lines(by_key['D1'], items['D1'])
    d2 = items['D2']
    out += [f"D2 (what influenced the action): answered {d2.get('answered_n', 'not found')}, blank {d2.get('blank_n', '')}. Text in `free_text_for_coding.md`; summarise descriptively with paraphrases.", '']

    out += ['## 6. Views on PyRIT-Bench (page 6)', '', '### O1 System Usability Scale', '']
    s = results['sus']
    if s['computed']:
        stats = [['Complete responses (all ten rows answered 1–5): scored', s['complete_n']],
                 ['Incomplete: some rows answered', s['incomplete_partly_answered_n']],
                 ['Incomplete: no rows answered', s['incomplete_no_rows_answered_n']],
                 ['Responses with an unreadable O1 value (counted as not answered for that row)', s['rows_with_unreadable_value_n']]]
        if s['complete_n']:
            stats += [['Median (0–100; not a percentage)', s['median']], ['Minimum–maximum', f"{s['minimum']}–{s['maximum']}"],
                      [f"Mean (n = {s['complete_n']})", s['mean']]]
        out += _table(['SUS', 'Value'], stats)
        out += ['Scoring: odd statements (answer − 1), even statements (5 − answer), sum × 2.5; only complete responses are scored; nothing is imputed. Not compared with published SUS benchmarks.', '']
        per = [[k, *[v['counts'][str(p)] for p in range(1, 6)], v['blank_n'], v['other_value_n']] for k, v in s['per_row'].items()]
        out += _table(['O1 row', '1', '2', '3', '4', '5', 'Blank', 'Other value'], per)
        if s['rows_with_unreadable_value_n']:
            out += ['Unreadable O1 values are listed in `free_text_for_coding.md`. Add a label to `sus_value_aliases` only if the record shows it is a scale point.', '']
    else:
        out += [s['reason'], '']
    out += ['### O2 usefulness compared with testing by hand', '']
    out += _choice_lines(by_key['O2'], items['O2'])
    out += ['### O3 clarity', '']
    o3 = [i for i in column_map['items'] if i.get('code') == 'O3']
    rows = []
    for item in o3:
        res = items[item['key']]
        if not res.get('found'):
            rows.append([_md(item['row']), 'not found', '', '', '', '', '', '', '', ''])
            continue
        rows.append([_md(item['row']), *[res['counts'][o] for o in item['options']], res['other_value_n'], res['blank_n'], f"{res['top_two_n']} of {res['rating_n']}"])
    out += _table(['Row', 'Not at all', 'Slightly', 'Moderately', 'Very', 'Completely', 'Did not see this', 'Other value', 'Blank', 'Very or completely clear (of rating answers)'], rows)
    out += ['"Did not see this" is reported separately and is not in the clarity denominator.', '']
    if any(items[i['key']].get('other_value_n') for i in o3):
        out += ['O3 values that matched no column label are listed in `free_text_for_coding.md`.', '']
    out += ['### O4 likelihood of future use', '']
    out += _choice_lines(by_key['O4'], items['O4'])
    out += ['### O5 and O6 (free text)', '']
    out += _table(['Item', 'Answered', 'Blank'], [[k, items[k].get('answered_n', 'not found'), items[k].get('blank_n', '')] for k in ('O5', 'O6')])
    out += ['Coded by theme by hand (plan §8 codebook): respondents per theme, paraphrases only, no quotations or times.', '']

    out += ['## 7. Side by side with September (descriptive only)', '']
    sept = results['september']
    if not sept:
        out += ['September summary not available; no side-by-side table.', '']
    else:
        out += [f"September counts: frozen `{sept['file']}` (scope \"all\", SHA-256 `{sept['sha256'][:16]}…`, {'matches' if sept['matches_SHA256SUMS'] else '**does not match**'} its SHA256SUMS entry).", '']
        for key, sitem in sept['items'].items():
            item, res = by_key[key], items[key]
            out += [f"**{key} ↔ September {sitem['september_item']}.** {_md(item['september']['comparability'])}", '']
            if not res.get('found'):
                out += ['This form: column not found.', '']
                continue
            options = list(item['options']) + [o for o in sitem['counts'] if o not in item['options']]
            rows = [[_md(o), res['counts'].get(o, '—'), sitem['counts'].get(o, 0)] for o in options]
            rows.append(['Blank', res['blank_n'], sitem['blank_n']])
            out += _table(['Answer', f'This form (n = {n})', f"September (n = {sitem['population_n']})"], rows)
        out += ['Differences to state: different students and course; SC3098 app copy (own prompts, Tier 2, two judge versions); self-setup with Docker; different targets; this form after the lab; September operators were a subset (13 of 20). Counts with each n; no significance test; no causal claim (plan §9). O5/O6 correspond to September P2/P3 and are compared only as coded themes.', '']

    conf = results['conflicts']
    out += ['## 8. Conflicting answers (kept unchanged)', '',
            f"- D2 answered although D1 says \"I did not review a result or error message.\" or \"I cannot remember.\" (D2 asks these respondents to skip): {conf['d2_answered_after_d1_no_review_n'] if conf['d2_answered_after_d1_no_review_n'] is not None else 'not computed (D1 or D2 missing)'}.",
            f"- Complete SUS responses with all ten rows identical and not 3 (agreeing equally with positive and negative statements): {conf['sus_straight_line_not_neutral_n'] if conf['sus_straight_line_not_neutral_n'] is not None else 'not computed'}; scored as given.",
            '- Case choice/reason inconsistencies are identified only during manual reason scoring.',
            'No answer was changed, merged or excluded because of a conflict (plan §4).', '']

    out += ['## 9. Limitations and claims the report must not make', '',
            'Quoted verbatim from `ANALYSIS_PLAN.md` §2, column "The paper must not say":', '']
    out += [f'> - {line}' for line in MAY_NOT_SAY] + ['', 'Further limits for this form:', ''] + [f'- {line}' for line in FORM_LIMITS] + ['']
    return '\n'.join(out)


def render_free_text(results, free_text, column_map):
    by_key = {item['key']: item for item in column_map['items']}
    out = ['# Free-text answers for manual coding (protected working file)', '',
           f"Input SHA-256 `{results['input']['sha256']}`; generated {results['run_time_sgt']} (SGT). Do not paste into the report or share. Report paraphrases and respondent counts only, with no quotations, times, names or details of another group's chatbot (plan §8).",
           '"Row" is the spreadsheet row of the export (header = row 1; in a .csv, the record number), used only to link a respondent\'s answers for respondent-level counts. Excluded rows are not included. Text is reproduced as submitted (outer spaces trimmed; continuation lines indented).', '',
           'Case reasons (A2, B2, C2): score each as supported / unsupported / blank / unresolved with the facilitator guide scoring table. The chosen letter is deliberately not shown here; combine by row afterwards for the joint outcome.', '']
    for key, answers in free_text['answers'].items():
        base = key.replace(' Other', '')
        title = by_key[base]['header'] + (' — "Other" text' if key.endswith(' Other') else '')
        out += [f'## {key}', '', f'Question: {title}', '', f'Answers: {len(answers)}', '']
        out += [f"- row {number}: " + text.strip().replace('\n', '\n  ') for number, text in answers] + ['']
    out += ['## Unexpected values in choice items', '',
            'Answers that matched no listed option (counts per value). Check whether each is a reworded option (then add it to the column map and record a deviation) or free text (then treat it as free text).', '']
    if free_text['unexpected']:
        for key, values in free_text['unexpected'].items():
            out += [f'- {key}: ' + '; '.join(f'"{v}" ({c})' for v, c in values.items())]
        out.append('')
    else:
        out += ['None.', '']
    return '\n'.join(out)


def write_outputs(results, free_text, column_map, out_dir, overwrite=False):
    out_dir = Path(out_dir)
    existing = [name for name in OUTPUTS if (out_dir / name).exists()]
    if existing and not overwrite:
        raise FormAnalysisError(f'{out_dir} already has {", ".join(existing)}. Choose a new --out-dir (for example analysis_<date>) or pass --overwrite deliberately.')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'summary.md').write_text(render_summary(results, column_map))
    write_json(out_dir / 'results.json', results)
    (out_dir / 'free_text_for_coding.md').write_text(render_free_text(results, free_text, column_map))


def run(input_path, out_dir, column_map_path=DEFAULT_MAP, invited=None, exclusions=None, overwrite=False, now=None):
    results, free_text = analyse(input_path, column_map_path, invited, exclusions, now)
    write_outputs(results, free_text, json.loads(Path(column_map_path).read_text()), out_dir, overwrite)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, type=Path, help='Google Sheets export of the responses (.xlsx or .csv)')
    parser.add_argument('--column-map', type=Path, default=DEFAULT_MAP, help=f'column map JSON (default: {DEFAULT_MAP.relative_to(ROOT)})')
    parser.add_argument('--invited', type=int, help='number of Lab 7 participants (response-rate denominator); omit to print no response rate')
    parser.add_argument('--exclude', action='append', metavar='ROW:REASON', help='exclude a test entry by spreadsheet row number (header = row 1) with the reason; repeat for several rows')
    parser.add_argument('--out-dir', required=True, type=Path, help='folder for summary.md, results.json and free_text_for_coding.md')
    parser.add_argument('--overwrite', action='store_true', help='replace existing output files in --out-dir')
    args = parser.parse_args(argv)
    try:
        results = run(args.input, args.out_dir, args.column_map, args.invited, parse_exclusions(args.exclude), args.overwrite)
    except FormAnalysisError as error:
        print(f'Refused: {error}', file=sys.stderr)
        return 2
    cols = results['columns']
    print(f"Retained submissions: {results['retained_n']} (excluded {len(results['exclusions'])}); keys matched {cols['matched_n']}/{cols['keys_n']}; "
          f"missing {len(cols['missing'])}, ambiguous {len(cols['ambiguous'])}, unmatched columns {len(cols['unmatched'])}, stale P1 columns {len(cols['stale'])}.")
    print(f'Outputs written to {args.out_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
