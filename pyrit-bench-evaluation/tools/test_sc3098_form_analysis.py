"""Offline checks for sc3098_form_analysis.py.

Every fixture is synthetic and built inside the tests from the recorded column map
(reports/final_review/sc3098_survey/analysis_tool/column_map.json); no response data are
read. Outputs go to temporary folders. No network or model call.

Run with:

    .venv/bin/python -B -m unittest discover -s tools -p 'test_sc3098_form_analysis.py' -v
"""
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sc3098_form_analysis as fa  # noqa: E402

MAP = json.loads(fa.DEFAULT_MAP.read_text())
ITEMS = {item['key']: item for item in MAP['items']}
SUS_KEYS = [f'O1_{i:02d}' for i in range(1, 11)]
FREE_MARKERS = ['FT-A2-ROW3', 'FT-B2-ROW3', 'FT-C2-ROW3', 'FT-D2-ROW5', 'FT-O5-ROW3', 'FT-O6-ROW4', 'FT-OTHER-D1-ROW6', 'FT-A2-ROW6']
TIMESTAMP_TEXT = '10/10/2026 14:03:27'
UNEXPECTED_A1 = 'E. An option the form never had'
STALE_HEADER = 'P1. Before this SC3098 course, how much hands-on experience did you have with each activity? [Building an LLM-powered chatbot]'


def header(key):
    item = ITEMS[key]
    return item['header'] + (f" [{item['row']}]" if item.get('row') else '')


def opt(key, letter):
    return next(o for o in ITEMS[key]['options'] if o.startswith(letter + '.'))


def sus(values):
    return {key: (f'{v} Strongly disagree' if v == 1 else f'{v} Strongly agree' if v == 5 else str(v)) if v is not None else None
            for key, v in zip(SUS_KEYS, values)}


def base_row():
    """A full answer set: all three intended choices, SUS 4/2 pattern (75)."""
    row = {'Timestamp': TIMESTAMP_TEXT, 'P1': 'No',
           'A1_choice': ITEMS['A1_choice']['intended'], 'A2_reason': 'FT-A2-ROW3',
           'B1_choice': ITEMS['B1_choice']['intended'], 'B2_reason': 'FT-B2-ROW3',
           'C1_choice': ITEMS['C1_choice']['intended'], 'C2_reason': 'FT-C2-ROW3',
           'D1': ITEMS['D1']['options'][0], 'D2': None,
           'O2': '4 Very useful', 'O3_1': 'Very clear', 'O3_2': 'Completely clear', 'O3_3': 'Moderately clear', 'O3_4': 'Did not see this',
           'O4': 'Somewhat likely', 'O5': 'FT-O5-ROW3', 'O6': None}
    row.update(sus([4, 2] * 5))
    return row


def fixture_rows():
    """Spreadsheet rows 2-7. Row 2 is a test entry to exclude; rows 3 and 4 look like duplicates."""
    test = base_row()
    test.update(A2_reason='TEST-ROW-TEXT')
    row3 = base_row()
    row4 = dict(row3)                                   # identical answers and timestamp: must NOT be merged
    row4.update(O6='FT-O6-ROW4')
    row5 = {'Timestamp': '10/10/2026 15:00:00', 'P1': 'Yes, once',
            'A1_choice': ITEMS['A1_choice']['not_sure'], 'B1_choice': ITEMS['B1_choice']['intended'], 'C1_choice': None,
            'D1': 'I did not review a result or error message.', 'D2': 'FT-D2-ROW5', 'O2': 'Cannot judge'}
    row5.update(sus([3, 3, 3, 3, 3, 3, 3, 3, 3, None]))  # incomplete SUS
    row6 = {'Timestamp': '10/10/2026 16:00:00', 'P1': None,
            'A1_choice': UNEXPECTED_A1, 'A2_reason': 'FT-A2-ROW6',
            'B1_choice': opt('B1_choice', 'B'), 'C1_choice': opt('C1_choice', 'A'),
            'D1': 'FT-OTHER-D1-ROW6', 'O2': '5 Extremely useful', 'O4': 'Cannot judge'}
    row6.update(sus([5] * 10))                            # straight line, score 50
    row7 = {'Timestamp': '10/10/2026 17:00:00'}           # submitted with every question skipped
    return [test, row3, row4, row5, row6, row7]


def headers(extra=()):
    return [header(item['key']) for item in MAP['items']] + list(extra)


def write_csv(path, head, rows, keys):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(head)
        for row in rows:
            writer.writerow([row.get(k) or '' for k in keys])


def write_xlsx(path, head, rows, keys):
    import openpyxl
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = 'Form Responses 1'
    sheet.append(head)
    for row in rows:
        values = [row.get(k) for k in keys]
        if keys[0] == 'Timestamp' and values[0]:
            values[0] = datetime.strptime(values[0], '%d/%m/%Y %H:%M:%S')
        # Grid answers that are plain numbers arrive as numbers in xlsx
        values = [int(v) if isinstance(v, str) and v.isdigit() else v for v in values]
        sheet.append(values)
    workbook.save(path)


class SusFormula(unittest.TestCase):
    def test_known_scores(self):
        self.assertEqual(fa.sus_score([3] * 10), 50.0)
        self.assertEqual(fa.sus_score([5, 1] * 5), 100.0)
        self.assertEqual(fa.sus_score([1, 5] * 5), 0.0)
        self.assertEqual(fa.sus_score([4, 2] * 5), 75.0)
        # odd: (5-1)+(4-1)+(3-1)+(2-1)+(1-1)=10; even: (5-1)+(5-2)+(5-3)+(5-4)+(5-5)=10; 20 x 2.5
        self.assertEqual(fa.sus_score([5, 1, 4, 2, 3, 3, 2, 4, 1, 5]), 50.0)
        self.assertEqual(fa.sus_score([5] * 10), 50.0)

    def test_rejects_incomplete_or_invalid(self):
        with self.assertRaises(ValueError):
            fa.sus_score([3] * 9)
        with self.assertRaises(ValueError):
            fa.sus_score([3] * 9 + [6])

    def test_value_parsing(self):
        aliases = {fa.norm_option(k): v for k, v in MAP['sus_value_aliases'].items()}
        self.assertEqual(fa.sus_value('1 Strongly disagree', aliases), 1)
        self.assertEqual(fa.sus_value('5 - Strongly agree', aliases), 5)
        self.assertEqual(fa.sus_value('3', aliases), 3)
        self.assertEqual(fa.sus_value('Strongly agree', aliases), 5)
        self.assertIsNone(fa.sus_value(None, aliases))
        self.assertEqual(fa.sus_value('10', aliases), 'invalid')
        self.assertEqual(fa.sus_value('maybe', aliases), 'invalid')


class ColumnMapRecord(unittest.TestCase):
    def test_map_is_consistent_with_the_guide(self):
        self.assertEqual(len(ITEMS), len(MAP['items']), 'duplicate keys')
        intended = {ITEMS[k]['case']: ITEMS[k]['intended'][0] for k in ('A1_choice', 'B1_choice', 'C1_choice')}
        self.assertEqual(intended, {'A': 'C', 'B': 'A', 'C': 'B'})   # FACILITATOR_GUIDE.md: A -> C, B -> A, C -> B
        for key in ('A1_choice', 'B1_choice', 'C1_choice'):
            self.assertIn(ITEMS[key]['intended'], ITEMS[key]['options'])
            self.assertIn(ITEMS[key]['not_sure'], ITEMS[key]['options'])
        self.assertEqual(sorted(ITEMS[k]['sus_item'] for k in SUS_KEYS), list(range(1, 11)))

    def test_limitations_quote_the_plan_verbatim(self):
        plan = fa.PLAN.read_text()
        for line in fa.MAY_NOT_SAY:
            self.assertIn(f'| {line} |', plan)


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.keys = [item['key'] for item in MAP['items']]

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, fmt='csv', head=None, rows=None, keys=None, invited=30, exclude=None):
        path = self.dir / f'export.{fmt}'
        writer = write_csv if fmt == 'csv' else write_xlsx
        writer(path, head or headers(), fixture_rows() if rows is None else rows, keys or self.keys)
        out = self.dir / f'out_{fmt}'
        exclusions = fa.parse_exclusions(['2:dummy test entry by the organiser'] if exclude is None else exclude)
        results = fa.run(path, out, invited=invited, exclusions=exclusions, overwrite=True)
        texts = {name: (out / name).read_text() for name in fa.OUTPUTS}
        return results, texts

    def check_counts(self, results):
        self.assertEqual(results['input']['response_rows_n'], 6)
        self.assertEqual(results['retained_n'], 5)          # rows 3 and 4 both kept (no deduplication)
        self.assertEqual(results['exclusions'], [{'sheet_row': 2, 'reason': 'dummy test entry by the organiser'}])
        a1, b1, c1 = (results['items'][k] for k in ('A1_choice', 'B1_choice', 'C1_choice'))
        self.assertEqual((a1['intended_n'], a1['other_listed_n'], a1['not_sure_n'], a1['other_value_n'], a1['blank_n'], a1['answered_n']), (2, 0, 1, 1, 1, 4))
        self.assertEqual((b1['intended_n'], b1['other_listed_n'], b1['not_sure_n'], b1['other_value_n'], b1['blank_n']), (3, 1, 0, 0, 1))
        self.assertEqual((c1['intended_n'], c1['other_listed_n'], c1['not_sure_n'], c1['other_value_n'], c1['blank_n']), (2, 1, 0, 0, 2))
        cases = results['cases']
        self.assertEqual(cases['answered_all_choices_n'], 3)   # rows 3, 4, 6 (an unexpected value counts as answered)
        self.assertEqual(cases['all_intended_n'], 2)           # rows 3, 4
        self.assertEqual(cases['not_all_choices_answered_n'], 2)
        self.assertEqual(cases['answered_all_choices_and_reasons_n'], 2)
        self.assertEqual(results['items']['P1']['counts'], {'No': 2, 'Yes, once': 1, 'Yes, more than once': 0, 'Not sure': 0})
        self.assertEqual(results['items']['P1']['blank_n'], 2)
        s = results['sus']
        self.assertEqual((s['complete_n'], s['incomplete_n'], s['incomplete_partly_answered_n'], s['incomplete_no_rows_answered_n']), (3, 2, 1, 1))
        self.assertEqual(s['scores_sorted'], [50.0, 75.0, 75.0])
        self.assertEqual((s['median'], s['minimum'], s['maximum'], s['mean']), (75.0, 50.0, 75.0, 66.67))
        self.assertEqual(results['conflicts'], {'sus_straight_line_not_neutral_n': 1, 'd2_answered_after_d1_no_review_n': 1})
        d1 = results['items']['D1']
        self.assertEqual((d1['other_free_text_n'], d1['blank_n'], d1['counts'][ITEMS['D1']['options'][0]]), (1, 1, 2))
        o2 = results['items']['O2']
        self.assertEqual((o2['rating_n'], o2['non_rating_counts']['Cannot judge'], o2['blank_n']), (3, 1, 1))
        o3 = results['items']['O3_4']
        self.assertEqual((o3['counts']['Did not see this'], o3['rating_n'], o3['top_two_n']), (2, 0, 0))
        self.assertEqual(results['items']['O3_1']['top_two_n'], 2)
        self.assertEqual(results['response_rate'], {'submissions_n': 5, 'invited_n': 30, 'note': results['response_rate']['note']})
        self.assertEqual(results['columns']['matched_n'], 28)

    def check_privacy(self, texts):
        for marker in FREE_MARKERS:
            self.assertNotIn(marker, texts['summary.md'], marker)
            self.assertNotIn(marker, texts['results.json'], marker)
            self.assertIn(marker, texts['free_text_for_coding.md'], marker)
        for name, text in texts.items():
            self.assertNotIn('TEST-ROW-TEXT', text, name)      # excluded row nowhere
            self.assertNotIn('14:03:27', text, name)            # no export timestamp anywhere
            self.assertNotIn('2026-10-10', text, name)
        self.assertNotIn(UNEXPECTED_A1, texts['summary.md'])
        self.assertNotIn(UNEXPECTED_A1, texts['results.json'])
        self.assertIn(UNEXPECTED_A1, texts['free_text_for_coding.md'])
        for line in fa.MAY_NOT_SAY:
            self.assertIn(line, texts['summary.md'])

    def test_csv_counts_and_privacy(self):
        results, texts = self.run_tool('csv')
        self.check_counts(results)
        self.check_privacy(texts)
        self.assertIn('5 submissions / 30 Lab 7 participants', texts['summary.md'])
        self.assertEqual(len(results['input']['sha256']), 64)
        self.assertTrue(results['run_time_sgt'].endswith('+08:00'))
        self.assertTrue(results['run_time_utc'].endswith('Z'))

    def test_xlsx_gives_the_same_counts(self):
        results, texts = self.run_tool('xlsx')
        self.assertEqual(results['input']['sheet'], 'Form Responses 1')
        self.check_counts(results)
        self.check_privacy(texts)

    def test_no_response_rate_without_invited(self):
        results, texts = self.run_tool('csv', invited=None)
        self.assertIsNone(results['response_rate'])
        self.assertIn('Response rate: not computed', texts['summary.md'])

    def test_edited_headers_still_match(self):
        head = headers()
        keys = list(self.keys)
        head[keys.index('A1_choice')] = 'A1. A reviewer says the reply kept the notes secret. Which answer is best supported?'  # stem edited, code kept
        head[keys.index('C2_reason')] = 'Explain your choice in one sentence, using what each judge checks.'                     # code removed
        head[keys.index('O3_2')] = 'O3. How clear was each of the following in PyRIT-Bench? [What the scores meant (1–5)]'      # row label extended
        head[keys.index('B1_choice')] = head[keys.index('B1_choice')].replace('“', '"').replace('”', '"').upper()                # quotes and case
        results, _ = self.run_tool('csv', head=head, keys=keys)
        matched = {m['key']: m for m in results['columns']['matched']}
        self.assertEqual(results['columns']['matched_n'], 28)
        self.assertEqual((matched['A1_choice']['level'], matched['A1_choice']['differs_from_record']), ('code', True))
        self.assertEqual(matched['C2_reason']['level'], 'prefix')
        self.assertEqual(matched['O3_2']['level'], 'code')
        self.assertEqual((matched['B1_choice']['level'], matched['B1_choice']['differs_from_record']), ('exact', False))
        self.check_counts(results)

    def test_unmatched_stale_missing_and_ambiguous_columns_are_reported(self):
        keys = [k for k in self.keys if k != 'O4'] + ['stale', 'extra']
        head = [header(k) for k in self.keys if k != 'O4'] + [STALE_HEADER, 'Score']
        rows = fixture_rows()
        for row in rows:
            row['extra'] = 'UNMATCHED-CONTENT'
        results, texts = self.run_tool('csv', head=head, rows=rows, keys=keys)
        cols = results['columns']
        self.assertEqual(cols['missing'], ['O4'])
        self.assertEqual(results['items']['O4'], {'found': False})
        self.assertEqual(cols['stale'], [{'header': STALE_HEADER, 'column_number': len(head) - 1, 'non_blank_n': 0}])
        self.assertEqual(cols['unmatched'], [{'header': 'Score', 'column_number': len(head), 'non_blank_n': 5}])
        for text in texts.values():
            self.assertNotIn('UNMATCHED-CONTENT', text)
        self.assertIn('Keys not found (not analysed; no zero counts are reported for them): O4', texts['summary.md'])
        self.assertIn('"Score": non-blank 5', texts['summary.md'])
        # Two columns that fit one key equally are reported as ambiguous, not guessed.
        head2 = headers()
        head2[self.keys.index('B2_reason')] = 'Explain your choice in one sentence, using what the saved result shows.'
        head2[self.keys.index('C2_reason')] = 'Explain your choice in one sentence, using what the saved result shows.'
        results2, _ = self.run_tool('csv', head=head2)
        self.assertEqual([a['key'] for a in results2['columns']['ambiguous']], ['B2_reason'])
        self.assertEqual(results2['columns']['missing'], ['C2_reason'])   # neither column fits C2's wording
        self.assertEqual(results2['items']['B2_reason'], {'found': False})
        self.assertIsNone(results2['cases']['answered_all_choices_and_reasons_n'])
        # Two truncated headers that fit both keys: both keys ambiguous.
        head3 = headers()
        head3[self.keys.index('B2_reason')] = 'Explain your choice in one sentence, using what'
        head3[self.keys.index('C2_reason')] = 'Explain your choice in one sentence, using what'
        results3, texts3 = self.run_tool('csv', head=head3)
        self.assertEqual({a['key'] for a in results3['columns']['ambiguous']}, {'B2_reason', 'C2_reason'})
        self.assertIn('Ambiguous keys (several columns fit; not analysed', texts3['summary.md'])

    def test_zero_rows_are_refused(self):
        for fmt in ('csv', 'xlsx'):
            with self.assertRaisesRegex(fa.FormAnalysisError, '0 response rows'):
                self.run_tool(fmt, rows=[])
        path = self.dir / 'header_only.csv'
        write_csv(path, headers(), [], self.keys)
        self.assertEqual(fa.main(['--input', str(path), '--out-dir', str(self.dir / 'never')]), 2)
        self.assertFalse((self.dir / 'never').exists())
        empty = self.dir / 'empty.csv'
        empty.write_text('')
        with self.assertRaisesRegex(fa.FormAnalysisError, 'no header row'):
            fa.analyse(empty)
        with self.assertRaisesRegex(fa.FormAnalysisError, 'No response rows remain'):
            self.run_tool('csv', rows=fixture_rows()[:1], exclude=['2:dummy test entry'])

    def test_exclusion_arguments_are_checked(self):
        for bad in (['2'], ['x:reason'], ['2:'], ['1:header'], ['3:a', '3:b']):
            with self.assertRaises(fa.FormAnalysisError, msg=bad):
                fa.parse_exclusions(bad)
        with self.assertRaisesRegex(fa.FormAnalysisError, 'not response rows'):
            self.run_tool('csv', exclude=['99:not there'])

    def test_existing_outputs_are_not_overwritten_silently(self):
        path = self.dir / 'export.csv'
        write_csv(path, headers(), fixture_rows(), self.keys)
        argv = ['--input', str(path), '--out-dir', str(self.dir / 'out'), '--exclude', '2:dummy test entry', '--invited', '30']
        self.assertEqual(fa.main(argv), 0)
        self.assertEqual(json.loads((self.dir / 'out' / 'results.json').read_text())['retained_n'], 5)
        self.assertEqual(fa.main(argv), 2)
        with self.assertRaisesRegex(fa.FormAnalysisError, 'already has'):
            fa.run(path, self.dir / 'out')


if __name__ == '__main__':
    unittest.main()
