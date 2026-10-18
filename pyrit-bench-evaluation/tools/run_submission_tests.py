#!/usr/bin/env python3
"""Run the included synthetic analysis suites, without hosted services."""
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PATTERNS=['test_controlled_benchmark_common.py','test_historical_threshold_analysis.py','test_study_s1_external_reference.py','test_study_s1_posthoc_sensitivity.py','test_study_s1_repeat_stability.py','test_study_s1_prepare_public_human.py','test_study_s1_semantic_tally.py','test_controlled_benchmark_analysis.py','test_sc3098_form_analysis.py']
suite=unittest.TestSuite()
for pattern in PATTERNS:suite.addTests(unittest.TestLoader().discover(str(ROOT),pattern=pattern))
result=unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(not result.wasSuccessful())
