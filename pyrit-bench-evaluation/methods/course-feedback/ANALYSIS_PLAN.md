> Original method/documentation snapshot. Working-project references below identify records outside this handover; they are not missing runtime dependencies.

# SC3098 PyRIT-Bench feedback survey: analysis plan

Version 2 · revised 2026-10-08T08:45:34+08:00 (SGT), before collection according to the user’s confirmation on 8 October. Version 1 was written on 6 October and is preserved in versions/v1/ANALYSIS_PLAN.md (working-project reference: `versions/v1/ANALYSIS_PLAN.md`; not bundled).
Questions and options: SURVEY_FORM.md (working-project reference: `SURVEY_FORM.md`; not bundled). This plan is what makes the results usable under the
report's completion rule (§6.4: the analysis plan is documented "before the results are seen"). Version 2 changes are recorded in §11. Any change after
responses arrive is recorded in §11 as a deviation, with its reason, and never silently.

## 1. Status and decisions

| When (SGT) | Who | Decision or recommendation |
|---|---|---|
| 6 Oct 2026, evening (time not recorded) | User | Asked for survey questions to "give me feedback about my app", made into a Google Form. |
| 6 Oct 2026, evening (time not recorded) | User | "ALL OF THEM will be operating the tool" → no operator/observer routing. |
| 6 Oct 2026, about 23:10 | User | Asked for a review "make sure they will be useful for the FYP paper". |
| 6 Oct 2026, about 23:12 | User decision, after the assistant's explanation and recommendation | Include the 5-question "Reading a result" interpretation check ("Include it (Recommended)"). This reverses, for this survey only, the 7 September choice to remove quizzes from the workshop survey (revision 6 (working-project reference: `../workshop_survey_r6/instrument/SURVEY_REVISION_6_CHANGES.md`; not bundled)). |
| 6 Oct 2026, about 23:12 | User decision, after recommendation | Save the form and this plan in the FYP records ("Save both (Recommended)"). |
| 6 Oct 2026 | Assistant recommendation, **not** a user decision | Ask the supervisor whether these responses may be used beyond descriptive feedback before sending (see §2). |
| 8 Oct 2026 | User | Requested “review, recommand, and rewrite for me”; confirmed “Not sent; no responses”. |
| 8 Oct 2026 | Assistant implementation under that request | Version 2 revised before collection; recommendations and checks in REVIEW_AND_REVISION.md (working-project reference: `REVIEW_AND_REVISION.md`; not bundled). This does not record independent user agreement with each design choice. |

Kit students used (for the method section): the SC3098 student kit with image archives dated 5 October 2026,
14:34–14:35 SGT (archive index digests: Apple-chip `9163938fe5e4…`, Windows/Intel `8eac61f2a45b…`), after
PLAN Tasks 11.51.25–11.51.55. Lab 7 was on 6 October 2026, 08:30–10:20 SGT. The survey is administered after the lab.


## 2. What the paper may and may not claim

Default status, unless the supervisor confirms otherwise in writing: **descriptive course feedback**, reported
like the September workshop survey (report §5.11), with the same safeguards.

| The paper may say | The paper must not say |
|---|---|
| How students rated ease, clarity and usefulness, as counts | That PyRIT-Bench is "usable" or "easy" in general, or easier than another tool |
| How many reported completing each task, independently or with help | That tasks were observed or verified |
| How many chose the intended answer on each interpretation question | That students "understood" or "learned"; that this is a knowledge or learning-gain test |
| The SUS score of this group, with n | A SUS grade or percentile as if the sample were representative |
| Descriptive side-by-side counts with September for the matched items | That changes between the two versions caused any difference |
| That RQ1 now has self-administered feedback, including an interpretation check | That RQ1 is answered, unless a participant route is confirmed (report §6.3); even then it is self-administered, not an observed task study |

Two concerns to state in the report regardless of approval: (1) the developer ran the activity and awarded
class-activity points, so answers may lean favourable or be cautious (social desirability; Dell et al. 2012
is already cited for facilitator bias); (2) the survey was answered after the lab, so recall may be imperfect.

## 3. Administration

1. Before opening the form: record the deployed version (v2), keep a copy of its wording and confirm that no live response collection has begun. A brief volunteer comprehension/timing pilot is recommended if feasible; record it separately, exclude its submissions, and revise the unverified 10–15-minute estimate if needed. Dummy form checks do not establish comprehension or completion time. Then test it on a phone (consent Yes and No, skipped rows, Other text), export the
   dummy responses, check the columns, delete the dummy responses. Record the time the form opens.
2. Send the link to both groups by **8 October 2026** (report due 19 October; red-team rounds start the same day).
   If possible, send after the class-activity points are final, and say again that answers do not affect them.
3. One general reminder is allowed. Never ask or check who has responded.
4. Close the form on **12 October 2026, 23:59 SGT** unless changed in §11. Record the closing time. *(Changed in §11 on 11 October 2026: the close is now 14 October 2026, 23:59 SGT.)*
5. Record how many students took part in Lab 7 (the response-rate denominator; September's was unknown).
6. Export once after closing to `data/raw_responses.xlsx` in this folder. Do not keep copies elsewhere. The raw
   export is never pasted into the report.

### Separate interpretation activity

The user accepted a separate three-case activity after a recommendation and chose two actual saved cases plus one labelled illustration after the evidence gap was explained. Pack (working-project reference: `interpretation_activity/README.md`; not bundled), private guide and separate analysis rules (working-project reference: `interpretation_activity/FACILITATOR_GUIDE.md`; not bundled). This does not replace survey R1–R5 or convert the anonymous feedback survey into an observed study.

Recommendation: activity first, survey second, with no answer feedback between them. Record administration order and prior exposure for each instrument without linking anonymous survey submissions to activity study codes. Completing the activity can influence later survey interpretation answers; disclose this and do not present the two as independent confirmation. If some survey respondents did not complete the activity, report the mixed exposure condition and do not infer who they were from timestamps.

## 4. Inclusion and exclusion

- Include every response with C1 = "Yes".
- Exclude C1 = "No" records, dummy test entries and separately recorded pilot submissions confirmed by the user (record how many and why).
- No other exclusions: no removal for speed, inconsistency or unexpected answers.
- Conflicting answers (for example, "No problem encountered" with a failed run) are kept unchanged and counted in
  a short "conflicting answers" note, as in September.

## 5. Denominators and missing answers

- Retained consenting submissions are n; item-level counts must also show answered, blank and not-applicable counts. Do not count blanks as negative answers or replace them with a midpoint. Default: counts only; any percentage states its numerator and denominator explicitly.
- A5: eligible = A4 “It reached a completed results view”. Report eligible n, time-band answers, Cannot remember and blanks separately. Keep answers from ineligible or unknown-status respondents, flag them and exclude them only from the conditional time summary.
- A6: report “I did not run Tier 2 attacks”, substantive time-impact answers and blanks. The time-impact denominator includes only the three time-impact options.
- S2–S3: analyse among S1 respondents who took part in setup; report unknown/skipped S1 separately. Keep and flag answers from S1 non-setup respondents. S1 “Cannot remember” remains distinct from non-participation or unsuccessful setup.
- A3 per row: attempted = independent + with help + attempted but not completed. Completion = independent + with help. Show all four categories and blanks; an attempted-task completion proportion divides by attempted, never by all n.
- K1 per row: rating denominator = five clarity categories. Show Not applicable and blanks separately. “Very or completely clear” uses the rating denominator.
- Checkbox items: respondent counts per option, not mentions; blank is unknown, not “No problem”. Preserve contradictory combinations and note them. A1 no usable target with a target type; S2/A7 No problem with another option are explicit conflicts.
- SUS: score only submissions answering all ten items with valid values 1–5. Report complete and incomplete n; do not impute missing items. The 0–100 score is not a percentage of tasks completed.
- R1–R5: per item show intended, other, Not sure and blank counts. Item percentages, if used, divide by all answered including Not sure. Also report answered-all-five n and all-five-intended n; use answered-all-five as the denominator for that proportion. Partial answers are not silently scored wrong.
- With anonymous submissions, n is a number of submissions, not a verified number of unique students. Ask for one submission but do not infer duplicates or deduplicate using timestamps. If a response rate is reported, call it submissions divided by invited eligible students and state that uniqueness cannot be verified.

## 6. Summaries per item

| Items | Summary | Report use |
|---|---|---|
| B1–B2 (adapted Sept B2–B3) | Counts; retain original categories in analysis | Describes the population against RQ1's "non-specialist developers" |
| B4 | Overall platform counts only | Kit diagnostics; no background cross-tabulation |
| P1 (adapted Sept P1) | Counts per point and the median | Overall ease, side by side with September |
| S1–S3 | Counts | New: self-setup (September used prepared machines) |
| A1–A2 | Counts | Target types; UI vs API connection |
| A3 (adapted Sept P7) | Table like report Table A1: independent / with help / attempted but not completed / did not attempt per task; attempted = the first three | Task completion half of RQ1; figure like report Figure 1 |
| A4–A5 (adapted Sept P8–P9) | Counts; time bands only, no mean | Run reliability and waiting time, side by side with September |
| A6 | Counts | Tier 2 run time (measured separately: one Tier 2 set at the default 10 prompts took about 3–17 minutes on 4 Oct) |
| A7 (adapted Sept P10), A8 (= Sept P11) | Counts; new options reported separately | Problems, side by side with September |
| K1 rows 1–7 (shared Sept P12 rows/options) | Table like report Table A3, plus "very or completely clear" | Clarity; row 4 (judge) is the follow-up to September's judge-visibility finding |
| K1 rows 8–11 | Same format, reported separately | Clarity of the judge explanation, "Successful", total run time and the two judge versions |
| K2 (= Sept P13) | Counts | Evidence inspection |
| U1–U10 | SUS score per complete response: odd items (answer − 1), even items (5 − answer), sum × 2.5 → 0–100. Report n, median, minimum–maximum; the mean only alongside n | Standard usability measure; answers the "custom questionnaire" limitation |
| R1–R5 | Per item: intended answer / other answer / "Not sure" / blank. Also the number of respondents with all five intended answers. No individual is graded or identified | Interpretation half of RQ1, mapped to report Table 20 |
| F1–F2 (= Sept P14–P15) | Counts | Usefulness and intention, side by side with September |
| P2, P3, F3 | Thematic coding (§8) | Themes with respondent counts |

Background answers (B1, B2, B4) are **never** cross-tabulated with other items: with two small groups, a cross-table
could identify a student. For publication, do not automatically merge singleton nominal categories into a “neighbour”. If a background category risks identification, omit that breakdown or use a clearly explained, conceptually valid aggregation. Retain original response coding in the protected analysis. The September analysis also avoided background cross-tabulation.

## 7. Answer guide for "Reading a result" (do not show to students)

| Item | Intended answer | Why, in the app students used | Report Table 20 question |
|---|---|---|---|
| R1 | "Nothing yet: no reply came back, so these cases were not tested" | Errors and invalid judge output are shown as "Not measured" and never counted as passes (SC3098 README) | Does a failed send demonstrate that the chatbot is safe? |
| R2 | “None of these 10 replies met the judge’s threshold for a successful attack; other attacks may still succeed” | The scenario explicitly provides ten valid scored replies; the count covers only this run, not all possible attacks | Does this percentage describe all possible attacks? |
| R3 | "The reply itself and the judge's explanation" | The run page says the score is a recommendation and replies must be read | What evidence supports or contradicts the score? |
| R4 | “Different scoring criteria were applied to the same saved reply” | Both versions assess the same saved reply; v1 measures harm severity, v2 assesses five safety criteria. Switching the display sends no target prompt and does not establish a change in the chatbot | Did the chatbot change, or only the scoring rule? (adapted from the threshold row) |
| R5 | "Only replies scored exactly 3" | The scenario explicitly defines a Version 1 threshold of 3; at 4, only rows scored 3 change | Interpret a changed threshold |

Exposure note for the report: the briefing slides and guide explain parts of R3–R5, so a correct answer may
reflect the teaching materials as well as the app. Report this; do not treat it as a weakness to hide.

## 8. Free-text coding

- Start codebook (extend inductively, record each new code): setup and Docker; target connection; run speed;
  failed or stopped runs; judge score or explanation; judge versions; own prompts and conversation formats;
  results and evidence; documentation and slides; positive comment; nothing or no change.
- Count respondents per theme, not mentions. One response can have several themes.
- Report paraphrases only: no quotations, no response times, nothing that names a group or describes a weakness
  in another group's chatbot. "Nothing was confusing" and "No change" are counted, as in September.
- If a second coder is available, report agreement on a sample; otherwise state single-coder coding.

## 9. Comparison with September

Comparability categories, verified against the September build sheet:

- Same main question and response options: C1, P2–P3, A8, K2, F1–F3. Help text, question order and administration differ; “exact” does not mean an identical full instrument.
- Adapted: B1–B2 change “before today” to “before this course”; P1 removes the non-operator option; A4–A5 replace “benchmark” with “run” and A4 replaces “I did not start a benchmark”. Keep their descriptive comparison qualified.
- Shared components: A7 preserves seven unchanged problem options; the settings option changes “Benchmark” to “Run”, and own-prompts/judge options are new. K1 preserves seven clarity rows and rating options, with four added rows and an adapted question stem.
- Approximate tasks: A3 rows 1, 2, 4, 5 and 8 relate to September T1–T4 and T5b, but activities/wording differ. Show as related tasks, not identical completion measures; own-prompt submission differs materially from using a workshop preset.

Present side by side as counts with each group's n. State the differences: different
students and course, app version (SC3098 copy with own prompts, Tier 2, two judge versions), self-setup with Docker
instead of prepared machines, different targets, survey after rather than at the end of the session, and
September's operator subset (13 of 20) versus all operators here. No significance tests; no causal claims.

## 10. Limitations to report

Small convenience sample in one course; self-report; the developer as instructor and points-giver; answers after
the lab; optional questions with item-level missing data; an unpiloted form; the interpretation check is
self-administered, multiple choice and partly covered by the teaching materials; no university ethics application
unless §2 changes (state exactly what was or was not obtained).

## 11. Revisions and deviations from this plan

| Date and time (SGT) | Change | Reason | Decided by |
|---|---|---|---|
| 2026-10-08T08:45:34+08:00 | Pre-collection version 2: B3/K3/F4 removed, SUS/order/mobile layout revised, platform/setup options and R2–R5 clarified, denominator and comparison rules corrected | User confirms not sent/no responses; review (working-project reference: `REVIEW_AND_REVISION.md`; not bundled) | Assistant under the user’s rewrite request |

| 2026-10-08T09:15:05+08:00 | Added separate interpretation-activity relationship and order/priming note; joined broken decision-table rows | User accepted the activity and chose a labelled illustration; survey answers/questions remain unchanged | Assistant preparation under user authority |
| 2026-10-11T21:35:00+08:00 (about) | Closing time moved from 12 October 2026, 23:59 SGT to **14 October 2026, 23:59 SGT**; the export, the Lab 7 participant count and the analysis follow the new close. No response has been read; the analysis rules are unchanged | User decision on 11 October 2026 ("postphone for 2 more days", in reply to the assistant's recommendation to report the form results descriptively after the close); reason not stated | User |
