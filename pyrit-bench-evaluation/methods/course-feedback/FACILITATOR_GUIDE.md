> Original method/documentation snapshot. Working-project references below identify records outside this handover; they are not missing runtime dependencies.

# Organiser guide - interpretation activity

Prepared 2026-10-08T09:08:59+08:00 (SGT). Private organiser document: do not share before participants finish. The user chose two actual cases plus a labelled illustrative Case C after the source gap and recommendation were explained. The complete pack is prepared but not piloted. No responses collected by this preparation.

## Purpose and scope

Measure whether course-trained student chatbot developers correctly interpret three supplied assessment cases. This is a short, observed interpretation activity using prepared app excerpts, not a validation of production-app navigation or general usability. It cannot establish learning gains, judge accuracy or causally improved understanding.

The user accepted the three-case proposal after explanation and rejected test design, written findings and safety/usefulness comparison as too burdensome. Do not expand to those tasks. A/B are actual historical saved evidence; no verified same-reply current v1/v2 pair was found. Never join separate target calls into one saved-reply case.

## Administration

1. Pilot the complete task with one volunteer. Check question comprehension, readability and the tentative 5–8-minute duration. Keep pilot submissions separate and exclude them from the main summary.
2. Invite volunteers who built LLM-powered chatbots in SC3098. The earlier 8–12-person target is a feasibility recommendation, not a statistically established minimum. Record number invited, number consenting and number completing, plus the actual app/task version.
3. Give STUDENT_PAGE.html (working-project reference: `STUDENT_PAGE.html`; not bundled) only, or use the text sheet (working-project reference: `STUDENT_TASK.md`; not bundled) if needed. Keep one presentation format across main participants and record it. Read: “Choose one answer and explain it in one sentence for each case. Use the evidence supplied and answer on your own. This does not affect your marks or class-activity points. You may skip questions or stop.” Do not add explanations of the correct answers.
4. Assign an arbitrary study code such as P01 without names or a name-to-code list. Keep this activity separate from the anonymous feedback survey. Confirm the participant information and permitted report use before administration; preparation is not institutional approval.
5. In the offline page, Begin starts the timer and reveals all three cases; Download answers ends the saved session timing. For a text-sheet session, start timing when the participant starts reading the cases after consent; stop when they finish or choose to stop. Record interruptions separately. Do not impose a scored time limit. Give logistical help if needed, record what was said, and never supply the intended answer.
6. Recommend completing this activity before the survey, with no teaching feedback between them. The survey's R1–R5 cover overlapping concepts, so record prior survey exposure if order differs. Do not silently pool exposure conditions. The activity itself may also influence later survey answers; disclose the order and avoid calling the two measures independent confirmations.
7. The offline page downloads an answer JSON file locally without showing correct answers. Collect that file only with the participant’s agreement; the page uploads nothing and stores no persistent browser data. For text-sheet use, collect the choices and one-sentence reasons; do not record screens, audio, private chatbot data or account information. Feedback can follow completion of both instruments. Retain anonymised answers and organiser observations in a restricted project location; agree the retention/access arrangement before collection rather than inventing an institutional rule.

## Prespecified scoring

Score each case's choice and reason separately, each 0 or 1. A maximum of 2 per case is possible (6 across three cases). Do not reveal scores to students as class grades. Blank and Not sure remain separate categories in the report even though they contribute no demonstrated-correctness point.

| Case | Intended choice | Reason earns 1 point when it states | Do not penalise |
|---|---|---|---|
| A | C | The reply includes notes explicitly marked staff-only/private, so the low harm score or No failure label cannot establish confidentiality compliance; identifying the distinction between harm and confidentiality is also sufficient | Agreement with harm score 1, disagreement with its exact numerical value, or different wording, if the confidentiality conclusion is correctly qualified |
| B | A | The runner/target compatibility error prevented a measured reply; absent reply/score is not evidence that the chatbot refused or resisted | Explaining this as an execution/configuration problem without naming the SDK, or wording such as “not tested in this attempt” |
| C | B | Different scoring criteria are applied to the same stored reply, so the score difference does not imply a changed target response; a correct harm-versus-scope distinction is also sufficient | Not listing all five criteria or challenging an illustrative numerical score while correctly identifying the scoring-construct difference |

A reason can earn its point even if the selected letter is wrong; preserve such inconsistencies. A correct letter without a substantive reason gets only the choice point. Do not infer a participant's intention from grammar, writing fluency or how confident they sound. Record genuinely ambiguous reasons as unresolved until adjudication; report any unresolved count separately rather than inventing a binary score. A second scorer should assess reasons independently, without seeing the first scorer's labels, if available; otherwise state single-scorer scoring.

Examples for the organiser (do not show students): A - “The reply includes staff-only notes, so a low harm score does not show that confidentiality was maintained.” B - “The attack failed to execute for this target, so the missing reply cannot be called a safe refusal.” C - “The same poem has low demonstrated harm but violates the supplied topic rule, so the two scoring criteria can produce different scores.”

## Analysis rules written before collection

- Include consenting non-pilot participants. Retain skipped items, incomplete sessions, wrong answers and help-dependent answers. Do not exclude poor performance or slow responses.
- Report each case's choice counts (each option, Not sure, blank), reason counts (supported, unsupported, blank, unresolved) and assistance status.
- Main joint outcome per case: intended choice AND supported reason, completed without answer-relevant help. Give the number observed and denominator: consenting non-pilot participants presented that case. Also show complete-response and assisted results separately. A skipped case is no demonstrated success, not automatically a wrong answer.
- Describe all-three success only among participants who answered all three choices and reasons; show complete n and incomplete n separately. Do not pool nine answers from three people as nine independent participants.
- Summarise total session time by median and range for finished sessions; show stopped sessions and interruptions separately. Interpretation points are a custom task score, not a validated scale or a percentage of general competence.
- Report prior SC3098 training, prior exposure to the survey/cases and whether the developer administered the activity. Do not claim novice performance or learning gains. Do not link study codes to the anonymous survey.
- Use paraphrases for report examples, with no identifiers. Report mistakes and conflicting responses alongside correct interpretations.
- Any wording, source, rubric or exclusion change after responses are seen requires a dated deviation and reason. Freeze the complete pack version before the first main participant; preparation is not participant validation.

## Evidence provenance and limits

CASES.json (working-project reference: `CASES.json`; not bundled) binds the A/B excerpts to original relative paths, locators and SHA-256 hashes. The student file shows exact decoded prompt/reply/judge/error text. The originals are unchanged.

A comes from the 3 October live-kit check, whose README records real target and judge calls. The saved HTML identifies RoboBot vulnerable; the README sometimes calls it BrewBot, so do not repeat that target-name inconsistency. Original Version 1 harm-only scoring ignores the saved confidential rule. Its number is not marked objectively wrong by this task; the confidentiality conclusion is the item being assessed.

B comes from a saved run page with a provider/SDK compatibility error and empty prompt/reply cells. Its historical missing-reply labels are not represented as an exact modern Not measured screenshot. The error is shown as evidence, not as current setup advice.

C is fully authored and explicitly labelled as an illustration, chosen by the user after recommendation. It is not evidence of actual v1/v2 disagreement frequency or of judge accuracy. The illustrative v2 score follows the app’s maximum-of-five rule; its outside-scope criterion is 4 and the others are 1. No direct override or fictional bypass instruction is present, so no extra injection/jailbreak failure is assumed.

The supplied excerpts state parts of the scoring context. Correct answers therefore show interpretation with context provided, not independent recall of the app's rules. If presented as Markdown/text, no original app-navigation behaviour is observed. Paper wording must name that presentation format.

## Observation template (one row per participant)

| Study code | Consent | Pilot/main | Prior case/survey exposure | Start/end (SGT) | Finished/stopped | Interruption | Help given (exact brief wording) | A choice/reason | B choice/reason | C choice/reason |
|---|---|---|---|---|---|---|---|---|---|---|
| P01 | | | | | | | | | | |

Do not use this illustrative row as participant data. Store raw answers separately from scorer labels; do not overwrite what the student actually wrote.

## Limits of the offline page

The page reproduces selected text evidence, not the original app interface. All evidence is visible after Begin. It does not track original-app navigation or prove that a participant read the explanation; inspection is not a separately scored outcome. The organiser records assistance and interruptions separately. Export does not automatically score answers, enforce uniqueness or establish successful participation. A timestamp-derived/random response code is for local file separation, not authenticated identity. Do not collect declined submissions.

## Google Forms adaptation

See GOOGLE_FORM.md (working-project reference: `GOOGLE_FORM.md`; not bundled) for the separately prepared form. Google Forms stores submitted answers and may store a consent-only declined record; exclude those records from activity analysis. It does not record elapsed time/help or assign the offline response code. Do not claim timing or participant-level observation/answer links without a documented observation protocol. The form remains unpublished; confirm access, retention and data use, then pilot before main collection.

## Complementary practical observation

The user accepted the separate three-check lab observation sheet (working-project reference: `LAB_OBSERVATION.md`; not bundled), with no added student questions or exercises. It observes target setup, obtaining a saved reply and opening its assessment during normal use. It does not expand the interpretation activity to test design, written findings or safety/usefulness comparisons. Group observations are group-level evidence; do not link observation codes to the anonymous forms or imply that activity consent covers observation.

## Version 2 Google Form — analysis rules added 9 October 2026, before any response

The deployed form is version 2 (working-project reference: `versions/v2_google_form_2026-10-09.md`; not bundled). These rules extend the rules above for it; they were written before any response was collected.

- **Consent.** There is no consent question (user decision, 9 October 2026); from version 4 the form states "By submitting this form, you agree to take part." Treat every non-pilot submission as agreement to take part; there are no declined records to exclude. Report this in the method.
- **Denominator.** n = number of non-pilot submissions. For each item, report answered, blank and (for A1/B1/C1) Not sure counts; do not treat blanks as wrong answers.
- **Cases A–C.** Intended choices are unchanged (A → C, B → A, C → B) and the scoring table above still applies. Timing and assistance are not recorded by Google Forms, so report the choice and reason outcomes without the "without answer-relevant help" condition, and say so.
- **Case B (version 3).** The answer cue was removed and B1 rewritten on 9 October 2026, before any response; the intended choice is still A and the scoring table above applies unchanged.
- **P1 (version 6: multiple choice No / Yes, once / Yes, more than once / Not sure).** Report counts per option, including blanks and Not sure. Use it only to describe how new automated red-teaming was to respondents; do not use it to explain differences between students (sample too small). Ignore empty sheet columns left by rows removed in version 5.
- **D1.** Report counts per option, including Other (with paraphrased Other text), "did not review" and "cannot remember". These are self-reports, not observed behaviour.
- **D2.** Summarise descriptively: how many mention the prompt, chatbot reply, score, judge explanation, error message or none. Paraphrase any example; never quote identifiers or private chatbot details.
- **Duplicates.** One response per person cannot be enforced without sign-in; report the number of submissions and note that duplicates cannot be ruled out.

- **Page 6 (version 6), rules written before any response.**
  - O1 SUS: score only respondents who answered all ten rows. Odd rows: score − 1; even rows: 5 − score; sum × 2.5 gives 0–100. Report n, median, range and mean, and the number of incomplete O1 responses separately. Do not compare with published SUS benchmarks as if the sample were representative; small n.
  - O2, O4: counts per option; report "Cannot judge" and "I do not expect to need this kind of tool" separately from the rating scale and do not count them as low ratings.
  - O3: counts per row and column; "Did not see this" is reported separately and excluded from clarity percentages.
  - O5, O6: descriptive themes with counts; paraphrase examples, no quotations that could identify a student or group.
  - O2, O4, O5 and O6 share wording with the September workshop survey (P14/P15/P2/P3); comparisons are descriptive only (different groups, setting and app version).
