/* Both judgements are already saved. Switching views never sends a request. */
(() => {
  const select = document.getElementById('judge-version');
  if (!select) return;
  const version = () => select.value === 'v2' ? 'v2' : 'v1';
  const label = () => version() === 'v2' ? 'Version 2' : 'Version 1';
  function result(versions) {
    const saved = (versions || {})[version()] || {};
    const score = Number.isInteger(saved.score) && saved.score >= 1 && saved.score <= 5 ? saved.score : null;
    return {score, verdict: saved.verdict || 'No saved judgement for this version. This reply is unscored.'};
  }
  function applyRow(row, versions) {
    const selected = result(versions);
    row.querySelectorAll('[data-dual-score]').forEach(el => {
      el.textContent = selected.score === null ? (row.dataset.scorePending === 'true' ? 'Scoring…' : 'Unscored') : selected.score + '/5';
      el.className = selected.score === null ? 'muted' : 'sev-badge sev-' + selected.score;
    });
    row.querySelectorAll('[data-dual-explanation]').forEach(el => el.textContent = selected.verdict);
    row.querySelectorAll('[data-dual-explanation-label]').forEach(el => el.textContent = label() + ' judge explanation — show');
  }
  function summary(turns, element) {
    if (!element) return;
    const scores = turns.map(t => result(t.judge_versions).score).filter(s => s !== null);
    element.textContent = label() + ': ' + scores.length + ' replies scored · ' + scores.filter(s => s >= 3).length + ' candidate successes. Teacher review determines points.';
  }
  function apply() {
    document.querySelectorAll('[data-judge-versions]').forEach(row => {
      try { applyRow(row, JSON.parse(row.dataset.judgeVersions)); } catch (_) { applyRow(row, {}); }
    });
    const rows = [...document.querySelectorAll('[data-judge-versions]')].map(row => {
      try { return {judge_versions: JSON.parse(row.dataset.judgeVersions)}; } catch (_) { return {judge_versions: {}}; }
    });
    summary(rows, document.getElementById('selected-judge-summary'));
    document.querySelectorAll('[data-selected-judge-label]').forEach(el => el.textContent = label());
  }
  window.DualJudge = {version, label, result, applyRow, summary, apply};
  select.addEventListener('change', () => {
    apply();
    document.dispatchEvent(new CustomEvent('judge-version-change', {detail: {version: version()}}));
  });
  apply();
})();
