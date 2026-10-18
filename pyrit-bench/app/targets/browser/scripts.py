"""In-page JavaScript injected by the selector-capture and login-capture helpers.

These are the only large JS blobs in the project; keeping them here leaves the
Python that drives them readable."""
from __future__ import annotations


# ---------- click-to-capture selector picker (Playwright) ----------------

# Injected into the target page during a capture session. Builds a small
# floating panel and lets the user click the message box, send button and a
# reply bubble; for each click it generates a CSS selector and reports the set
# back to Python via the exposed __pyritCaptureReport binding. The selector
# helpers (uniq / generic) are also exposed on window.__pyritCapture so they
# can be unit-tested headlessly without the panel.
CAPTURE_JS = r"""
(() => {
  if (window.__pyritCaptureInit) {
    // Script already ran in this document (e.g. a second manual trigger) —
    // just make sure the panel is on screen, don't redeclare everything.
    if (window.__pyritCapture && window.__pyritCapture._build) window.__pyritCapture._build();
    return;
  }
  window.__pyritCaptureInit = true;

  // ---- selector generation -------------------------------------------------
  const q = (s) => { try { return document.querySelectorAll(s); } catch (e) { return []; } };
  const esc = (s) => (window.CSS && CSS.escape) ? CSS.escape(s)
                                                : String(s).replace(/[^a-zA-Z0-9_-]/g, '\\$&');
  const classesOf = (el) => {
    let raw = [];
    if (el.classList && el.classList.length) raw = Array.from(el.classList);
    else if (typeof el.className === 'string') raw = el.className.trim().split(/\s+/);
    return raw.filter((c) => c && c.length);
  };
  const attrSel = (el, a, v) => el.tagName.toLowerCase() + '[' + a + '=' + JSON.stringify(v) + ']';

  // A *unique* selector — for the input field and send button (one element).
  function uniq(el) {
    if (!el || el.nodeType !== 1) return '';
    if (el.id) { const s = '#' + esc(el.id); if (q(s).length === 1) return s; }
    for (const a of ['data-testid', 'data-test', 'data-qa', 'name', 'aria-label', 'placeholder']) {
      const v = el.getAttribute && el.getAttribute(a);
      if (v) { const s = attrSel(el, a, v); if (q(s).length === 1) return s; }
    }
    const parts = [];
    let node = el;
    while (node && node.nodeType === 1 && node !== document.documentElement) {
      if (node.id) {
        parts.unshift('#' + esc(node.id));
        const s = parts.join(' > ');
        if (q(s).length === 1) return s;
        break;
      }
      let seg = node.tagName.toLowerCase();
      const cls = classesOf(node).slice(0, 2);
      if (cls.length) seg += '.' + cls.map(esc).join('.');
      const parent = node.parentElement;
      if (parent) {
        const sibs = Array.from(parent.children).filter((c) => c.tagName === node.tagName);
        if (sibs.length > 1) seg += ':nth-of-type(' + (sibs.indexOf(node) + 1) + ')';
      }
      parts.unshift(seg);
      const s = parts.join(' > ');
      if (q(s).length === 1) return s;
      node = parent;
    }
    return parts.join(' > ');
  }

  // A *generic* selector — for reply bubbles, so it matches every assistant
  // message (current and future), not just the one clicked. Prefers a role
  // attribute or a class combination, never an id or :nth-of-type.
  function generic(el) {
    if (!el || el.nodeType !== 1) return '';
    for (const a of ['data-message-author-role', 'data-role', 'data-author', 'data-testid', 'role']) {
      const v = el.getAttribute && el.getAttribute(a);
      if (v) { const s = attrSel(el, a, v); if (q(s).length) return s; }
    }
    let node = el, hops = 0;
    while (node && node.nodeType === 1 && hops < 5) {
      const cls = classesOf(node);
      if (cls.length) { const s = '.' + cls.map(esc).join('.'); if (q(s).length) return s; }
      node = node.parentElement; hops++;
    }
    return uniq(el);
  }

  // ---- snap a click to the real control ------------------------------------
  // Chat composers overlay placeholder text (e.g. ChatGPT's "Ask anything"
  // <p class="placeholder">) and wrap send buttons around an SVG, so a click
  // often lands on the wrong node. Resolve to the element we can actually drive.
  const isEditable = (n) => n && n.nodeType === 1 && (
    n.tagName === 'TEXTAREA' ||
    (n.tagName === 'INPUT' &&
      !/^(button|submit|checkbox|radio|file|image|reset|hidden)$/i.test(n.getAttribute('type') || 'text')) ||
    n.isContentEditable
  );
  function resolveInput(el) {
    if (!el || el.nodeType !== 1) return el;
    let node = el;
    for (let i = 0; node && i < 6; i++) {        // self or nearby ancestor editable
      if (isEditable(node)) return node;
      node = node.parentElement;
    }
    const scope = el.closest('form, [class*="composer"], [class*="Composer"], '
      + '[class*="input"], [class*="prompt"], [class*="chat"]') || document;
    const sel = 'textarea, [contenteditable="true"], [contenteditable=""], '
      + 'input:not([type=button]):not([type=submit]):not([type=checkbox])'
      + ':not([type=radio]):not([type=file]):not([type=hidden])';
    return scope.querySelector(sel) || document.querySelector('textarea, [contenteditable="true"]') || el;
  }
  function resolveButton(el) {
    if (!el || el.nodeType !== 1) return el;
    return el.closest('button, [role="button"], input[type="submit"], a[role="button"]') || el;
  }

  // ---- persistent state (survives both panel rebuilds AND a hard page
  // refresh, via sessionStorage — this document's JS state is wiped on
  // refresh, but the browser tab's sessionStorage is not) -------------------
  const STORE_KEY = '__pyritCaptureStore';
  function loadStore() {
    try {
      const raw = sessionStorage.getItem(STORE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch (e) { return null; }
  }
  const _stored = loadStore();
  const data = Object.assign({ input: '', submit: '', response: '' }, (_stored && _stored.data) || {});
  let mode = null, hovered = null, prevOutline = '';
  let finished = false;   // true once "Use these" / "Cancel" was pressed
  const pre = (_stored && Array.isArray(_stored.pre)) ? _stored.pre : [];
  function persist() {
    try { sessionStorage.setItem(STORE_KEY, JSON.stringify({ data, pre })); } catch (e) {}
  }
  const report = (p) => {
    // Patchright may expose the report binding in an isolated world this overlay
    // (main world) can't reach, so also write a DOM sentinel Python can poll
    // from the shared DOM — works on stock Playwright and Patchright alike.
    try { document.documentElement.setAttribute('data-pyrit-capture', JSON.stringify(p)); } catch (e) {}
    (window.__pyritCaptureReport || function () {})(p);
  };

  const labels = { input: 'message box', submit: 'send button', response: 'reply bubble',
                   pre_click: 'element to click (e.g. an Accept or × button)' };
  let rowFor = {};
  function makeRow(key, n, text) {
    const wrap = document.createElement('div');
    wrap.style.cssText = 'margin:8px 0;';
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = n + ' · Pick ' + text;
    btn.style.cssText = 'width:100%;text-align:left;background:#eef2ff;border:1px solid #c7d2fe;color:#3730a3;'
      + 'border-radius:8px;padding:8px 10px;cursor:pointer;font:inherit;font-weight:600;';
    btn.addEventListener('click', () => setMode(key));
    const status = document.createElement('div');
    status.style.cssText = 'font-size:12px;color:#64748b;margin:3px 2px 0;word-break:break-all;min-height:14px;';
    wrap.appendChild(btn); wrap.appendChild(status);
    rowFor[key] = { btn, status, label: text };
    // Restore a previous pick after a rebuild (refresh or SPA re-render) so
    // progress already reported to Python isn't visually lost.
    if (key !== 'pre_click' && data[key]) {
      status.textContent = '✓ ' + (data[key] === 'Enter' ? 'Enter key' : data[key]);
      status.style.color = '#15803d';
      btn.style.background = '#dcfce7';
      btn.style.borderColor = '#86efac';
    }
    return wrap;
  }

  function mkPreBtn(label, fn) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = label;
    b.style.cssText = 'background:#f1f5f9;border:1px solid #cbd5e1;color:#334155;border-radius:7px;'
      + 'padding:5px 9px;font:inherit;font-size:12px;cursor:pointer;';
    b.addEventListener('click', fn);
    return b;
  }

  let preList;   // (re)assigned each build() — the setup-steps list container
  function renderPre() {
    preList.innerHTML = '';
    if (!pre.length) {
      const empty = document.createElement('div');
      empty.textContent = 'No setup steps.';
      empty.style.cssText = 'font-size:11.5px;color:#cbd5e1;';
      preList.appendChild(empty);
      return;
    }
    pre.forEach((act, i) => {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;align-items:center;gap:6px;font-size:12px;color:#334155;'
        + 'background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:4px 6px;';
      const num = document.createElement('span');
      num.textContent = (i + 1) + '.';
      num.style.cssText = 'color:#94a3b8;flex:none;';
      const desc = document.createElement('span');
      desc.style.cssText = 'flex:1;min-width:0;word-break:break-all;display:flex;align-items:center;gap:4px;';
      if (act.type === 'click') {
        desc.textContent = 'Click  ' + act.selector;
      } else if (act.type === 'key') {
        desc.textContent = 'Press ' + act.key + ' key';
      } else if (act.type === 'wait') {
        desc.appendChild(document.createTextNode('Wait'));
        const ms = document.createElement('input');
        ms.type = 'number'; ms.min = '0'; ms.step = '100'; ms.value = act.ms;
        ms.style.cssText = 'width:62px;font:inherit;font-size:12px;padding:1px 4px;'
          + 'border:1px solid #cbd5e1;border-radius:4px;';
        ms.addEventListener('input', () => { act.ms = parseInt(ms.value, 10) || 0; persist(); });
        desc.appendChild(ms);
        desc.appendChild(document.createTextNode('ms'));
      }
      const del = document.createElement('button');
      del.type = 'button';
      del.textContent = '×';
      del.title = 'Remove step';
      del.style.cssText = 'flex:none;width:20px;height:20px;line-height:1;border:1px solid #fecaca;'
        + 'background:#fef2f2;color:#b91c1c;border-radius:5px;cursor:pointer;font:inherit;font-weight:700;';
      del.addEventListener('click', () => { pre.splice(i, 1); renderPre(); persist(); });
      row.appendChild(num); row.appendChild(desc); row.appendChild(del);
      preList.appendChild(row);
    });
  }

  // ---- interaction (bound once — the panel elements they touch are looked up
  // fresh each time, so these keep working across rebuilds) ------------------
  let sub;   // (re)assigned each build() — the hint line under the header
  function setStatusHint(txt) { if (sub) sub.textContent = txt; }
  function clearHighlight() {
    if (hovered) { hovered.style.outline = prevOutline; hovered = null; }
  }
  function setMode(m) {
    mode = m;
    document.body.style.cursor = 'crosshair';
    setStatusHint('Now click the ' + labels[m] + ' on the page…');
    if (rowFor[m]) rowFor[m].btn.style.background = '#c7d2fe';
  }
  function inPanel(t) {
    const p = document.getElementById('__pyritCapturePanel');
    return !!(p && t && p.contains(t));
  }
  function onMove(e) {
    if (!mode) return;
    const t = e.target;
    if (!t || inPanel(t)) return;
    if (hovered !== t) {
      clearHighlight();
      hovered = t;
      prevOutline = t.style.outline;
      t.style.outline = '2px solid #4f46e5';
    }
  }
  function onClick(e) {
    if (!mode) return;                 // not capturing → let the page handle it
    const t = e.target;
    if (!t || inPanel(t)) return;
    e.preventDefault();
    e.stopPropagation();
    if (mode === 'pre_click') {
      const psel = uniq(resolveButton(t));
      if (psel) { pre.push({ type: 'click', selector: psel }); renderPre(); persist(); }
      clearHighlight();
      mode = null;
      document.body.style.cursor = '';
      setStatusHint('Setup step added. Click a step button, then the element on the page.');
      return;
    }
    // Snap to the element we can actually drive: the editable box for the
    // message field, the enclosing button for send. Reply uses the raw target.
    let picked = t;
    if (mode === 'input') picked = resolveInput(t);
    else if (mode === 'submit') picked = resolveButton(t);
    const sel = (mode === 'response') ? generic(t) : uniq(picked);
    data[mode] = sel;
    persist();
    const row = rowFor[mode];
    if (row) {
      row.status.textContent = '✓ ' + sel;
      row.status.style.color = '#15803d';
      row.btn.style.background = '#dcfce7';
      row.btn.style.borderColor = '#86efac';
    }
    clearHighlight();
    mode = null;
    document.body.style.cursor = '';
    setStatusHint('Click each button, then click the matching element on the page.');
  }
  document.addEventListener('click', onClick, true);
  document.addEventListener('mousemove', onMove, true);

  function teardown(msg) {
    finished = true;   // stop build() from resurrecting the panel after this
    try { sessionStorage.removeItem(STORE_KEY); } catch (e) {}
    document.removeEventListener('click', onClick, true);
    document.removeEventListener('mousemove', onMove, true);
    clearHighlight();
    document.body.style.cursor = '';
    const panelEl = document.getElementById('__pyritCapturePanel');
    if (panelEl) {
      panelEl.innerHTML = '';
      const m = document.createElement('div');
      m.textContent = msg;
      m.style.cssText = 'font-weight:600;text-align:center;padding:6px 0;';
      panelEl.appendChild(m);
    }
  }

  // ---- floating panel -------------------------------------------------------
  // Rebuilt whenever it's missing from the DOM: a hard page refresh reruns this
  // whole script (via context.add_init_script), and an SPA that wipes <body>
  // without navigating just drops the panel node, so the interval below catches
  // that case too. All the picked-selector state above (data/pre) survives
  // either way, and makeRow() re-renders any checkmarks already captured.
  function build() {
    if (finished) return;
    const existing = document.getElementById('__pyritCapturePanel');
    if (existing && existing.isConnected) return;
    const root = document.body || document.documentElement;
    if (!root) return;   // DOM not ready yet — a later tick will retry

    rowFor = {};

    const panel = document.createElement('div');
    panel.id = '__pyritCapturePanel';
    panel.style.cssText = [
      'position:fixed', 'top:14px', 'right:14px', 'width:300px', 'max-width:calc(100vw - 28px)',
      // Give flexbox a definite height. With max-height alone, a content-sized
      // flex item can still push the footer below the visible scrollport.
      'height:min(680px, calc(100vh - 28px))', 'height:min(680px, calc(100dvh - 28px))',
      'box-sizing:border-box',
      'display:flex', 'flex-direction:column', 'overflow:hidden',
      'z-index:2147483647', 'background:#ffffff', 'color:#1f2937', 'border:1px solid #cbd5e1',
      'border-radius:12px', 'box-shadow:0 10px 30px rgba(15,23,42,.25)', 'padding:10px 14px 14px',
      'font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif',
    ].join(';');

    // Draggable header so the panel never traps a control underneath it, plus a
    // minimise toggle that collapses it to a thin bar.
    const header = document.createElement('div');
    header.style.cssText = 'display:flex;align-items:center;justify-content:space-between;gap:8px;'
      + 'flex:none;cursor:move;user-select:none;touch-action:none;';
    const title = document.createElement('div');
    title.textContent = '⋮ Capture chat selectors';
    title.title = 'Drag to move';
    title.style.cssText = 'font-weight:700;font-size:15px;';
    const minBtn = document.createElement('button');
    minBtn.type = 'button';
    minBtn.textContent = '–';
    minBtn.title = 'Minimise';
    minBtn.style.cssText = 'flex:none;width:26px;height:26px;line-height:1;border:1px solid #cbd5e1;'
      + 'background:#f1f5f9;color:#475569;border-radius:7px;cursor:pointer;font:inherit;font-weight:700;';
    header.appendChild(title);
    header.appendChild(minBtn);

    const body = document.createElement('div');
    body.style.cssText = 'flex:1 1 0;min-height:0;overflow-y:auto;overscroll-behavior:contain;'
      + 'scrollbar-gutter:stable;padding-right:2px;';
    sub = document.createElement('div');
    sub.textContent = 'Drag this bar aside if it covers a button. Click a step, then click the element.';
    sub.style.cssText = 'font-size:12px;color:#64748b;margin:6px 0;';
    body.appendChild(sub);

    panel.appendChild(header);
    panel.appendChild(body);
    body.appendChild(makeRow('input', '1', 'message box'));

    // submit row + "use Enter instead" toggle
    const submitRow = makeRow('submit', '2', 'send button');
    const enterWrap = document.createElement('label');
    enterWrap.style.cssText = 'display:flex;align-items:center;gap:6px;font-size:12px;color:#475569;margin:4px 2px 0;font-weight:400;cursor:pointer;';
    const enterCb = document.createElement('input');
    enterCb.type = 'checkbox';
    enterCb.checked = data.submit === 'Enter';
    enterWrap.appendChild(enterCb);
    enterWrap.appendChild(document.createTextNode('No button — send with the Enter key'));
    enterCb.addEventListener('change', () => {
      if (enterCb.checked) {
        data.submit = 'Enter';
        rowFor.submit.btn.disabled = true;
        rowFor.submit.btn.style.opacity = '.5';
        rowFor.submit.status.textContent = '✓ Enter key';
        rowFor.submit.status.style.color = '#15803d';
      } else {
        data.submit = '';
        rowFor.submit.btn.disabled = false;
        rowFor.submit.btn.style.opacity = '1';
        rowFor.submit.status.textContent = '';
      }
      persist();
    });
    if (enterCb.checked) {
      rowFor.submit.btn.disabled = true;
      rowFor.submit.btn.style.opacity = '.5';
      rowFor.submit.status.textContent = '✓ Enter key';
      rowFor.submit.status.style.color = '#15803d';
    }
    submitRow.appendChild(enterWrap);
    body.appendChild(submitRow);

    body.appendChild(makeRow('response', '3', 'reply bubble'));

    const tip = document.createElement('div');
    tip.textContent = 'Tip: send a real message first so a reply exists, then pick it.';
    tip.style.cssText = 'font-size:11.5px;color:#94a3b8;margin:8px 2px;';
    body.appendChild(tip);

    // ---- optional setup steps (dismiss popups before sending) --------------
    // Recorded actions replayed once per run, after the page loads and before the
    // prompt is typed — to clear cookie banners / welcome modals over the input.
    const preWrap = document.createElement('div');
    preWrap.style.cssText = 'margin:10px 0 4px;padding-top:8px;border-top:1px dashed #e2e8f0;';
    const preTitle = document.createElement('div');
    preTitle.textContent = 'Setup steps (optional) — dismiss popups before sending';
    preTitle.style.cssText = 'font-weight:600;font-size:12.5px;color:#475569;margin-bottom:2px;';
    const preHint = document.createElement('div');
    preHint.textContent = 'Runs before each prompt: click a consent/close button, press a key, or wait.';
    preHint.style.cssText = 'font-size:11.5px;color:#94a3b8;margin-bottom:6px;';
    preList = document.createElement('div');
    // Setup steps are the only unbounded part of this panel. Scroll this list
    // after roughly four rows instead of letting it displace the controls.
    preList.style.cssText = 'display:flex;flex-direction:column;gap:4px;margin-bottom:6px;'
      + 'max-height:min(220px, 30dvh);overflow-y:auto;overscroll-behavior:contain;'
      + 'scrollbar-gutter:stable;padding-right:2px;';
    const preBtns = document.createElement('div');
    preBtns.style.cssText = 'display:flex;gap:6px;flex-wrap:wrap;';

    preBtns.appendChild(mkPreBtn('+ Click element', () => setMode('pre_click')));
    preBtns.appendChild(mkPreBtn('+ Enter key', () => { pre.push({ type: 'key', key: 'Enter' }); renderPre(); persist(); }));
    preBtns.appendChild(mkPreBtn('+ Esc key', () => { pre.push({ type: 'key', key: 'Escape' }); renderPre(); persist(); }));
    preBtns.appendChild(mkPreBtn('+ Wait', () => { pre.push({ type: 'wait', ms: 1000 }); renderPre(); persist(); }));
    preWrap.appendChild(preTitle);
    preWrap.appendChild(preHint);
    preWrap.appendChild(preList);
    preWrap.appendChild(preBtns);
    body.appendChild(preWrap);
    renderPre();

    const btnRow = document.createElement('div');
    btnRow.style.cssText = 'flex:none;display:flex;gap:8px;margin-top:6px;padding-top:8px;'
      + 'border-top:1px solid #e2e8f0;background:#fff;';
    const doneBtn = document.createElement('button');
    doneBtn.type = 'button';
    doneBtn.textContent = 'Use these';
    doneBtn.style.cssText = 'flex:1;background:#4f46e5;color:#fff;border:none;border-radius:8px;padding:9px;font:inherit;font-weight:700;cursor:pointer;';
    const cancelBtn = document.createElement('button');
    cancelBtn.type = 'button';
    cancelBtn.textContent = 'Cancel';
    cancelBtn.style.cssText = 'background:#f1f5f9;color:#475569;border:1px solid #cbd5e1;border-radius:8px;padding:9px 12px;font:inherit;cursor:pointer;';
    btnRow.appendChild(doneBtn); btnRow.appendChild(cancelBtn);
    panel.appendChild(btnRow);

    root.appendChild(panel);

    // --- minimise + drag (keep the panel out of the way of page controls) ---
    let collapsed = false;
    minBtn.addEventListener('click', () => {
      collapsed = !collapsed;
      body.style.display = collapsed ? 'none' : '';
      btnRow.style.display = collapsed ? 'none' : 'flex';
      minBtn.textContent = collapsed ? '+' : '–';
      minBtn.title = collapsed ? 'Expand' : 'Minimise';
    });
    let dragging = false, sx = 0, sy = 0, ox = 0, oy = 0;
    header.addEventListener('pointerdown', (e) => {
      if (e.target === minBtn) return;       // let the minimise button click through
      const r = panel.getBoundingClientRect();
      panel.style.left = r.left + 'px';
      panel.style.top = r.top + 'px';
      panel.style.right = 'auto';
      dragging = true; sx = e.clientX; sy = e.clientY; ox = r.left; oy = r.top;
      if (header.setPointerCapture) { try { header.setPointerCapture(e.pointerId); } catch (x) {} }
      e.preventDefault();
    });
    header.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const w = panel.offsetWidth || 300;
      let nx = ox + (e.clientX - sx), ny = oy + (e.clientY - sy);
      nx = Math.max(4, Math.min(nx, window.innerWidth - w - 4));
      const h = panel.offsetHeight || 30;
      ny = Math.max(4, Math.min(ny, window.innerHeight - h - 4));
      panel.style.left = nx + 'px';
      panel.style.top = ny + 'px';
    });
    const endDrag = (e) => {
      dragging = false;
      if (header.releasePointerCapture && e && e.pointerId != null) {
        try { header.releasePointerCapture(e.pointerId); } catch (x) {}
      }
    };
    header.addEventListener('pointerup', endDrag);
    header.addEventListener('pointercancel', endDrag);

    // Keep the whole panel reachable when the window is resized or moved to a
    // smaller projector display. Its body scrolls while the header stays put.
    const keepOnScreen = () => {
      const r = panel.getBoundingClientRect();
      const left = Math.max(4, Math.min(r.left, window.innerWidth - r.width - 4));
      const top = Math.max(4, Math.min(r.top, window.innerHeight - r.height - 4));
      panel.style.left = left + 'px';
      panel.style.top = top + 'px';
      panel.style.right = 'auto';
    };
    window.addEventListener('resize', keepOnScreen);

    doneBtn.addEventListener('click', () => {
      if (enterCb.checked) data.submit = 'Enter';
      report({ type: 'done', input: data.input, submit: data.submit, response: data.response, pre: pre });
      teardown('Captured ✓ — returning to the form.');
    });
    cancelBtn.addEventListener('click', () => {
      report({ type: 'cancel' });
      teardown('Capture cancelled.');
    });
  }

  // Build now, then keep watch: rebuild on every new document (this whole IIFE
  // reruns via add_init_script) and on any in-page DOM wipe an SPA causes.
  build();
  document.addEventListener('DOMContentLoaded', build);
  setInterval(build, 1000);

  window.__pyritCapture = {
    uniq, generic, _data: data,
    _setMode: setMode, _onClick: onClick, _build: build,
    _show: () => build(),
  };
})();
"""


# ---------- one-time login session capture -------------------------------

# A slim top bar injected on every document via add_init_script, so it stays on
# the visible edge of short/projected displays and survives login navigation.
LOGIN_JS = r"""
(() => {
  if (window.__pyritLoginBarInit) return;   // run the installer only once per document
  window.__pyritLoginBarInit = true;

  function build() {
    // Already present and still attached? nothing to do.
    const existing = document.getElementById('__pyritLoginBar');
    if (existing && existing.isConnected) return;
    const root = document.body || document.documentElement;
    if (!root) return;   // DOM not ready yet — a later tick will retry

    const bar = document.createElement('div');
    bar.id = '__pyritLoginBar';
    bar.style.cssText = 'position:fixed;left:0;right:0;top:0;z-index:2147483647;'
      + 'background:#1e293b;color:#fff;display:flex;align-items:center;gap:12px;'
      + 'flex-wrap:wrap;justify-content:flex-end;'
      + 'max-height:calc(100vh - 16px);max-height:calc(100dvh - 16px);overflow-y:auto;box-sizing:border-box;'
      + 'padding:max(10px, env(safe-area-inset-top)) max(16px, env(safe-area-inset-right)) 10px '
      + 'max(16px, env(safe-area-inset-left));font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;'
      + 'box-shadow:0 4px 16px rgba(0,0,0,.25);';
    const msg = document.createElement('div');
    msg.style.cssText = 'flex:1 1 240px;min-width:0;';
    msg.textContent = 'Log in to your chatbot in this window. Once you can see the chat page, click “I’m logged in”.';
    const save = document.createElement('button');
    save.textContent = "I’m logged in — save session";
    save.style.cssText = 'background:#4f46e5;color:#fff;border:none;border-radius:8px;'
      + 'padding:9px 14px;font:inherit;font-weight:700;cursor:pointer;flex:none;';
    const cancel = document.createElement('button');
    cancel.textContent = 'Cancel';
    cancel.style.cssText = 'background:transparent;color:#cbd5e1;border:1px solid #475569;'
      + 'border-radius:8px;padding:9px 12px;font:inherit;cursor:pointer;flex:none;';
    const actions = document.createElement('div');
    actions.style.cssText = 'position:sticky;bottom:0;display:flex;align-items:center;justify-content:flex-end;'
      + 'gap:8px;flex:0 1 auto;flex-wrap:wrap;background:#1e293b;';
    const report = (p) => { (window.__pyritLoginReport || function () {})(p); };
    // Patchright exposes the report binding in an isolated world the injected
    // bar can't reach, so also write a DOM sentinel Python polls from the shared
    // DOM — without this the "I'm logged in" click silently does nothing.
    const mark = (s) => { try { document.documentElement.setAttribute('data-pyrit-login', s); } catch (e) {} };
    save.addEventListener('click', () => { msg.textContent = 'Saving session…'; save.disabled = true; mark('done'); report({ type: 'done' }); });
    cancel.addEventListener('click', () => { mark('cancel'); report({ type: 'cancel' }); });
    actions.appendChild(save); actions.appendChild(cancel);
    bar.appendChild(msg); bar.appendChild(actions);
    root.appendChild(bar);

    const fitBar = () => {
      const narrow = window.innerWidth < 700;
      bar.style.alignItems = narrow ? 'stretch' : 'center';
      bar.style.flexDirection = narrow ? 'column' : 'row';
      msg.style.flex = narrow ? '0 1 auto' : '1 1 240px';
      actions.style.width = narrow ? '100%' : 'auto';
      save.style.flex = narrow ? '1 1 220px' : 'none';
      cancel.style.flex = narrow ? '1 1 90px' : 'none';
    };
    fitBar();
    window.addEventListener('resize', fitBar);
  }

  // Build as soon as we can, and keep rebuilding if the page (e.g. an SPA that
  // re-renders <body>) throws the bar away. Runs at document-start, so body may
  // not exist on the first call — the interval covers that and any later wipes.
  build();
  document.addEventListener('DOMContentLoaded', build);
  setInterval(build, 1000);
})();
"""
