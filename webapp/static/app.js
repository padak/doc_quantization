/* =========================================================================
 * doc_quantization — observability console
 * Vanilla ES2020+. No frameworks, no build step, no external assets.
 *
 * The app is a guided stepper: every view knows its prerequisite, says so in
 * plain language when it is not met, and ends with the next action.
 * ========================================================================= */
'use strict';

(function () {
  // --------------------------------------------------------------- constants

  const VIEWS = ['document', 'chunks', 'batch', 'llm', 'redaction', 'report', 'settings'];
  const KINDS = ['real', 'honeytoken', 'chaff', 'canary'];
  const EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max'];
  const INTRO_KEY = 'dq.intro.dismissed';
  const LOG_LIMIT = 4000;

  const STEPS = [
    { id: 'document', n: 1, label: 'Document' },
    { id: 'chunks', n: 2, label: 'Chunks' },
    { id: 'batch', n: 3, label: 'Batch' },
    { id: 'llm', n: 4, label: 'LLM I/O' },
    { id: 'redaction', n: 5, label: 'Redaction' },
  ];

  const AUX_STEPS = [
    { id: 'report', label: 'Report' },
    { id: 'settings', label: 'Settings' },
  ];

  const STEP_LABELS = {
    document: 'Document',
    chunks: 'Chunks',
    batch: 'Batch',
    llm: 'LLM I/O',
    redaction: 'Redaction',
    report: 'Report',
    settings: 'Settings',
  };

  // Plain-language definitions, shown by the circled "i" buttons.
  const GLOSSARY = {
    honeytoken: {
      title: 'Honeytoken',
      body: 'A planted fragment with fake names we already know the answer for. It travels with the real ones, and because we know what should come back, it measures how many names the detector actually catches (its recall).',
    },
    chaff: {
      title: 'Chaff',
      body: 'Decoy fragments the provider cannot tell apart from real ones. They dilute and poison any copy that is retained on the other side — like radar chaff, they make the real signal one blip among many.',
    },
    canary: {
      title: 'Canary',
      body: 'A unique fabricated fact seeded into the traffic. Nobody else on earth knows it, so if a model ever repeats it back to us, that is evidence our data was used for training.',
    },
    extended: {
      title: 'Extended cut',
      body: 'The chunk boundary was moved by a few tokens so that a name is never split in half. A name cut in two would be invisible to the detector, and would survive into the redacted document.',
    },
    effort: {
      title: 'Effort',
      body: 'How much internal reasoning the provider model is allowed to spend on each fragment. Higher effort usually catches more names and costs more time and money.',
    },
    recall: {
      title: 'Recall',
      body: 'The share of planted names that came back found. Recall of 1.0 means the detector spotted every honeytoken name; 0.8 means one name in five slipped through unredacted.',
    },
    real: {
      title: 'Real fragment',
      body: 'One chunk of your actual document, stored and submitted under a random ID. Nothing tells the provider which fragments are yours or in which order they belong.',
    },
    chunk: {
      title: 'Chunk',
      body: 'A short slice of the document, a few hundred tokens long, stored under a random ID. Chunks are what gets sent — never the whole document, never in order.',
    },
    provider: {
      title: 'Provider view',
      body: 'The same batch, rendered the way the receiving side sees it: shuffled fragments under opaque IDs, roughly half of them synthetic, with no labels and no ordering.',
    },
    probe: {
      title: 'Canary probe',
      body: 'Asks a model, in plain conversation, about each planted canary person. If the answer contains the fabricated fact, the canary is tripped and the batch is evidence of training misuse.',
    },
    chaff_ratio: {
      title: 'chaff_ratio',
      body: 'How many chaff fragments are generated per real chunk. A ratio of 1.0 means one decoy for every real fragment, so half of the batch is noise.',
    },
    honeytoken_rate: {
      title: 'honeytoken_rate',
      body: 'How often a honeytoken is planted, relative to the number of real chunks. More honeytokens means a more precise recall measurement and slightly more traffic.',
    },
    canaries_per_batch: {
      title: 'canaries_per_batch',
      body: 'How many unique fabricated facts are seeded into each batch, to be probed for later.',
    },
    chunk_size_tokens: {
      title: 'chunk_size_tokens',
      body: 'The target size of a chunk in tokens. Smaller chunks leak less context per request; larger chunks give the detector more context to recognise a name.',
    },
    local_llm: {
      title: 'Local LLM',
      body: 'A model running on this machine (typically Ollama) that writes the synthetic prose for honeytokens and chaff. It never sees your document — it only invents decoy text — so nothing real leaves the machine to produce it.',
    },
  };

  const REMEDY = {
    anthropic: 'Paste a valid key into the Anthropic API key field above and save.',
    local_llm: 'Start the local model server (for example: ollama serve) and pull the configured model, or turn off "Use local LLM for synthetic prose".',
    markitdown: 'Install the markitdown package into the server environment.',
    database: 'Check that the data directory exists and is writable by the server process.',
  };

  // ------------------------------------------------------------------- state

  const store = {
    settings: null,
    docs: [],
    doc: null,               // { doc_id, name, path, markdown, chunks[] }
    detect: null,            // final (or in-flight) detection object
    detectDocId: null,
    run: null,               // live streaming run: see startRun()
    redaction: null,         // GET .../redaction payload (+ has_detection, chunk_count, ...)
    redactionDocId: null,
    report: null,
    verify: null,            // POST /api/verify payload
    verifyError: null,       // why the check itself could not run
    verifyAuto: false,
    batchView: 'local',      // 'local' | 'provider'
    introDismissed: false,
    hint: null,              // { view, text } — why the user was sent here
    busy: {
      upload: false,
      detect: false,
      redaction: false,
      report: false,
      probe: false,
      settings: false,
      verify: false,
    },
    alerts: {},              // view -> { kind, text }
  };

  let runTimer = null;

  // ------------------------------------------------------------------- utils

  const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

  function esc(value) {
    if (value === null || value === undefined) return '';
    return String(value).replace(/[&<>"']/g, (c) => ESCAPES[c]);
  }

  function num(value, fallback) {
    const n = Number(value);
    return Number.isFinite(n) ? n : (fallback === undefined ? 0 : fallback);
  }

  function fmtInt(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n.toLocaleString('en-US') : '—';
  }

  function fmtPct(fraction) {
    const n = Number(fraction);
    if (!Number.isFinite(n)) return '—';
    return (n * 100).toFixed(1) + '%';
  }

  function fmtMs(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '—';
    if (n < 1000) return n.toFixed(0) + ' ms';
    return (n / 1000).toFixed(2) + ' s';
  }

  function fmtSec(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '?s';
    return (n / 1000).toFixed(1) + 's';
  }

  function fmtDate(value) {
    if (!value) return '—';
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return String(value);
    return d.toLocaleString();
  }

  function basename(path) {
    if (!path) return '';
    const parts = String(path).split(/[\\/]/);
    return parts[parts.length - 1] || String(path);
  }

  function safeArray(value) {
    return Array.isArray(value) ? value : [];
  }

  function pretty(value) {
    try {
      return JSON.stringify(value, null, 2);
    } catch (err) {
      return String(value);
    }
  }

  function shortId(value) {
    const text = String(value === null || value === undefined ? '' : value);
    return text.length > 10 ? text.slice(0, 8) : text;
  }

  function kindClass(kind) {
    return KINDS.includes(kind) ? kind : 'neutral';
  }

  function kindBadge(kind) {
    const k = String(kind || 'unknown');
    return '<span class="badge badge-' + esc(kindClass(k)) + '">' + esc(k) + '</span>';
  }

  function statusPill(status) {
    const s = String(status || 'unknown').toLowerCase();
    const known = ['ok', 'refusal', 'error', 'pending'].includes(s) ? s : 'unknown';
    return '<span class="pill pill-' + known + '">' + esc(s) + '</span>';
  }

  function copyId(value, label) {
    if (!value) return '<span class="mono dim">—</span>';
    const text = label === undefined ? value : label;
    return '<button type="button" class="copy-id" data-copy="' + esc(value) +
      '" title="Click to copy ' + esc(value) + '">' + esc(text) + '</button>';
  }

  /** A circled "i" that opens the plain-language definition of `key`. */
  function info(key) {
    const entry = GLOSSARY[key];
    if (!entry) return '';
    return '<button type="button" class="info" data-info="' + esc(key) +
      '" aria-expanded="false" aria-haspopup="dialog" title="What is ' + esc(entry.title) +
      '?" aria-label="What is ' + esc(entry.title) + '?">i</button>';
  }

  /** Term text followed by its info button. */
  function term(text, key) {
    return '<span class="term">' + esc(text) + '</span>' + info(key);
  }

  function emptyState(title, hintHtml, buttonsHtml) {
    return '<div class="empty"><div class="empty-title">' + esc(title) + '</div>' +
      '<div class="empty-hint">' + (hintHtml || '') + '</div>' +
      (buttonsHtml ? '<div class="btn-row">' + buttonsHtml + '</div>' : '') + '</div>';
  }

  function ctaHtml(targetId, label, noteHtml) {
    let html = '<div class="cta-row">';
    html += '<a class="btn btn-primary" href="#' + esc(targetId) + '">' + esc(label) + ' &rarr;</a>';
    if (noteHtml) html += '<span class="cta-note">' + noteHtml + '</span>';
    return html + '</div>';
  }

  // ---------------------------------------------------------- toasts, alerts

  function toast(kind, message) {
    const host = document.getElementById('toasts');
    if (!host) return;
    const node = document.createElement('div');
    node.className = 'toast toast-' + (['ok', 'error', 'info'].includes(kind) ? kind : 'info');
    node.textContent = message;
    host.appendChild(node);
    window.setTimeout(() => {
      if (node.parentNode) node.parentNode.removeChild(node);
    }, kind === 'error' ? 9000 : 4500);
  }

  function setAlert(view, kind, text) {
    store.alerts[view] = { kind: kind, text: text };
  }

  function clearAlert(view) {
    delete store.alerts[view];
  }

  function alertHtml(view) {
    const a = store.alerts[view];
    if (!a) return '';
    const cls = ['error', 'ok', 'warn', 'info'].includes(a.kind) ? a.kind : 'info';
    const title = cls === 'error' ? 'Request failed' : (cls === 'ok' ? 'Done' : 'Notice');
    return '<div class="alert alert-' + cls + '"><span class="alert-title">' +
      esc(title) + '</span> — ' + esc(a.text) + '</div>';
  }

  function hintHtml(view) {
    if (!store.hint || store.hint.view !== view) return '';
    // The hint explains a lock; once the lock is gone the hint is noise.
    if (store.hint.about && !stepState(store.hint.about).locked) {
      store.hint = null;
      return '';
    }
    return '<div class="hint-banner"><span class="hint-mark">&rarr;</span><span>' +
      store.hint.text + '</span></div>';
  }

  function fail(view, err) {
    const message = err && err.message ? err.message : String(err);
    setAlert(view, 'error', message);
    toast('error', message);
  }

  // --------------------------------------------------------------------- api

  function errorDetail(raw, res, path) {
    let data = null;
    if (raw) {
      try {
        data = JSON.parse(raw);
      } catch (err) {
        data = null;
      }
    }
    if (data && typeof data.detail === 'string') return data.detail;
    if (data && data.detail !== undefined && data.detail !== null) return pretty(data.detail);
    const trimmed = (raw || '').trim();
    if (!trimmed || trimmed.charAt(0) === '<') {
      return 'HTTP ' + res.status + ' ' + res.statusText + ' from ' + path;
    }
    return trimmed.slice(0, 300);
  }

  async function api(path, options) {
    let res;
    try {
      res = await fetch(path, options);
    } catch (err) {
      throw new Error('Network error contacting ' + path + ' (' +
        (err && err.message ? err.message : 'unknown') + ')');
    }

    const raw = await res.text();

    if (!res.ok) {
      const error = new Error(errorDetail(raw, res, path));
      error.status = res.status;
      throw error;
    }

    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch (err) {
      return null;
    }
  }

  function apiJson(path, method, body) {
    return api(path, {
      method: method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  // ----------------------------------------------------------------- loaders

  async function loadSettings() {
    store.settings = await api('/api/settings');
  }

  async function loadDocs() {
    store.docs = safeArray(await api('/api/documents'));
  }

  function adoptDocument(payload, fallbackName) {
    if (!payload) return;
    const path = payload.path || '';
    store.doc = {
      doc_id: payload.doc_id,
      name: payload.filename || basename(path) || fallbackName || payload.doc_id,
      path: path,
      markdown: typeof payload.markdown === 'string' ? payload.markdown : '',
      chunks: safeArray(payload.chunks),
    };
    // A new active document invalidates everything derived from the old one.
    if (store.detectDocId !== store.doc.doc_id) {
      store.detect = null;
      store.detectDocId = null;
      store.run = null;
    }
    if (store.redactionDocId !== store.doc.doc_id) {
      store.redaction = null;
      store.redactionDocId = null;
    }
    updateSidebarDoc();
  }

  function updateSidebarDoc() {
    const nameEl = document.getElementById('active-doc-name');
    const idEl = document.getElementById('active-doc-id');
    if (!nameEl || !idEl) return;
    if (store.doc) {
      nameEl.textContent = store.doc.name;
      nameEl.classList.remove('is-empty');
      idEl.textContent = store.doc.doc_id || '';
    } else {
      nameEl.textContent = 'none';
      nameEl.classList.add('is-empty');
      idEl.textContent = '';
    }
  }

  // ------------------------------------------------------------ step states

  function hasRun() {
    return Boolean(store.detect && store.doc && store.detectDocId === store.doc.doc_id &&
      safeArray(store.detect.results).length > 0);
  }

  function hasSubmittedBatch() {
    return Boolean(store.detect && store.doc && store.detectDocId === store.doc.doc_id);
  }

  /** True when we know a detection has been recorded for the active document. */
  function detectionKnown() {
    if (hasRun()) return true;
    return Boolean(store.redaction && store.doc && store.redactionDocId === store.doc.doc_id &&
      store.redaction.has_detection === true);
  }

  /**
   * State of one stepper entry.
   *
   * Returns { locked, done, current, reason, unblock }. `reason` is the plain
   * sentence shown on hover and inline once the user clicks a locked step.
   */
  function stepState(id) {
    const view = currentView();
    const hasDoc = Boolean(store.doc);
    let locked = null;
    let done = false;

    switch (id) {
      case 'document':
        done = hasDoc;
        break;
      case 'chunks':
        if (!hasDoc) locked = { reason: 'Upload a document first', unblock: 'document' };
        else done = safeArray(store.doc.chunks).length > 0;
        break;
      case 'batch':
        if (!hasDoc) locked = { reason: 'Upload a document first', unblock: 'document' };
        else done = hasRun();
        break;
      case 'llm':
        if (!hasDoc) locked = { reason: 'Upload a document first', unblock: 'document' };
        else if (!hasSubmittedBatch()) locked = { reason: 'Run detection first', unblock: 'batch' };
        else done = hasRun();
        break;
      case 'redaction':
        if (!hasDoc) locked = { reason: 'Upload a document first', unblock: 'document' };
        else done = detectionKnown() && Boolean(store.redaction) &&
          store.redactionDocId === store.doc.doc_id;
        break;
      default:
        break;
    }

    return {
      locked: Boolean(locked),
      done: done,
      current: id === view,
      reason: locked ? locked.reason : '',
      unblock: locked ? locked.unblock : null,
    };
  }

  function renderNav() {
    const view = currentView();
    const stepsHost = document.getElementById('nav-steps');
    const auxHost = document.getElementById('nav-aux');
    if (!stepsHost || !auxHost) return;

    let html = '';
    STEPS.forEach((step) => {
      const state = stepState(step.id);
      const classes = ['nav-item'];
      if (state.current) classes.push('is-active');
      if (state.locked) classes.push('is-locked');
      else if (state.done) classes.push('is-done');

      const href = state.locked ? '#' + state.unblock : '#' + step.id;
      const title = state.locked
        ? step.label + ' is locked: ' + state.reason + ' — click to go there'
        : STEP_LABELS[step.id];
      const mark = state.done && !state.locked ? '&#10003;' : String(step.n);

      html += '<a class="' + classes.join(' ') + '" href="' + href + '"' +
        (state.locked ? ' data-locked="' + esc(step.id) + '" aria-disabled="true"' : '') +
        ' data-nav="' + esc(step.id) + '" title="' + esc(title) +
        '" aria-label="' + esc(title) + '">';
      html += '<span class="nav-mark" aria-hidden="true">' + mark + '</span>';
      html += '<span class="nav-label">' + esc(step.label) + '</span>';
      if (state.locked) html += '<span class="nav-tag">locked</span>';
      else if (state.done && !state.current) html += '<span class="nav-tag">done</span>';
      html += '</a>';
    });
    stepsHost.innerHTML = html;

    let aux = '';
    AUX_STEPS.forEach((step) => {
      aux += '<a class="nav-item' + (view === step.id ? ' is-active' : '') +
        '" href="#' + step.id + '" data-nav="' + step.id + '" title="' + esc(step.label) + '">';
      aux += '<span class="nav-mark" aria-hidden="true">&#183;</span>';
      aux += '<span class="nav-label">' + esc(step.label) + '</span></a>';
    });
    auxHost.innerHTML = aux;

    stepsHost.querySelectorAll('[data-locked]').forEach((node) => {
      node.addEventListener('click', (event) => {
        event.preventDefault();
        const id = node.getAttribute('data-locked');
        const state = stepState(id);
        if (!state.locked) {
          location.hash = '#' + id;
          return;
        }
        goToStep(state.unblock, {
          view: state.unblock,
          about: id,
          text: esc(STEP_LABELS[id]) + ' is locked: ' + esc(state.reason) + '.',
        });
      });
    });
  }

  function goToStep(id, hint) {
    store.hint = hint || null;
    const target = '#' + id;
    if (location.hash === target) {
      onRoute();
    } else {
      location.hash = target;
    }
  }

  // ------------------------------------------------------------------ router

  function currentView() {
    const hash = String(location.hash || '').replace(/^#/, '');
    return VIEWS.includes(hash) ? hash : 'document';
  }

  function viewNode(name) {
    return document.querySelector('.view[data-view="' + name + '"]');
  }

  function render() {
    const view = currentView();

    if (popoverTrigger && !document.contains(popoverTrigger)) closePopover(false);

    VIEWS.forEach((name) => {
      const node = viewNode(name);
      if (node) node.hidden = name !== view;
    });

    renderNav();
    updateSidebarDoc();

    const node = viewNode(view);
    if (!node) return;

    switch (view) {
      case 'document': renderDocument(node); break;
      case 'chunks': renderChunks(node); break;
      case 'batch': renderBatch(node); break;
      case 'llm': renderLlm(node); break;
      case 'redaction': renderRedaction(node); break;
      case 'report': renderReport(node); break;
      case 'settings': renderSettings(node); break;
      default: break;
    }
  }

  function onRoute() {
    const view = currentView();

    // A locked step never renders: send the user where the lock is opened.
    const state = stepState(view);
    if (state.locked) {
      store.hint = {
        view: state.unblock,
        about: view,
        text: esc(STEP_LABELS[view]) + ' is locked: ' + esc(state.reason) + '.',
      };
      location.hash = '#' + state.unblock;
      return;
    }

    if (store.hint && store.hint.view !== view) store.hint = null;

    render();

    // Lazily fetch what this view needs.
    if (view === 'redaction' && store.doc && !store.busy.redaction &&
        (!store.redaction || store.redactionDocId !== store.doc.doc_id)) {
      fetchRedaction();
    }
    if (view === 'report' && !store.report && !store.busy.report) {
      fetchReport();
    }
    if (view === 'settings' && !store.verifyAuto && !store.busy.verify) {
      store.verifyAuto = true;
      runVerify();
    }
  }

  // -------------------------------------------------------- view: document

  function introHtml() {
    if (store.introDismissed) return '';
    let html = '<div class="intro" id="intro-card">';
    html += '<button type="button" class="intro-dismiss" id="intro-dismiss" aria-label="Dismiss the introduction">Dismiss</button>';
    html += '<h3>What this pipeline does</h3>';
    html += '<p>Your document is converted to Markdown and cut into small ' + term('chunks', 'chunk') +
      ', each stored under a random ID, so the whole document never travels as one piece.</p>';
    html += '<p>Those chunks are shuffled together with synthetic decoys — ' + term('honeytokens', 'honeytoken') +
      ', ' + term('chaff', 'chaff') + ' and ' + term('canaries', 'canary') +
      ' — and every fragment is sent to the provider on its own, unlabelled and out of order.</p>';
    html += '<p>The names that come back are stored, and the document is reassembled locally with every detected name replaced by a placeholder.</p>';
    html += '</div>';
    return html;
  }

  function renderDocument(root) {
    const busy = store.busy.upload;

    let html = '';
    html += '<div class="page-head">';
    html += '<h2 class="page-title">Document</h2>';
    html += '<p class="page-lead">Step 1 of 5 — upload a source document; it is converted to Markdown and split into chunks stored under random IDs.</p>';
    html += '</div>';

    html += introHtml();
    html += hintHtml('document');
    html += alertHtml('document');

    html += '<div class="dropzone' + (busy ? ' is-busy' : '') +
      '" id="dropzone" tabindex="0" role="button" aria-label="Upload a document">';
    if (busy) {
      html += '<div class="dropzone-busy"><span class="spinner"></span><span>Converting and chunking</span></div>';
    } else {
      html += '<div class="dropzone-title">Drop a document here, or click to choose a file</div>';
      html += '<div class="dropzone-hint">PDF, DOCX, PPTX, HTML, TXT and Markdown. The file is converted to Markdown on this machine.</div>';
    }
    html += '</div>';
    html += '<input type="file" id="file-input" style="display:none">';

    if (store.doc) {
      const chunkCount = safeArray(store.doc.chunks).length;
      html += '<h3 class="section-title">Active document</h3>';
      html += '<div class="card">';
      html += '<div class="row-between" style="margin-bottom:12px">';
      html += '<div style="min-width:0"><div style="font-size:15px;font-weight:600;word-break:break-all">' +
        esc(store.doc.name) + '</div>';
      if (store.doc.path && store.doc.path !== store.doc.name) {
        html += '<div class="mono dim" style="word-break:break-all">' + esc(store.doc.path) + '</div>';
      }
      html += '</div>';
      html += '<div class="chips">';
      html += '<span class="chip"><span class="chip-k">doc_id</span><span class="chip-v">' + copyId(store.doc.doc_id) + '</span></span>';
      html += '<span class="chip"><span class="chip-k">chunks</span><span class="chip-v">' + fmtInt(chunkCount) + '</span></span>';
      html += '</div>';
      html += '</div>';
      html += '<h4 class="section-title" style="margin-top:0">Converted Markdown</h4>';
      html += '<pre class="code code-tall">' + esc(store.doc.markdown || '(empty)') + '</pre>';
      html += '</div>';
    }

    html += '<h3 class="section-title">Previously ingested documents</h3>';
    if (!store.docs.length) {
      html += emptyState('No documents yet', 'Uploaded documents appear here and can be re-opened at any time.');
    } else {
      html += '<div class="doc-list">';
      store.docs.slice().reverse().forEach((doc) => {
        const active = store.doc && store.doc.doc_id === doc.doc_id;
        html += '<button type="button" class="doc-row' + (active ? ' is-active' : '') +
          '" data-doc="' + esc(doc.doc_id) + '">';
        html += '<div class="doc-row-main">';
        html += '<div class="doc-row-path">' + esc(basename(doc.path) || doc.doc_id) + '</div>';
        html += '<div class="doc-row-meta">' + esc(doc.doc_id) + ' &middot; ' + esc(fmtDate(doc.created_at)) + '</div>';
        html += '</div>';
        html += '<div class="doc-row-count">' + fmtInt(doc.chunk_count) + ' chunks</div>';
        html += '</button>';
      });
      html += '</div>';
    }

    if (store.doc) {
      html += ctaHtml('chunks', 'Continue to Chunks',
        'See exactly how "' + esc(store.doc.name) + '" was cut up.');
    }

    root.innerHTML = html;
    wireDocument(root);
  }

  function wireDocument(root) {
    const zone = root.querySelector('#dropzone');
    const input = root.querySelector('#file-input');

    if (zone && input) {
      zone.addEventListener('click', () => {
        if (!store.busy.upload) input.click();
      });
      zone.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          if (!store.busy.upload) input.click();
        }
      });
      ['dragenter', 'dragover'].forEach((name) => {
        zone.addEventListener(name, (event) => {
          event.preventDefault();
          event.stopPropagation();
          zone.classList.add('is-drag');
        });
      });
      ['dragleave', 'dragend'].forEach((name) => {
        zone.addEventListener(name, (event) => {
          event.preventDefault();
          zone.classList.remove('is-drag');
        });
      });
      zone.addEventListener('drop', (event) => {
        event.preventDefault();
        event.stopPropagation();
        zone.classList.remove('is-drag');
        const files = event.dataTransfer && event.dataTransfer.files;
        if (files && files.length) uploadFile(files[0]);
      });
      input.addEventListener('change', () => {
        if (input.files && input.files.length) uploadFile(input.files[0]);
        input.value = '';
      });
    }

    root.querySelectorAll('[data-doc]').forEach((button) => {
      button.addEventListener('click', () => openDocument(button.getAttribute('data-doc')));
    });

    const dismiss = root.querySelector('#intro-dismiss');
    if (dismiss) {
      dismiss.addEventListener('click', () => {
        store.introDismissed = true;
        try {
          window.localStorage.setItem(INTRO_KEY, '1');
        } catch (err) {
          /* private mode / storage disabled — the dismissal is session-only */
        }
        render();
      });
    }
  }

  async function uploadFile(file) {
    if (!file || store.busy.upload) return;
    clearAlert('document');
    store.busy.upload = true;
    render();

    const form = new FormData();
    form.append('file', file, file.name);

    try {
      const payload = await api('/api/documents', { method: 'POST', body: form });
      adoptDocument(payload, file.name);
      toast('ok', 'Ingested "' + (store.doc ? store.doc.name : file.name) + '" — ' +
        (store.doc ? safeArray(store.doc.chunks).length : 0) + ' chunks');
      try {
        await loadDocs();
      } catch (err) {
        /* the document list is secondary */
      }
      prefetchRedaction();
    } catch (err) {
      fail('document', err);
    } finally {
      store.busy.upload = false;
      render();
    }
  }

  async function openDocument(docId) {
    if (!docId) return;
    clearAlert('document');
    try {
      const payload = await api('/api/documents/' + encodeURIComponent(docId));
      adoptDocument(payload);
      toast('info', 'Active document: ' + (store.doc ? store.doc.name : docId));
      prefetchRedaction();
    } catch (err) {
      fail('document', err);
    }
    render();
  }

  // ----------------------------------------------------------- view: chunks

  function renderChunks(root) {
    let html = '';
    html += '<div class="page-head">';
    html += '<h2 class="page-title">Chunks</h2>';
    html += '<p class="page-lead">Step 2 of 5 — the document as it is actually stored: small slices under random IDs, with token boundaries shown as alternating shading.</p>';
    html += '</div>';

    html += hintHtml('chunks');
    html += alertHtml('chunks');

    if (!store.doc) {
      root.innerHTML = html + emptyState('No active document',
        'Upload a document first.',
        '<a class="btn btn-primary" href="#document">Go to Document</a>');
      return;
    }

    const chunks = safeArray(store.doc.chunks).slice().sort((a, b) => num(a.seq) - num(b.seq));
    const totalTokens = chunks.reduce((sum, chunk) => {
      const count = Number(chunk.token_count);
      if (Number.isFinite(count)) return sum + count;
      return sum + safeArray(chunk.tokens).length;
    }, 0);
    const extended = chunks.filter((chunk) => chunk.extended === true).length;

    html += '<div class="stat-row">';
    html += '<div class="stat"><div class="stat-k">Chunks</div><div class="stat-v">' + fmtInt(chunks.length) + '</div></div>';
    html += '<div class="stat"><div class="stat-k">Total tokens</div><div class="stat-v">' + fmtInt(totalTokens) + '</div></div>';
    html += '<div class="stat"><div class="stat-k">Extended cuts' + info('extended') + '</div><div class="stat-v">' +
      fmtInt(extended) + '</div><div class="stat-sub">boundary moved to keep a name whole</div></div>';
    html += '</div>';

    if (!chunks.length) {
      root.innerHTML = html + emptyState('No chunks', 'This document produced no chunks.');
      return;
    }

    html += '<div class="chunk-list">';
    chunks.forEach((chunk) => {
      const tokens = safeArray(chunk.tokens);
      html += '<article class="chunk">';
      html += '<header class="chunk-head">';
      html += '<span class="chunk-seq">seq ' +
        esc(chunk.seq === null || chunk.seq === undefined ? '?' : chunk.seq) + '</span>';
      html += copyId(chunk.chunk_id);
      if (chunk.extended === true) {
        html += '<span class="badge badge-extended" title="the cut was moved so a name is never split in half">extended cut</span>';
      }
      html += '<span class="chunk-tokens-count">' +
        fmtInt(chunk.token_count !== undefined && chunk.token_count !== null ? chunk.token_count : tokens.length) +
        ' tokens</span>';
      html += '</header>';
      html += '<div class="chunk-body">';
      if (tokens.length) {
        tokens.forEach((token, index) => {
          html += '<span class="tok ' + (index % 2 === 0 ? 'tok-a' : 'tok-b') + '">' + esc(token) + '</span>';
        });
      } else {
        html += esc(chunk.text || '');
      }
      html += '</div>';
      html += '</article>';
    });
    html += '</div>';

    html += ctaHtml('batch', 'Continue to Batch',
      'Build the batch and run detection on these ' + fmtInt(chunks.length) + ' chunks.');

    root.innerHTML = html;
  }

  // ------------------------------------------------------------ view: batch

  function compositionHtml(composition) {
    const comp = composition && typeof composition === 'object' ? composition : {};
    const values = KINDS.map((kind) => ({ kind: kind, count: num(comp[kind], 0) }));
    const total = values.reduce((sum, item) => sum + item.count, 0);

    const why = {
      real: 'your document, one chunk per fragment',
      honeytoken: 'planted fake names we know the answer for — they measure recall',
      chaff: 'decoys the provider cannot tell from real ones — they dilute any retained copy',
      canary: 'unique fabricated facts — probe them later to detect training misuse',
    };

    let html = '<div class="card">';
    html += '<div class="row-between" style="margin-bottom:10px"><h3 class="card-title" style="margin:0">Batch composition</h3>';
    html += '<span class="mono dim">' + fmtInt(total) + ' fragments</span></div>';

    if (total <= 0) {
      html += '<div class="empty-hint">No composition data returned.</div></div>';
      return html;
    }

    html += '<div class="comp-bar">';
    values.forEach((item) => {
      if (item.count <= 0) return;
      const pct = (item.count / total) * 100;
      html += '<div class="comp-seg comp-seg-' + item.kind + '" style="width:' + pct.toFixed(3) + '%" ' +
        'title="' + esc(item.kind) + ': ' + item.count + ' (' + pct.toFixed(1) + '%)">' +
        (pct >= 8 ? esc(String(item.count)) : '') + '</div>';
    });
    html += '</div>';

    html += '<div class="comp-legend">';
    values.forEach((item) => {
      const pct = total > 0 ? (item.count / total) * 100 : 0;
      html += '<span class="legend-item"><span class="legend-dot legend-dot-' + item.kind + '"></span>';
      html += '<span><span class="legend-name">' + esc(item.kind) + '</span>' +
        (item.kind === 'real' ? info('real') : info(item.kind)) +
        ' <span class="legend-n">' + fmtInt(item.count) + '</span> ' +
        '<span class="dim">(' + pct.toFixed(1) + '%)</span>' +
        '<span class="legend-why" style="display:block">' + esc(why[item.kind] || '') + '</span></span></span>';
    });
    html += '</div>';
    html += '</div>';
    return html;
  }

  function runPlanHtml() {
    const settings = store.settings || {};
    const chunkCount = store.doc ? safeArray(store.doc.chunks).length : 0;
    const llmOn = settings.llm_enabled !== false;

    let html = '<div class="run-plan chips">';
    html += '<span class="chip"><span class="chip-k">real chunks</span><span class="chip-v">' + fmtInt(chunkCount) + '</span></span>';
    html += '<span class="chip"><span class="chip-k">provider model</span><span class="chip-v">' + esc(settings.model || 'unset') + '</span></span>';
    html += '<span class="chip"><span class="chip-k">effort</span><span class="chip-v">' + esc(settings.effort || '—') + '</span></span>';
    html += '<span class="chip"><span class="chip-k">synthetic prose</span><span class="chip-v">' +
      (llmOn ? esc((settings.llm_model || 'local model') + ' @ ' + (settings.llm_base_url || 'local')) : 'deterministic templates') +
      '</span></span>';
    html += '<span class="chip"><span class="chip-k">chaff_ratio</span><span class="chip-v">' + esc(settings.chaff_ratio) + '</span></span>';
    html += '<span class="chip"><span class="chip-k">honeytoken_rate</span><span class="chip-v">' + esc(settings.honeytoken_rate) + '</span></span>';
    html += '<span class="chip"><span class="chip-k">canaries</span><span class="chip-v">' + esc(settings.canaries_per_batch) + '</span></span>';
    html += '</div>';
    return html;
  }

  function phaseLabel(phase) {
    switch (phase) {
      case 'planning': return 'Planning the batch';
      case 'synthetics': return 'Generating synthetic fragments';
      case 'canaries': return 'Seeding canaries';
      case 'submitted': return 'Waiting for the provider';
      case 'done': return 'Detection complete';
      case 'starting': return 'Starting';
      default: return phase ? String(phase) : 'Working';
    }
  }

  function runCountText() {
    const run = store.run;
    if (!run) return '';
    if (run.mode === 'synthetics' && run.total > 0) {
      return run.cur + ' / ' + run.total + ' synthetic fragments generated';
    }
    if (run.mode === 'results' && run.total > 0) {
      return run.cur + ' / ' + run.total + ' provider calls returned';
    }
    if (run.done) return 'finished';
    return 'preparing';
  }

  function runElapsedText() {
    const run = store.run;
    if (!run || !run.startedAt) return '';
    const end = run.endedAt || Date.now();
    const seconds = Math.max(0, Math.round((end - run.startedAt) / 1000));
    const mins = Math.floor(seconds / 60);
    const rest = seconds % 60;
    return (mins > 0 ? mins + 'm ' : '') + rest + 's elapsed';
  }

  function runLogHtml() {
    const run = store.run;
    if (!run) return '';
    return run.log.map((line) =>
      '<span class="log-line is-' + esc(line.cls) + '">' + esc(line.text) + '</span>').join('');
  }

  function runPanelHtml() {
    const run = store.run;
    if (!run) return '';

    const total = num(run.total, 0);
    const pct = total > 0 ? Math.max(0, Math.min(100, (num(run.cur, 0) / total) * 100)) : 0;
    const indeterminate = run.active && total <= 0;

    let html = '<div class="run-panel">';
    html += '<div class="run-head">';
    html += '<div class="run-phase">';
    if (run.active) html += '<span class="spinner"></span>';
    html += '<span id="run-phase-label">' + esc(phaseLabel(run.phase)) + '</span>';
    html += '<span class="run-phase-detail" id="run-phase-detail">' + esc(run.detail || '') + '</span>';
    html += '</div>';
    html += '<span class="run-elapsed" id="run-elapsed">' + esc(runElapsedText()) + '</span>';
    html += '</div>';

    html += '<div class="bar' + (indeterminate ? ' is-indeterminate' : '') + '" id="run-bar" role="progressbar"' +
      ' aria-valuemin="0" aria-valuemax="100" aria-valuenow="' + Math.round(pct) + '">';
    html += '<div class="bar-fill" id="run-bar-fill" style="width:' + pct.toFixed(1) + '%"></div></div>';
    html += '<div class="bar-caption"><span id="run-count">' + esc(runCountText()) + '</span>';
    html += '<span>' + (run.error ? 'stopped' : (run.done ? 'complete' : '')) + '</span></div>';

    html += '<div class="run-log" id="run-log" role="log" aria-label="Detection activity log">' +
      runLogHtml() + '</div>';

    if (run.error) {
      html += '<div class="alert alert-error" style="margin:14px 0 0"><span class="alert-title">Detection failed</span> — ' +
        esc(run.error) + '</div>';
    }

    if (!run.active) {
      html += '<div class="btn-row" style="margin-top:14px">';
      html += '<button type="button" class="btn" id="run-detect">Run detection again</button>';
      html += '</div>';
    }

    html += '</div>';
    return html;
  }

  function renderBatch(root) {
    const hasDoc = Boolean(store.doc);
    const settings = store.settings || {};

    let html = '';
    html += '<div class="page-head">';
    html += '<h2 class="page-title">Batch</h2>';
    html += '<p class="page-lead">Step 3 of 5 — real chunks are shuffled together with synthetic fragments (' +
      term('honeytokens', 'honeytoken') + ', ' + term('chaff', 'chaff') + ', ' + term('canaries', 'canary') +
      ') and each fragment is sent on its own, so nothing recognisable leaves this machine in one piece.</p>';
    html += '</div>';

    html += hintHtml('batch');
    html += alertHtml('batch');

    if (!hasDoc) {
      root.innerHTML = html + emptyState('No active document',
        'Upload a document first.',
        '<a class="btn btn-primary" href="#document">Go to Document</a>');
      return;
    }

    if (store.run) {
      html += runPanelHtml();
    } else {
      html += '<div class="card">';
      html += '<h3 class="card-title">Run detection</h3>';
      html += '<p class="muted" style="margin:0 0 4px">Builds the batch from ' +
        fmtInt(safeArray(store.doc.chunks).length) + ' real chunks plus synthetic decoys, then makes one provider call per fragment. ' +
        'Progress is streamed live below — you will see every fragment as it is generated and every answer as it returns.</p>';
      html += runPlanHtml();
      if (settings.has_api_key === false) {
        html += '<div class="alert alert-warn" style="margin:12px 0 0"><span class="alert-title">No API key</span> — ' +
          'detection needs an Anthropic key. <a href="#settings">Open Settings</a> to add one.</div>';
      }
      html += '<div class="btn-row" style="margin-top:14px">';
      html += '<button type="button" class="btn btn-primary" id="run-detect"' +
        (store.busy.detect ? ' disabled' : '') + '>Run detection</button>';
      html += '</div>';
      html += '</div>';
    }

    if (store.detect) {
      const detect = store.detect;
      const requests = safeArray(detect.requests);
      const results = safeArray(detect.results);
      const provider = store.batchView === 'provider';

      const resultById = new Map();
      results.forEach((result) => {
        if (result && result.custom_id !== undefined) resultById.set(String(result.custom_id), result);
      });

      html += '<div class="chips" style="margin:22px 0 12px">';
      html += '<span class="chip"><span class="chip-k">batch_id</span><span class="chip-v">' + copyId(detect.batch_id) + '</span></span>';
      if (detect.entities_stored !== undefined && detect.entities_stored !== null) {
        html += '<span class="chip"><span class="chip-k">names stored</span><span class="chip-v">' +
          fmtInt(detect.entities_stored) + '</span></span>';
      }
      const recall = detect.honeytoken_recall;
      if (recall && typeof recall === 'object') {
        html += '<span class="chip"><span class="chip-k">honeytoken recall</span><span class="chip-v">' +
          fmtPct(recall.recall) + ' (' + fmtInt(recall.found) + '/' + fmtInt(recall.planted) + ')</span></span>';
      }
      html += '</div>';

      html += compositionHtml(detect.composition);

      html += '<div class="row-between" style="margin:22px 0 0;align-items:center">';
      html += '<h3 class="section-title" style="margin:0">Submitted fragments (' +
        fmtInt(requests.length) + ', in submission order)</h3>';
      html += '<div class="seg-toggle" role="group" aria-label="Fragment view">';
      html += '<button type="button" class="seg-btn' + (provider ? '' : ' is-active') +
        '" data-batchview="local" aria-pressed="' + (provider ? 'false' : 'true') + '">Local view</button>';
      html += '<button type="button" class="seg-btn' + (provider ? ' is-active' : '') +
        '" data-batchview="provider" aria-pressed="' + (provider ? 'true' : 'false') + '">Provider view' + info('provider') + '</button>';
      html += '</div>';
      html += '</div>';

      if (provider) {
        html += '<div class="provider-caption">This is everything the provider can see: shuffled fragments under opaque IDs, ' +
          'roughly half of them synthetic, with no ordering and no labels.</div>';
      } else {
        html += '<div class="provider-caption">Local view — what only you know: which fragment is real, where it came from, and how the provider answered.</div>';
      }

      if (!requests.length) {
        html += emptyState('No fragments', 'The batch contained no fragments.');
      } else {
        html += '<div class="frag-list' + (provider ? ' is-provider' : '') + '">';
        requests.forEach((req) => {
          const kind = String(req.kind || 'unknown');
          const customId = String(req.custom_id === undefined ? '' : req.custom_id);
          if (provider) {
            html += '<div class="frag">';
            html += '<div>' + copyId(req.custom_id) + '</div>';
            html += '<div class="frag-text">' + esc(req.text || '') + '</div>';
            html += '</div>';
          } else {
            const result = resultById.get(customId) || null;
            html += '<div class="frag frag-kind-' + esc(kindClass(kind)) + '" data-frag-id="' + esc(customId) + '">';
            html += '<div class="frag-meta">' + kindBadge(kind);
            if (req.seq !== null && req.seq !== undefined) {
              html += '<span class="frag-seq">seq ' + esc(req.seq) + '</span>';
            }
            html += '<span class="frag-status">' +
              (result ? statusPill(result.status) : (store.run && store.run.active ? statusPill('pending') : '')) +
              '</span>';
            html += '</div>';
            html += '<div>' + copyId(req.custom_id) + '</div>';
            html += '<div class="frag-text">' + esc(req.text || '') + '</div>';
            html += '</div>';
          }
        });
        html += '</div>';
      }
    }

    if (hasRun()) {
      html += ctaHtml('llm', 'Continue to LLM I/O',
        'Inspect the exact request and response for every fragment, or jump straight to the <a href="#redaction">redacted document</a>.');
    }

    root.innerHTML = html;
    wireBatch(root);
  }

  function wireBatch(root) {
    const button = root.querySelector('#run-detect');
    if (button) button.addEventListener('click', runDetect);

    root.querySelectorAll('[data-batchview]').forEach((node) => {
      node.addEventListener('click', () => {
        store.batchView = node.getAttribute('data-batchview') === 'provider' ? 'provider' : 'local';
        render();
      });
    });

    const log = root.querySelector('#run-log');
    if (log) log.scrollTop = log.scrollHeight;
  }

  // ------------------------------------------------------- detection stream

  function logLine(cls, text) {
    if (!store.run) return;
    store.run.log.push({ cls: cls, text: text });
    if (store.run.log.length > LOG_LIMIT) store.run.log.shift();

    const host = document.getElementById('run-log');
    if (!host) return;
    const node = document.createElement('span');
    node.className = 'log-line is-' + cls;
    node.textContent = text;
    host.appendChild(node);
    host.scrollTop = host.scrollHeight;
  }

  function updateRunDom() {
    const run = store.run;
    if (!run) return;

    const phaseLabelNode = document.getElementById('run-phase-label');
    if (phaseLabelNode) phaseLabelNode.textContent = phaseLabel(run.phase);

    const detailNode = document.getElementById('run-phase-detail');
    if (detailNode) detailNode.textContent = run.detail || '';

    const total = num(run.total, 0);
    const pct = total > 0 ? Math.max(0, Math.min(100, (num(run.cur, 0) / total) * 100)) : 0;

    const bar = document.getElementById('run-bar');
    if (bar) {
      bar.classList.toggle('is-indeterminate', Boolean(run.active) && total <= 0);
      bar.setAttribute('aria-valuenow', String(Math.round(pct)));
    }
    const fill = document.getElementById('run-bar-fill');
    if (fill && total > 0) fill.style.width = pct.toFixed(1) + '%';

    const count = document.getElementById('run-count');
    if (count) count.textContent = runCountText();

    const elapsed = document.getElementById('run-elapsed');
    if (elapsed) elapsed.textContent = runElapsedText();
  }

  function markFragmentResult(result) {
    if (!result || result.custom_id === undefined) return;
    const node = document.querySelector('[data-frag-id="' + String(result.custom_id).replace(/"/g, '') + '"]');
    if (!node) return;
    const slot = node.querySelector('.frag-status');
    if (slot) slot.innerHTML = statusPill(result.status);
  }

  function startRunTimer() {
    stopRunTimer();
    runTimer = window.setInterval(() => {
      const elapsed = document.getElementById('run-elapsed');
      if (elapsed) elapsed.textContent = runElapsedText();
    }, 1000);
  }

  function stopRunTimer() {
    if (runTimer !== null) {
      window.clearInterval(runTimer);
      runTimer = null;
    }
  }

  function handleEvent(event) {
    const run = store.run;
    if (!run || !event || typeof event !== 'object') return;

    switch (event.type) {
      case 'phase': {
        run.phase = String(event.phase || '');
        run.detail = event.detail ? String(event.detail) : '';
        if (run.phase === 'synthetics') {
          run.mode = 'synthetics';
          run.cur = 0;
          run.total = 0;
        }
        logLine('phase', run.phase + (run.detail ? ' - ' + run.detail : ''));
        updateRunDom();
        break;
      }

      case 'synthetic': {
        run.mode = 'synthetics';
        run.cur = num(event.index, run.cur);
        run.total = num(event.total, run.total);
        const kind = String(event.kind || 'synthetic');
        logLine(kindClass(kind) === 'neutral' ? 'info' : kindClass(kind),
          '  ' + kind + ' ' + num(event.index, 0) + '/' + num(event.total, 0) +
          ' generated (' + shortId(event.fragment_id) + ')');
        updateRunDom();
        break;
      }

      case 'submitted': {
        const requests = safeArray(event.requests);
        store.detect = {
          batch_id: event.batch_id,
          composition: event.composition,
          payload_template: event.payload_template,
          requests: requests,
          results: [],
        };
        store.detectDocId = store.doc ? store.doc.doc_id : null;
        run.phase = 'submitted';
        run.detail = '';
        run.mode = 'results';
        run.cur = 0;
        run.total = requests.length;
        logLine('info', 'submitted - ' + requests.length + ' fragments, batch ' +
          (event.batch_id === undefined || event.batch_id === null ? '?' : event.batch_id));
        render();
        break;
      }

      case 'result': {
        run.mode = 'results';
        run.cur = num(event.index, run.cur);
        run.total = num(event.total, run.total);

        const result = {
          custom_id: event.custom_id,
          kind: event.kind,
          status: event.status,
          entities: safeArray(event.entities),
          raw_text: event.raw_text,
          latency_ms: event.latency_ms,
          detail: event.detail,
        };
        if (store.detect) {
          if (!Array.isArray(store.detect.results)) store.detect.results = [];
          store.detect.results.push(result);
        }

        const status = String(event.status || 'unknown').toLowerCase();
        const cls = status === 'error' ? 'error' : (status === 'refusal' ? 'refusal' : 'ok');
        let line = '  ' + num(event.index, 0) + '/' + num(event.total, 0) + ' ' +
          String(event.kind || '?') + ' ' + shortId(event.custom_id) + ' - ' + status +
          ', ' + result.entities.length + ' entities, ' + fmtSec(event.latency_ms);
        if (event.detail) line += ' - ' + String(event.detail);
        logLine(cls, line);

        markFragmentResult(result);
        updateRunDom();
        break;
      }

      case 'done': {
        store.detect = {
          batch_id: event.batch_id,
          composition: event.composition,
          payload_template: event.payload_template,
          requests: safeArray(event.requests),
          results: safeArray(event.results),
          honeytoken_recall: event.honeytoken_recall === undefined ? null : event.honeytoken_recall,
          entities_stored: event.entities_stored,
        };
        store.detectDocId = store.doc ? store.doc.doc_id : null;
        // The stored entities changed, so everything derived from them is stale.
        store.redaction = null;
        store.redactionDocId = null;
        store.report = null;

        run.phase = 'done';
        run.detail = '';
        run.done = true;
        run.active = false;
        run.endedAt = Date.now();
        run.cur = safeArray(event.results).length;
        run.total = safeArray(event.results).length || run.total;
        store.busy.detect = false;
        stopRunTimer();

        const results = safeArray(event.results);
        const bad = results.filter((r) => String(r.status || '').toLowerCase() !== 'ok').length;
        let line = 'done - ' + results.length + ' results, ' +
          fmtInt(event.entities_stored) + ' names stored';
        const recall = event.honeytoken_recall;
        if (recall && typeof recall === 'object') {
          line += ', honeytoken recall ' + fmtPct(recall.recall) +
            ' (' + num(recall.found, 0) + '/' + num(recall.planted, 0) + ')';
        }
        logLine('ok', line);
        if (bad) logLine('warn', '  ' + bad + ' of ' + results.length + ' fragments were not ok');
        toast(bad ? 'info' : 'ok', 'Detection finished — ' + results.length + ' results' +
          (bad ? ', ' + bad + ' not ok' : ''));
        render();
        break;
      }

      case 'error': {
        const detail = event.detail ? String(event.detail) : 'The server reported an error.';
        run.error = detail;
        run.active = false;
        run.endedAt = Date.now();
        store.busy.detect = false;
        stopRunTimer();
        logLine('error', 'error - ' + detail);
        setAlert('batch', 'error', detail);
        toast('error', detail);
        render();
        break;
      }

      default: {
        logLine('warn', 'unknown event: ' + esc(event.type === undefined ? '(no type)' : event.type));
        break;
      }
    }
  }

  function handleLine(line) {
    const trimmed = String(line || '').replace(/\r$/, '').trim();
    if (!trimmed) return;
    let event;
    try {
      event = JSON.parse(trimmed);
    } catch (err) {
      logLine('warn', 'unreadable line from the server: ' + trimmed.slice(0, 160));
      return;
    }
    handleEvent(event);
  }

  async function consumeStream(reader) {
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      let index = buffer.indexOf('\n');
      while (index >= 0) {
        handleLine(buffer.slice(0, index));
        buffer = buffer.slice(index + 1);
        index = buffer.indexOf('\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim()) handleLine(buffer);
  }

  function abortRun(message) {
    if (store.run) {
      store.run.error = message;
      store.run.active = false;
      store.run.endedAt = Date.now();
      logLine('error', 'error - ' + message);
    }
    store.busy.detect = false;
    stopRunTimer();
    setAlert('batch', 'error', message);
    toast('error', message);
    render();
  }

  async function runDetect() {
    if (!store.doc || store.busy.detect) return;

    clearAlert('batch');
    store.busy.detect = true;
    store.detect = null;
    store.detectDocId = null;
    store.redaction = null;
    store.redactionDocId = null;
    store.report = null;
    store.run = {
      active: true,
      done: false,
      error: null,
      phase: 'starting',
      detail: 'contacting the server',
      mode: '',
      cur: 0,
      total: 0,
      log: [],
      startedAt: Date.now(),
      endedAt: null,
    };
    logLine('info', 'run started - document ' + store.doc.doc_id);
    startRunTimer();
    render();

    let res;
    try {
      res = await fetch('/api/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_id: store.doc.doc_id }),
      });
    } catch (err) {
      abortRun('Network error contacting /api/detect (' +
        (err && err.message ? err.message : 'unknown') + ')');
      return;
    }

    if (!res.ok) {
      let raw = '';
      try {
        raw = await res.text();
      } catch (err) {
        raw = '';
      }
      abortRun(errorDetail(raw, res, '/api/detect'));
      return;
    }

    try {
      if (res.body && typeof res.body.getReader === 'function') {
        await consumeStream(res.body.getReader());
      } else {
        // No streaming support in this browser: read the whole body, then split.
        const raw = await res.text();
        raw.split('\n').forEach(handleLine);
      }
    } catch (err) {
      abortRun('The detection stream broke (' +
        (err && err.message ? err.message : 'unknown') + ')');
      return;
    }

    stopRunTimer();
    store.busy.detect = false;
    if (store.run) {
      store.run.active = false;
      if (!store.run.endedAt) store.run.endedAt = Date.now();
      if (!store.run.done && !store.run.error) {
        store.run.error = 'The stream ended before a final result arrived.';
        logLine('error', 'error - stream ended before a final result arrived');
        setAlert('batch', 'error', store.run.error);
      }
    }
    render();
  }

  // --------------------------------------------------------- view: llm i/o

  function entityHtml(entity) {
    const type = String(entity && entity.type ? entity.type : 'unknown');
    const upper = type.toUpperCase();
    const cls = upper === 'PERSON' ? 'person' : (upper === 'COMPANY' ? 'company' : 'other');
    return '<span class="ent"><span class="ent-text">' + esc(entity && entity.text ? entity.text : '') +
      '</span><span class="ent-type ent-type-' + cls + '">' + esc(upper) + '</span></span>';
  }

  function renderLlm(root) {
    let html = '';
    html += '<div class="page-head">';
    html += '<h2 class="page-title">LLM I/O</h2>';
    html += '<p class="page-lead">Step 4 of 5 — exactly what was sent and what came back for this session\'s run: the shared payload template, then one row per fragment with status, latency and the names it returned.</p>';
    html += '</div>';

    html += hintHtml('llm');
    html += alertHtml('llm');

    if (!store.detect) {
      const priorRun = Boolean(store.redaction && store.redaction.has_detection);
      if (priorRun) {
        root.innerHTML = html + emptyState('Nothing to inspect in this session',
          'This document already has detection results stored, but the request and response detail is only kept for a run made in this browser session. Run detection again to inspect the exchange.',
          '<a class="btn btn-primary" href="#batch">Go to Batch</a>' +
          '<a class="btn" href="#redaction">See the redacted document</a>');
      } else {
        root.innerHTML = html + emptyState('No detection run yet',
          'Run detection first — the request and response detail appears here as soon as the batch is submitted.',
          '<a class="btn btn-primary" href="#batch">Go to Batch</a>');
      }
      return;
    }

    const detect = store.detect;
    const requests = safeArray(detect.requests);
    const results = safeArray(detect.results);

    const byId = new Map();
    results.forEach((result) => {
      if (result && result.custom_id !== undefined) byId.set(String(result.custom_id), result);
    });

    // Any result without a matching request still deserves a row.
    const seen = new Set();
    const rows = requests.map((req) => {
      const id = String(req.custom_id);
      seen.add(id);
      return { request: req, result: byId.get(id) || null };
    });
    results.forEach((result) => {
      const id = String(result && result.custom_id);
      if (!seen.has(id)) rows.push({ request: null, result: result });
    });

    const latencies = results
      .map((result) => Number(result.latency_ms))
      .filter((value) => Number.isFinite(value));
    const totalLatency = latencies.reduce((sum, value) => sum + value, 0);
    const avgLatency = latencies.length ? totalLatency / latencies.length : NaN;
    const counts = { ok: 0, refusal: 0, error: 0, other: 0 };
    results.forEach((result) => {
      const status = String(result.status || '').toLowerCase();
      if (counts[status] === undefined) counts.other += 1;
      else counts[status] += 1;
    });

    html += '<div class="stat-row">';
    html += '<div class="stat"><div class="stat-k">Fragments</div><div class="stat-v">' + fmtInt(rows.length) + '</div></div>';
    html += '<div class="stat"><div class="stat-k">ok</div><div class="stat-v stat-v-ok">' + fmtInt(counts.ok) + '</div></div>';
    html += '<div class="stat"><div class="stat-k">refusal</div><div class="stat-v' +
      (counts.refusal ? ' stat-v-warn' : '') + '">' + fmtInt(counts.refusal) + '</div></div>';
    html += '<div class="stat"><div class="stat-k">error</div><div class="stat-v' +
      (counts.error ? ' stat-v-error' : '') + '">' + fmtInt(counts.error) + '</div></div>';
    html += '<div class="stat"><div class="stat-k">Total latency</div><div class="stat-v">' + fmtMs(totalLatency) + '</div></div>';
    html += '<div class="stat"><div class="stat-k">Avg latency</div><div class="stat-v">' +
      (latencies.length ? fmtMs(avgLatency) : '—') + '</div></div>';

    const recall = detect.honeytoken_recall;
    if (recall && typeof recall === 'object') {
      html += '<div class="stat"><div class="stat-k">Honeytoken recall' + info('recall') + '</div><div class="stat-v">' +
        fmtPct(recall.recall) + '</div><div class="stat-sub">' +
        fmtInt(recall.found) + ' / ' + fmtInt(recall.planted) + ' planted names found</div></div>';
    }
    html += '</div>';

    const template = detect.payload_template && typeof detect.payload_template === 'object'
      ? detect.payload_template : null;
    html += '<details class="payload" open><summary><span class="io-caret">&#9662;</span>' +
      '<strong>Shared payload template</strong><span class="dim" style="font-size:12.5px">' +
      '— identical for every fragment; only the text differs</span></summary>';
    html += '<div class="payload-body">';
    if (!template) {
      html += '<div class="empty-hint">No payload template returned.</div>';
    } else {
      html += '<div class="chips" style="margin-bottom:12px">';
      if (template.model) {
        html += '<span class="chip"><span class="chip-k">model</span><span class="chip-v">' + esc(template.model) + '</span></span>';
      }
      if (template.max_tokens !== undefined && template.max_tokens !== null) {
        html += '<span class="chip"><span class="chip-k">max_tokens</span><span class="chip-v">' + fmtInt(template.max_tokens) + '</span></span>';
      }
      if (store.settings && store.settings.effort) {
        html += '<span class="chip"><span class="chip-k">effort</span><span class="chip-v">' +
          esc(store.settings.effort) + '</span>' + info('effort') + '</span>';
      }
      html += '</div>';
      html += '<h4>System prompt</h4>';
      html += '<pre class="code">' +
        esc(typeof template.system === 'string' ? template.system : pretty(template.system)) + '</pre>';
      html += '<h4>Output schema</h4>';
      html += '<pre class="code code-nowrap">' + esc(pretty(template.output_config)) + '</pre>';
    }
    html += '</div></details>';

    html += '<h3 class="section-title">Per-fragment exchange</h3>';
    if (!rows.length) {
      html += emptyState('No fragments', 'The run produced no request rows.');
    } else {
      html += '<div class="io-list">';
      rows.forEach((row) => {
        const req = row.request;
        const result = row.result;
        const status = String((result && result.status) || 'pending').toLowerCase();
        const kind = String((req && req.kind) || (result && result.kind) || 'unknown');
        const customId = (req && req.custom_id) || (result && result.custom_id) || '';
        const rowCls = status === 'refusal' ? ' is-refusal' : (status === 'error' ? ' is-error' : '');

        html += '<details class="io-row' + rowCls + '">';
        html += '<summary>';
        html += '<span class="io-caret">&#9656;</span>';
        html += '<span class="mono dim">' + esc(customId) + '</span>';
        html += kindBadge(kind);
        html += statusPill(status);
        html += '<span class="io-latency">' +
          (result && result.latency_ms !== undefined && result.latency_ms !== null
            ? fmtInt(result.latency_ms) + ' ms' : '—') + '</span>';
        html += '</summary>';
        html += '<div class="io-detail">';

        html += '<h4>Request text</h4>';
        html += '<pre class="code">' + esc(req ? (req.text || '') : '(no matching request)') + '</pre>';

        if (result) {
          const entities = safeArray(result.entities);
          html += '<h4>Names returned (' + fmtInt(entities.length) + ')</h4>';
          if (entities.length) {
            html += '<div class="ent-list">' + entities.map(entityHtml).join('') + '</div>';
          } else {
            html += '<div class="empty-hint" style="text-align:left;margin:0">No names returned for this fragment.</div>';
          }
          if (result.raw_text) {
            html += '<h4>Raw response</h4><pre class="code">' + esc(result.raw_text) + '</pre>';
          }
          if (result.detail) {
            html += '<h4>Detail</h4><div class="alert alert-' + (status === 'error' ? 'error' : 'warn') +
              '" style="margin:0">' + esc(result.detail) + '</div>';
          }
        } else {
          html += '<h4>Response</h4><div class="empty-hint" style="text-align:left;margin:0">' +
            (store.run && store.run.active ? 'Still waiting for this fragment.' : 'No result returned for this custom_id.') +
            '</div>';
        }

        html += '</div></details>';
      });
      html += '</div>';
    }

    if (hasRun()) {
      html += ctaHtml('redaction', 'Continue to Redaction',
        'See the document reassembled with every detected name replaced.');
    }

    root.innerHTML = html;
  }

  // ------------------------------------------------------ view: redaction

  function highlightRedacted(text) {
    // esc() leaves the literal ** markers intact, so highlight after escaping.
    return esc(text).replace(/\*\*(PERSON|COMPANY)\*\*/g,
      (match, label) => '<mark class="redacted">**' + label + '**</mark>');
  }

  function renderRedaction(root) {
    let html = '';
    html += '<div class="page-head"><div class="row-between">';
    html += '<div><h2 class="page-title">Redaction</h2>';
    html += '<p class="page-lead">Step 5 of 5 — the document reassembled from its chunks on this machine, with every name the detector found replaced by a placeholder.</p></div>';
    html += '<button type="button" class="btn" id="refresh-redaction"' +
      (store.busy.redaction || !store.doc ? ' disabled' : '') + '>' +
      (store.busy.redaction ? '<span class="spinner"></span>Loading' : 'Refresh') + '</button>';
    html += '</div></div>';

    html += hintHtml('redaction');
    html += alertHtml('redaction');

    if (!store.doc) {
      root.innerHTML = html + emptyState('No active document',
        'Upload a document first.',
        '<a class="btn btn-primary" href="#document">Go to Document</a>');
      return;
    }

    if (!store.redaction) {
      root.innerHTML = html + emptyState(
        store.busy.redaction ? 'Loading' : 'Nothing loaded yet',
        store.busy.redaction
          ? 'Fetching the reassembled document.'
          : 'Press Refresh to fetch the reassembled document.',
        store.busy.redaction ? '' : '<button type="button" class="btn btn-primary" id="refresh-redaction-2">Refresh</button>');
      wireRedaction(root);
      return;
    }

    const data = store.redaction;
    const entities = safeArray(data.entities);
    const hasDetection = data.has_detection === undefined
      ? (hasRun() || entities.length > 0)
      : data.has_detection === true;

    // Stats about what the batch actually covered, when the backend reports them.
    if (data.chunk_count !== undefined || data.chunks_submitted !== undefined) {
      html += '<div class="stat-row">';
      if (data.chunk_count !== undefined) {
        html += '<div class="stat"><div class="stat-k">Chunks in document</div><div class="stat-v">' +
          fmtInt(data.chunk_count) + '</div></div>';
      }
      if (data.chunks_submitted !== undefined) {
        html += '<div class="stat"><div class="stat-k">Chunks submitted</div><div class="stat-v">' +
          fmtInt(data.chunks_submitted) + '</div></div>';
      }
      html += '<div class="stat"><div class="stat-k">Names found</div><div class="stat-v">' +
        fmtInt(entities.length) + '</div></div>';
      html += '</div>';
    }

    if (!hasDetection) {
      html += emptyState('No detection has run for this document',
        'Nothing has been sent to the provider yet, so no names are known and there is nothing to redact. ' +
        'Run detection on the Batch step, then come back here.',
        '<a class="btn btn-primary" href="#batch">Go to Batch</a>');
      root.innerHTML = html;
      wireRedaction(root);
      return;
    }

    if (!entities.length) {
      html += '<div class="alert alert-info"><span class="alert-title">Nothing to redact</span> — ' +
        'detection ran but found no names in this document.</div>';
      html += '<div class="pane"><div class="pane-head">Document (unchanged)</div><pre class="code code-tall">' +
        esc(data.original || '(empty)') + '</pre></div>';
      html += ctaHtml('report', 'Continue to Report',
        'Check honeytoken recall to see whether the detector was actually looking properly.');
      root.innerHTML = html;
      wireRedaction(root);
      return;
    }

    html += '<div class="diff-grid">';
    html += '<div class="pane"><div class="pane-head">Original</div><pre class="code code-tall">' +
      esc(data.original || '(empty)') + '</pre></div>';
    html += '<div class="pane"><div class="pane-head">Redacted</div><pre class="code code-tall">' +
      highlightRedacted(data.redacted || '(empty)') + '</pre></div>';
    html += '</div>';

    html += '<h3 class="section-title">Names replaced (' + fmtInt(entities.length) + ')</h3>';
    html += '<div class="card"><div class="ent-list">' + entities.map(entityHtml).join('') + '</div></div>';

    html += ctaHtml('report', 'Continue to Report',
      'How many planted names came back, and whether any canary has been tripped.');

    root.innerHTML = html;
    wireRedaction(root);
  }

  function wireRedaction(root) {
    ['#refresh-redaction', '#refresh-redaction-2'].forEach((selector) => {
      const button = root.querySelector(selector);
      if (button) button.addEventListener('click', () => fetchRedaction());
    });
  }

  async function fetchRedaction() {
    if (!store.doc || store.busy.redaction) return;
    clearAlert('redaction');
    store.busy.redaction = true;
    render();
    try {
      const data = await api('/api/documents/' + encodeURIComponent(store.doc.doc_id) + '/redaction');
      store.redaction = data;
      store.redactionDocId = store.doc.doc_id;
    } catch (err) {
      store.redaction = null;
      store.redactionDocId = null;
      fail('redaction', err);
    } finally {
      store.busy.redaction = false;
      render();
    }
  }

  /**
   * Quietly learn whether the active document already has detection results,
   * so the stepper and the Redaction view can tell the truth after a reload.
   */
  async function prefetchRedaction() {
    if (!store.doc || store.busy.redaction) return;
    const docId = store.doc.doc_id;
    try {
      const data = await api('/api/documents/' + encodeURIComponent(docId) + '/redaction');
      if (store.doc && store.doc.doc_id === docId) {
        store.redaction = data;
        store.redactionDocId = docId;
        render();
      }
    } catch (err) {
      /* silent: the Redaction view will report the failure if the user goes there */
    }
  }

  // ---------------------------------------------------------- view: report

  function recallBar(fraction) {
    const n = Number(fraction);
    const pct = Number.isFinite(n) ? Math.max(0, Math.min(1, n)) * 100 : 0;
    const cls = !Number.isFinite(n) ? '' : (n >= 0.9 ? ' is-high' : (n < 0.5 ? ' is-low' : ''));
    return '<span class="recall-bar" aria-hidden="true"><span class="recall-fill' + cls +
      '" style="width:' + pct.toFixed(1) + '%"></span></span>';
  }

  function renderReport(root) {
    let html = '';
    html += '<div class="page-head"><div class="row-between">';
    html += '<div><h2 class="page-title">Report</h2>';
    html += '<p class="page-lead">The evidence that the defences work: how many synthetic fragments exist, what share of planted names the detector found (' +
      term('recall', 'recall') + '), and whether any ' + term('canary', 'canary') + ' has ever been repeated back to us.</p></div>';
    html += '<div class="btn-row">';
    html += '<button type="button" class="btn" id="refresh-report"' + (store.busy.report ? ' disabled' : '') + '>' +
      (store.busy.report ? '<span class="spinner"></span>Loading' : 'Refresh') + '</button>';
    html += '<button type="button" class="btn btn-primary" id="run-probe"' + (store.busy.probe ? ' disabled' : '') + '>' +
      (store.busy.probe ? '<span class="spinner"></span>Probing' : 'Run canary probe') + '</button>' + info('probe');
    html += '</div></div></div>';

    html += hintHtml('report');
    html += alertHtml('report');

    if (!store.report) {
      root.innerHTML = html + emptyState(
        store.busy.report ? 'Loading' : 'No report loaded',
        store.busy.report ? 'Fetching synthetic-fragment statistics.' : 'Press Refresh to load the synthetic report.');
      wireReport(root);
      return;
    }

    const report = store.report;
    const counts = report.counts && typeof report.counts === 'object' ? report.counts : {};
    const stats = safeArray(report.honeytoken_stats);
    const probes = safeArray(report.canary_probes);

    html += '<h3 class="section-title">Synthetic fragments stored</h3>';
    const countKeys = Object.keys(counts);
    if (!countKeys.length) {
      html += emptyState('No synthetic fragments', 'Run detection to plant honeytokens, chaff and canaries.');
    } else {
      html += '<div class="card"><div class="comp-legend" style="margin-top:0">';
      countKeys.forEach((key) => {
        const normalized = String(key).toLowerCase().replace(/s$/, '');
        const known = KINDS.includes(normalized);
        html += '<span class="legend-item"><span class="legend-dot' +
          (known ? ' legend-dot-' + normalized : '') + '"></span>';
        html += '<span><span class="legend-name">' + esc(key) + '</span>' +
          (known ? info(normalized) : '') +
          ' <span class="legend-n">' + fmtInt(counts[key]) + '</span></span></span>';
      });
      html += '</div></div>';
    }

    html += '<h3 class="section-title">Honeytoken recall per batch' + info('recall') + '</h3>';
    if (!stats.length) {
      html += emptyState('No batches scored', 'Run detection to plant and score honeytokens.');
    } else {
      html += '<div class="table-wrap"><table class="tbl"><thead><tr>';
      html += '<th>batch_id</th><th>scored</th><th>planted</th><th>found</th><th>recall</th>';
      html += '</tr></thead><tbody>';
      stats.forEach((row) => {
        html += '<tr>';
        html += '<td>' + copyId(row.batch_id) + '</td>';
        html += '<td class="mono">' + fmtInt(row.honeytokens_scored) + '</td>';
        html += '<td class="mono">' + fmtInt(row.planted_total) + '</td>';
        html += '<td class="mono">' + fmtInt(row.found_total) + '</td>';
        html += '<td class="mono" style="white-space:nowrap">' + fmtPct(row.recall) + recallBar(row.recall) + '</td>';
        html += '</tr>';
      });
      html += '</tbody></table></div>';
    }

    html += '<h3 class="section-title">Canary probe history</h3>';
    if (!probes.length) {
      html += emptyState('No probes yet',
        'A probe asks a model about each planted canary person. If the model knows the fabricated fact, the canary is tripped.');
    } else {
      html += '<div class="table-wrap"><table class="tbl"><thead><tr>';
      html += '<th>probed_at</th><th>model</th><th>fragment</th><th>result</th><th>response excerpt</th>';
      html += '</tr></thead><tbody>';
      probes.forEach((probe) => {
        const tripped = probe.tripped === true;
        html += '<tr>';
        html += '<td class="mono" style="white-space:nowrap">' + esc(fmtDate(probe.probed_at)) + '</td>';
        html += '<td class="mono">' + esc(probe.model || '—') + '</td>';
        html += '<td>' + copyId(probe.fragment_id) + '</td>';
        html += '<td>' + (tripped ? '<span class="tripped">TRIPPED</span>' : '<span class="clean">clean</span>') + '</td>';
        html += '<td><div class="excerpt">' + esc(probe.response_excerpt || probe.excerpt || '—') + '</div></td>';
        html += '</tr>';
      });
      html += '</tbody></table></div>';
    }

    root.innerHTML = html;
    wireReport(root);
  }

  function wireReport(root) {
    const refresh = root.querySelector('#refresh-report');
    if (refresh) refresh.addEventListener('click', fetchReport);
    const probe = root.querySelector('#run-probe');
    if (probe) probe.addEventListener('click', runCanaryProbe);
  }

  async function fetchReport() {
    if (store.busy.report) return;
    clearAlert('report');
    store.busy.report = true;
    render();
    try {
      store.report = await api('/api/synthetic/report');
    } catch (err) {
      store.report = null;
      fail('report', err);
    } finally {
      store.busy.report = false;
      render();
    }
  }

  async function runCanaryProbe() {
    if (store.busy.probe) return;

    let planned = null;
    if (store.report && store.report.counts) {
      const counts = store.report.counts;
      const candidate = counts.canary !== undefined ? counts.canary : counts.canaries;
      if (Number.isFinite(Number(candidate))) planned = Number(candidate);
    }

    const message = planned === null
      ? 'Run the canary probe? This makes one API call per stored canary fragment.'
      : 'Run the canary probe? This makes ' + planned + ' API call' + (planned === 1 ? '' : 's') + '.';
    if (!window.confirm(message)) return;

    clearAlert('report');
    store.busy.probe = true;
    render();

    try {
      const result = await apiJson('/api/canary-probe', 'POST', {});
      const tripped = num(result && result.tripped, 0);
      const total = num(result && result.total, safeArray(result && result.results).length);
      if (tripped > 0) {
        setAlert('report', 'error', tripped + ' of ' + total +
          ' canaries TRIPPED — a probed model reproduced a planted fact.');
        toast('error', tripped + ' of ' + total + ' canaries tripped');
      } else {
        setAlert('report', 'ok', 'Probe clean: 0 of ' + total + ' canaries tripped.');
        toast('ok', 'Probe clean — 0 of ' + total + ' canaries tripped');
      }
    } catch (err) {
      fail('report', err);
    } finally {
      store.busy.probe = false;
    }

    try {
      store.report = await api('/api/synthetic/report');
    } catch (err) {
      /* keep the probe outcome visible even if the refresh fails */
    }
    render();
  }

  // -------------------------------------------------------- view: settings

  function checkBadge(check) {
    if (!check || check.ok !== true) return '<span class="pill pill-error">fail</span>';
    const detail = String(check.detail || '').toLowerCase();
    if (detail.indexOf('skip') >= 0 || detail.indexOf('disabled') >= 0 || detail.indexOf('not used') >= 0) {
      return '<span class="pill pill-unknown">not used</span>';
    }
    return '<span class="pill pill-ok">ok</span>';
  }

  function verifyCardHtml() {
    let html = '<div class="card">';
    html += '<div class="row-between" style="align-items:center;margin-bottom:6px">';
    html += '<div><h3 class="card-title" style="margin:0">Setup check</h3>';
    html += '<div class="dim" style="font-size:13px">Confirms the Anthropic key works, the local model server answers, document conversion runs and the database is writable.</div></div>';
    html += '<button type="button" class="btn btn-primary" id="run-verify"' + (store.busy.verify ? ' disabled' : '') + '>' +
      (store.busy.verify ? '<span class="spinner"></span>Checking' : 'Verify setup') + '</button>';
    html += '</div>';

    if (store.verifyError) {
      html += '<div class="alert alert-error" style="margin:6px 0 0"><span class="alert-title">Setup check unavailable</span> — ' +
        esc(store.verifyError) + '</div></div>';
      return html;
    }

    if (store.busy.verify && !store.verify) {
      html += '<div class="empty-hint" style="text-align:left;margin:0">Running the checks.</div></div>';
      return html;
    }

    if (!store.verify) {
      html += '<div class="empty-hint" style="text-align:left;margin:0">Not checked yet — press "Verify setup".</div></div>';
      return html;
    }

    const checks = safeArray(store.verify.checks);
    if (!checks.length) {
      html += '<div class="empty-hint" style="text-align:left;margin:0">The server returned no checks.</div></div>';
      return html;
    }

    html += '<div class="check-list">';
    checks.forEach((check) => {
      const name = String(check.name || '');
      html += '<div class="check-row">';
      html += '<div><span class="check-name">' + esc(check.label || name || 'check') + '</span>' +
        (name ? '<span class="check-key">' + esc(name) + '</span>' : '') + '</div>';
      html += '<div>' + checkBadge(check) + '</div>';
      html += '<div><div class="check-detail">' + esc(check.detail || '') + '</div>';
      if (check.ok !== true && REMEDY[name]) {
        html += '<div class="check-remedy">' + esc(REMEDY[name]) + '</div>';
      }
      html += '</div>';
      html += '<div class="check-latency">' +
        (check.latency_ms === undefined || check.latency_ms === null ? '' : fmtInt(check.latency_ms) + ' ms') +
        '</div>';
      html += '</div>';
    });
    html += '</div>';

    if (store.verify.all_ok === true) {
      html += '<div class="alert alert-ok" style="margin:14px 0 0"><span class="alert-title">All checks passed</span> — the pipeline is ready to run.</div>';
    } else {
      html += '<div class="alert alert-warn" style="margin:14px 0 0"><span class="alert-title">Something needs attention</span> — fix the failing rows above, then verify again.</div>';
    }

    html += '</div>';
    return html;
  }

  function renderSettings(root) {
    const settings = store.settings;

    let html = '';
    html += '<div class="page-head">';
    html += '<h2 class="page-title">Settings</h2>';
    html += '<p class="page-lead">Credentials, models and pipeline parameters. The API key is never echoed back — only a masked form of it.</p>';
    html += '</div>';

    html += hintHtml('settings');
    html += alertHtml('settings');

    html += verifyCardHtml();

    if (!settings) {
      html += '<h3 class="section-title">Configuration</h3>';
      html += emptyState('Settings not loaded', 'Could not read the current configuration.',
        '<button type="button" class="btn btn-primary" id="reload-settings">Reload settings</button>');
      root.innerHTML = html;
      wireSettings(root);
      return;
    }

    const keyPlaceholder = settings.has_api_key
      ? (settings.anthropic_api_key_masked || 'key set')
      : 'not set — paste a key';

    const efforts = EFFORTS.slice();
    if (settings.effort && !efforts.includes(settings.effort)) efforts.unshift(settings.effort);
    const llmOn = settings.llm_enabled !== false;

    html += '<h3 class="section-title">Configuration</h3>';
    html += '<form class="card" id="settings-form" autocomplete="off">';
    html += '<div class="form-grid">';

    html += '<div class="field" style="grid-column:1/-1">';
    html += '<label for="f-key">Anthropic API key</label>';
    html += '<input type="password" id="f-key" name="anthropic_api_key" placeholder="' +
      esc(keyPlaceholder) + '" autocomplete="new-password">';
    html += '<div class="hint">' + (settings.has_api_key
      ? 'A key is stored. Leave this empty to keep it unchanged.'
      : 'No key stored. Detection will fail until a key is set.') + '</div>';
    html += '</div>';

    html += '<div class="field"><label for="f-model">Provider model</label>';
    html += '<input type="text" id="f-model" name="model" value="' + esc(settings.model || '') + '"></div>';

    html += '<div class="field"><label for="f-effort">Effort' + info('effort') + '</label><select id="f-effort" name="effort">';
    efforts.forEach((value) => {
      html += '<option value="' + esc(value) + '"' + (settings.effort === value ? ' selected' : '') +
        '>' + esc(value) + '</option>';
    });
    html += '</select></div>';

    html += '<div class="field" style="grid-column:1/-1">';
    html += '<label for="f-llm-enabled" style="cursor:pointer">' +
      '<input type="checkbox" id="f-llm-enabled" name="llm_enabled"' + (llmOn ? ' checked' : '') + '> ' +
      'Use local LLM for synthetic prose' + info('local_llm') + '</label>';
    html += '<div class="hint">ON — honeytokens and chaff are phrased by the local model, so the decoys read like natural prose ' +
      '(slower: roughly a minute of generation per batch). OFF — instant deterministic templates, no local server needed.</div>';
    html += '</div>';

    html += '<div class="field"><label for="f-base">Local LLM base URL</label>';
    html += '<input type="text" id="f-base" name="llm_base_url" value="' + esc(settings.llm_base_url || '') + '"></div>';

    html += '<div class="field"><label for="f-llm-model">Local LLM model</label>';
    html += '<input type="text" id="f-llm-model" name="llm_model" value="' + esc(settings.llm_model || '') + '"></div>';

    html += '</div>';

    html += '<div class="btn-row">';
    html += '<button type="submit" class="btn btn-primary" id="save-settings"' +
      (store.busy.settings ? ' disabled' : '') + '>' +
      (store.busy.settings ? '<span class="spinner"></span>Saving' : 'Save settings') + '</button>';
    html += '<button type="button" class="btn btn-danger" id="clear-key"' +
      (store.busy.settings || !settings.has_api_key ? ' disabled' : '') + '>Clear key</button>';
    html += '</div>';
    html += '</form>';

    html += '<h3 class="section-title">Pipeline parameters (read-only)</h3>';
    html += '<div class="card"><div class="chips">';
    html += '<span class="chip"><span class="chip-k">chunk_size_tokens</span><span class="chip-v">' +
      esc(settings.chunk_size_tokens) + '</span>' + info('chunk_size_tokens') + '</span>';
    html += '<span class="chip"><span class="chip-k">chaff_ratio</span><span class="chip-v">' +
      esc(settings.chaff_ratio) + '</span>' + info('chaff_ratio') + '</span>';
    html += '<span class="chip"><span class="chip-k">honeytoken_rate</span><span class="chip-v">' +
      esc(settings.honeytoken_rate) + '</span>' + info('honeytoken_rate') + '</span>';
    html += '<span class="chip"><span class="chip-k">canaries_per_batch</span><span class="chip-v">' +
      esc(settings.canaries_per_batch) + '</span>' + info('canaries_per_batch') + '</span>';
    html += '</div></div>';

    root.innerHTML = html;
    wireSettings(root);
  }

  function wireSettings(root) {
    const verify = root.querySelector('#run-verify');
    if (verify) verify.addEventListener('click', () => runVerify());

    const reload = root.querySelector('#reload-settings');
    if (reload) {
      reload.addEventListener('click', async () => {
        clearAlert('settings');
        try {
          await loadSettings();
        } catch (err) {
          fail('settings', err);
        }
        render();
      });
    }

    const form = root.querySelector('#settings-form');
    if (form) {
      form.addEventListener('submit', (event) => {
        event.preventDefault();
        const modelNode = root.querySelector('#f-model');
        const effortNode = root.querySelector('#f-effort');
        const baseNode = root.querySelector('#f-base');
        const llmModelNode = root.querySelector('#f-llm-model');
        const llmEnabledNode = root.querySelector('#f-llm-enabled');
        const keyNode = root.querySelector('#f-key');

        const body = {
          model: modelNode ? (modelNode.value || '').trim() : '',
          effort: effortNode ? effortNode.value : '',
          llm_base_url: baseNode ? (baseNode.value || '').trim() : '',
          llm_model: llmModelNode ? (llmModelNode.value || '').trim() : '',
          llm_enabled: llmEnabledNode ? Boolean(llmEnabledNode.checked) : true,
        };
        // Never send an empty key: that would clear a stored one by accident.
        const key = keyNode ? keyNode.value : '';
        if (key && key.trim()) body.anthropic_api_key = key.trim();
        saveSettings(body, 'Settings saved.');
      });
    }

    const clear = root.querySelector('#clear-key');
    if (clear) {
      clear.addEventListener('click', () => {
        if (!window.confirm('Clear the stored Anthropic API key? Detection will fail until a new key is set.')) return;
        saveSettings({ anthropic_api_key: '' }, 'API key cleared.');
      });
    }
  }

  async function saveSettings(body, successText) {
    if (store.busy.settings) return;
    clearAlert('settings');
    store.busy.settings = true;
    render();
    try {
      store.settings = await apiJson('/api/settings', 'PUT', body);
      setAlert('settings', 'ok', successText);
      toast('ok', successText);
    } catch (err) {
      fail('settings', err);
    } finally {
      store.busy.settings = false;
      render();
    }
    // Configuration changed, so the previous verdict is stale.
    runVerify();
  }

  async function runVerify() {
    if (store.busy.verify) return;
    store.busy.verify = true;
    store.verifyError = null;
    render();
    try {
      store.verify = await apiJson('/api/verify', 'POST', {});
      if (store.verify && store.verify.all_ok === false) {
        const failed = safeArray(store.verify.checks).filter((c) => c.ok !== true).length;
        toast('info', 'Setup check: ' + failed +
          (failed === 1 ? ' check needs' : ' checks need') + ' attention');
      }
    } catch (err) {
      // Keep this out of the settings-form alert: only the check is broken.
      store.verify = null;
      store.verifyError = err && err.message ? err.message : String(err);
    } finally {
      store.busy.verify = false;
      render();
    }
  }

  // ------------------------------------------------------- info popovers

  let popoverTrigger = null;

  function popoverNode() {
    return document.getElementById('popover');
  }

  function closePopover(returnFocus) {
    const pop = popoverNode();
    if (pop) {
      pop.hidden = true;
      pop.innerHTML = '';
    }
    if (popoverTrigger) {
      popoverTrigger.setAttribute('aria-expanded', 'false');
      if (returnFocus && document.contains(popoverTrigger)) popoverTrigger.focus();
    }
    popoverTrigger = null;
  }

  function openPopover(button) {
    const key = button.getAttribute('data-info');
    const entry = GLOSSARY[key];
    const pop = popoverNode();
    if (!entry || !pop) return;

    closePopover(false);

    pop.innerHTML = '<div class="pop-head"><h4 id="pop-title">' + esc(entry.title) + '</h4>' +
      '<button type="button" class="pop-close" data-pop-close aria-label="Close definition">&times;</button></div>' +
      '<p class="pop-body">' + esc(entry.body) + '</p>';
    pop.hidden = false;

    const rect = button.getBoundingClientRect();
    const width = pop.offsetWidth || 310;
    const height = pop.offsetHeight || 140;
    let left = Math.min(rect.left, window.innerWidth - width - 12);
    left = Math.max(12, left);
    let top = rect.bottom + 8;
    if (top + height > window.innerHeight - 12) top = rect.top - height - 8;
    top = Math.max(12, Math.min(top, window.innerHeight - height - 12));
    pop.style.left = left + 'px';
    pop.style.top = top + 'px';

    button.setAttribute('aria-expanded', 'true');
    popoverTrigger = button;

    const close = pop.querySelector('[data-pop-close]');
    if (close) close.focus();
  }

  // --------------------------------------------------------- global wiring

  async function copyToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      try {
        await navigator.clipboard.writeText(text);
        return true;
      } catch (err) {
        /* fall through to the legacy path */
      }
    }
    try {
      const helper = document.createElement('textarea');
      helper.value = text;
      helper.setAttribute('readonly', '');
      helper.style.position = 'fixed';
      helper.style.opacity = '0';
      document.body.appendChild(helper);
      helper.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(helper);
      return ok;
    } catch (err) {
      return false;
    }
  }

  document.addEventListener('click', async (event) => {
    const target = event.target;
    if (!target || !target.closest) return;

    const infoButton = target.closest('[data-info]');
    if (infoButton) {
      event.preventDefault();
      if (popoverTrigger === infoButton) closePopover(true);
      else openPopover(infoButton);
      return;
    }

    if (target.closest('[data-pop-close]')) {
      event.preventDefault();
      closePopover(true);
      return;
    }

    if (popoverTrigger && !target.closest('#popover')) closePopover(false);

    const copyButton = target.closest('[data-copy]');
    if (copyButton) {
      event.preventDefault();
      const value = copyButton.getAttribute('data-copy');
      const ok = await copyToClipboard(value);
      if (!ok) {
        toast('error', 'Could not copy to clipboard');
        return;
      }
      const original = copyButton.textContent;
      copyButton.classList.add('copied');
      copyButton.textContent = 'copied';
      window.setTimeout(() => {
        copyButton.classList.remove('copied');
        copyButton.textContent = original;
      }, 900);
    }
  });

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && popoverTrigger) {
      event.preventDefault();
      closePopover(true);
    }
  });

  window.addEventListener('resize', () => closePopover(false));
  window.addEventListener('hashchange', onRoute);

  async function init() {
    try {
      store.introDismissed = window.localStorage.getItem(INTRO_KEY) === '1';
    } catch (err) {
      store.introDismissed = false;
    }

    if (!location.hash || !VIEWS.includes(location.hash.replace(/^#/, ''))) {
      location.hash = '#document';
    }

    const main = document.getElementById('main');
    if (main) main.addEventListener('scroll', () => closePopover(false), { passive: true });

    render();

    await Promise.all([
      loadSettings().catch((err) => { setAlert('settings', 'error', err.message); }),
      loadDocs().catch((err) => { setAlert('document', 'error', err.message); }),
    ]);

    onRoute();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
