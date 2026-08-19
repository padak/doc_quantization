/* =========================================================================
 * doc_quantization — observability console
 * Vanilla ES2020+. No frameworks, no build step, no external assets.
 * ========================================================================= */
'use strict';

(function () {
  // ----------------------------------------------------------------- state

  const VIEWS = ['document', 'chunks', 'batch', 'llm', 'redaction', 'report', 'settings'];
  const KINDS = ['real', 'honeytoken', 'chaff', 'canary'];
  const EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max'];

  const store = {
    settings: null,
    docs: [],
    doc: null,              // { doc_id, name, path, markdown, chunks[] }
    detect: null,           // last POST /api/detect payload
    detectDocId: null,
    redaction: null,
    redactionDocId: null,
    report: null,
    batchView: 'local',     // 'local' | 'provider'
    busy: {                 // in-flight flags
      upload: false,
      detect: false,
      redaction: false,
      report: false,
      probe: false,
      settings: false,
    },
    alerts: {},             // view -> { kind, text }
  };

  // ---------------------------------------------------------------- utils

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

  function kindClass(kind) {
    return KINDS.includes(kind) ? kind : 'neutral';
  }

  function kindBadge(kind) {
    const k = String(kind || 'unknown');
    return '<span class="badge badge-' + esc(kindClass(k)) + '">' + esc(k) + '</span>';
  }

  function statusPill(status) {
    const s = String(status || 'unknown').toLowerCase();
    const known = ['ok', 'refusal', 'error'].includes(s) ? s : 'unknown';
    return '<span class="pill pill-' + known + '">' + esc(s) + '</span>';
  }

  function copyId(value, label) {
    if (!value) return '<span class="mono" style="color:var(--fg-dim)">—</span>';
    const text = label === undefined ? value : label;
    return '<button type="button" class="copy-id" data-copy="' + esc(value) +
      '" title="Click to copy ' + esc(value) + '">' + esc(text) + '</button>';
  }

  function emptyState(title, hint) {
    return '<div class="empty"><div class="empty-title">' + esc(title) +
      '</div><div class="empty-hint">' + hint + '</div></div>';
  }

  // ---------------------------------------------------------------- toasts

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
    return '<div class="alert alert-' + cls + '"><div class="alert-body"><span class="alert-title">' +
      esc(title) + '</span> — ' + esc(a.text) + '</div></div>';
  }

  function fail(view, err) {
    const message = err && err.message ? err.message : String(err);
    setAlert(view, 'error', message);
    toast('error', message);
  }

  // ------------------------------------------------------------------ api

  async function api(path, options) {
    let res;
    try {
      res = await fetch(path, options);
    } catch (err) {
      throw new Error('Network error contacting ' + path + ' (' + (err && err.message ? err.message : 'unknown') + ')');
    }

    const raw = await res.text();
    let data = null;
    if (raw) {
      try {
        data = JSON.parse(raw);
      } catch (err) {
        data = null;
      }
    }

    if (!res.ok) {
      let detail;
      if (data && typeof data.detail === 'string') {
        detail = data.detail;
      } else if (data && data.detail !== undefined && data.detail !== null) {
        detail = pretty(data.detail);
      } else {
        // Not a JSON error body (proxy error page, empty response, ...).
        const trimmed = (raw || '').trim();
        detail = (!trimmed || trimmed.charAt(0) === '<')
          ? ('HTTP ' + res.status + ' ' + res.statusText + ' from ' + path)
          : trimmed.slice(0, 300);
      }
      const error = new Error(detail);
      error.status = res.status;
      throw error;
    }

    return data;
  }

  function apiJson(path, method, body) {
    return api(path, {
      method: method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  }

  // -------------------------------------------------------------- loaders

  async function loadSettings() {
    store.settings = await api('/api/settings');
  }

  async function loadDocs() {
    const list = await api('/api/documents');
    store.docs = safeArray(list);
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

  // --------------------------------------------------------------- router

  function currentView() {
    const hash = String(location.hash || '').replace(/^#/, '');
    return VIEWS.includes(hash) ? hash : 'document';
  }

  function viewNode(name) {
    return document.querySelector('.view[data-view="' + name + '"]');
  }

  function render() {
    const view = currentView();

    VIEWS.forEach((name) => {
      const node = viewNode(name);
      if (node) node.hidden = name !== view;
    });

    document.querySelectorAll('[data-nav]').forEach((node) => {
      node.classList.toggle('is-active', node.getAttribute('data-nav') === view);
    });

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
    render();
    // Lazily fetch what the view needs.
    if (view === 'redaction' && store.doc && !store.busy.redaction &&
        (!store.redaction || store.redactionDocId !== store.doc.doc_id)) {
      fetchRedaction();
    }
    if (view === 'report' && !store.report && !store.busy.report) {
      fetchReport();
    }
  }

  // ------------------------------------------------------- view: document

  function renderDocument(root) {
    const busy = store.busy.upload;

    let html = '';
    html += '<div class="page-head">';
    html += '<h2 class="page-title">Document</h2>';
    html += '<p class="page-sub">Upload a source document. It is converted to Markdown, then split into small chunks stored under random IDs. Everything downstream operates on the active document.</p>';
    html += '</div>';

    html += alertHtml('document');

    html += '<div class="dropzone' + (busy ? ' is-busy' : '') + '" id="dropzone" tabindex="0" role="button" aria-label="Upload a document">';
    if (busy) {
      html += '<div class="dropzone-busy"><span class="spinner"></span><span>Converting and chunking…</span></div>';
    } else {
      html += '<div class="dropzone-title">Drop a document here, or click to choose a file</div>';
      html += '<div class="dropzone-hint">The file is converted to Markdown on the server.</div>';
    }
    html += '</div>';
    html += '<input type="file" id="file-input" style="display:none">';

    if (store.doc) {
      const chunkCount = store.doc.chunks.length;
      html += '<h3 class="section-title">Active document</h3>';
      html += '<div class="card">';
      html += '<div class="row-between" style="margin-bottom:12px">';
      html += '<div style="min-width:0"><div style="font-size:15px;word-break:break-all">' + esc(store.doc.name) + '</div>';
      if (store.doc.path && store.doc.path !== store.doc.name) {
        html += '<div class="mono" style="color:var(--fg-dim);word-break:break-all">' + esc(store.doc.path) + '</div>';
      }
      html += '</div>';
      html += '<div class="chips">';
      html += '<span class="chip"><span class="chip-k">doc_id</span><span class="chip-v">' + copyId(store.doc.doc_id) + '</span></span>';
      html += '<span class="chip"><span class="chip-k">chunks</span><span class="chip-v">' + fmtInt(chunkCount) + '</span></span>';
      html += '</div>';
      html += '</div>';
      html += '<h3 class="section-title" style="margin-top:0">Converted Markdown</h3>';
      html += '<pre class="code code-tall">' + esc(store.doc.markdown || '(empty)') + '</pre>';
      html += '</div>';
    }

    html += '<h3 class="section-title">Previously ingested documents</h3>';
    if (!store.docs.length) {
      html += '<div class="empty"><div class="empty-title">No documents yet</div><div class="empty-hint">Uploaded documents appear here.</div></div>';
    } else {
      html += '<div class="doc-list">';
      store.docs.forEach((doc) => {
        const active = store.doc && store.doc.doc_id === doc.doc_id;
        html += '<button type="button" class="doc-row' + (active ? ' is-active' : '') + '" data-doc="' + esc(doc.doc_id) + '">';
        html += '<div class="doc-row-main">';
        html += '<div class="doc-row-path">' + esc(basename(doc.path) || doc.doc_id) + '</div>';
        html += '<div class="doc-row-meta">' + esc(doc.doc_id) + ' · ' + esc(fmtDate(doc.created_at)) + '</div>';
        html += '</div>';
        html += '<div class="doc-row-count">' + fmtInt(doc.chunk_count) + ' chunks</div>';
        html += '</button>';
      });
      html += '</div>';
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
        (store.doc ? store.doc.chunks.length : 0) + ' chunks');
      try {
        await loadDocs();
      } catch (err) {
        /* document list is secondary; ignore */
      }
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
    } catch (err) {
      fail('document', err);
    }
    render();
  }

  // --------------------------------------------------------- view: chunks

  function renderChunks(root) {
    let html = '';
    html += '<div class="page-head">';
    html += '<h2 class="page-title">Chunks</h2>';
    html += '<p class="page-sub">The document split into small chunks, each stored under a random ID. Token boundaries are shown with alternating shading; whitespace is preserved exactly.</p>';
    html += '</div>';

    html += alertHtml('chunks');

    if (!store.doc) {
      root.innerHTML = html + emptyState('No active document',
        'Upload a document first on the <a href="#document">Document</a> step.');
      return;
    }

    const chunks = store.doc.chunks.slice().sort((a, b) => num(a.seq) - num(b.seq));
    const totalTokens = chunks.reduce((sum, chunk) => {
      const count = Number(chunk.token_count);
      if (Number.isFinite(count)) return sum + count;
      return sum + safeArray(chunk.tokens).length;
    }, 0);
    const extended = chunks.filter((chunk) => chunk.extended === true).length;

    html += '<div class="stat-row">';
    html += '<div class="stat"><div class="stat-k">Chunks</div><div class="stat-v">' + fmtInt(chunks.length) + '</div></div>';
    html += '<div class="stat"><div class="stat-k">Total tokens</div><div class="stat-v">' + fmtInt(totalTokens) + '</div></div>';
    html += '<div class="stat"><div class="stat-k">Extended cuts</div><div class="stat-v">' + fmtInt(extended) + '</div></div>';
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
      html += '<span class="chunk-seq">seq ' + esc(chunk.seq === null || chunk.seq === undefined ? '?' : chunk.seq) + '</span>';
      html += copyId(chunk.chunk_id);
      if (chunk.extended === true) {
        html += '<span class="badge badge-extended" title="cut moved to keep a name whole">extended</span>';
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

    root.innerHTML = html;
  }

  // ---------------------------------------------------------- view: batch

  function compositionHtml(composition) {
    const comp = composition && typeof composition === 'object' ? composition : {};
    const values = KINDS.map((kind) => ({ kind: kind, count: num(comp[kind], 0) }));
    const total = values.reduce((sum, item) => sum + item.count, 0);

    let html = '<div class="card">';
    html += '<div class="row-between" style="margin-bottom:10px"><h3 class="section-title" style="margin:0">Batch composition</h3>';
    html += '<span class="mono" style="color:var(--fg-muted)">' + fmtInt(total) + ' fragments submitted</span></div>';

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
        (pct >= 7 ? esc(String(item.count)) : '') + '</div>';
    });
    html += '</div>';

    html += '<div class="comp-legend">';
    values.forEach((item) => {
      const pct = total > 0 ? (item.count / total) * 100 : 0;
      html += '<span class="legend-item"><span class="legend-dot legend-dot-' + item.kind + '"></span>' +
        esc(item.kind) + ' <span class="legend-n">' + fmtInt(item.count) + '</span> ' +
        '<span style="color:var(--fg-dim)">(' + pct.toFixed(1) + '%)</span></span>';
    });
    html += '</div>';
    html += '</div>';
    return html;
  }

  function renderBatch(root) {
    const running = store.busy.detect;
    const hasDoc = Boolean(store.doc);

    let html = '';
    html += '<div class="page-head"><div class="row-between">';
    html += '<div><h2 class="page-title">Batch</h2>';
    html += '<p class="page-sub">Real chunks are shuffled together with synthetic fragments — honeytokens (recall meters), chaff (1:1 decoys) and canaries (training-misuse tripwires) — before anything leaves the machine.</p></div>';
    html += '<button type="button" class="btn btn-primary" id="run-detect"' +
      (running || !hasDoc ? ' disabled' : '') + '>' +
      (running ? '<span class="spinner"></span>Running detection…' : 'Run detection') + '</button>';
    html += '</div></div>';

    html += alertHtml('batch');

    if (!hasDoc) {
      root.innerHTML = html + emptyState('No active document',
        'Upload a document first on the <a href="#document">Document</a> step.');
      wireBatch(root);
      return;
    }

    if (!store.detect) {
      html += emptyState('No detection run yet',
        running ? 'Submitting the batch to the provider…' : 'Press "Run detection" to build and submit the batch.');
      root.innerHTML = html;
      wireBatch(root);
      return;
    }

    const detect = store.detect;
    const requests = safeArray(detect.requests);
    const provider = store.batchView === 'provider';

    html += '<div class="chips" style="margin-bottom:12px">';
    html += '<span class="chip"><span class="chip-k">batch_id</span><span class="chip-v">' + copyId(detect.batch_id) + '</span></span>';
    if (detect.entities_stored !== undefined && detect.entities_stored !== null) {
      html += '<span class="chip"><span class="chip-k">entities stored</span><span class="chip-v">' + fmtInt(detect.entities_stored) + '</span></span>';
    }
    html += '</div>';

    html += compositionHtml(detect.composition);

    html += '<div class="row-between" style="margin:22px 0 0">';
    html += '<h3 class="section-title" style="margin:0">Submitted fragments <span style="color:var(--fg-muted);text-transform:none;letter-spacing:0">(' + fmtInt(requests.length) + ', in submission order)</span></h3>';
    html += '<div class="seg-toggle" role="group" aria-label="Fragment view">';
    html += '<button type="button" class="seg-btn' + (provider ? '' : ' is-active') + '" data-batchview="local"' +
      (provider ? '' : ' aria-pressed="true"') + '>Local view</button>';
    html += '<button type="button" class="seg-btn' + (provider ? ' is-active' : '') + '" data-batchview="provider"' +
      (provider ? ' aria-pressed="true"' : '') + '>Provider view</button>';
    html += '</div>';
    html += '</div>';

    if (provider) {
      html += '<div class="provider-caption">This is everything the provider can see: shuffled fragments, half of them synthetic, no ordering, no labels.</div>';
    } else {
      html += '<div style="height:12px"></div>';
    }

    if (!requests.length) {
      html += emptyState('No fragments', 'The batch contained no fragments.');
    } else {
      html += '<div class="frag-list' + (provider ? ' is-provider' : '') + '">';
      requests.forEach((req) => {
        const kind = String(req.kind || 'unknown');
        if (provider) {
          html += '<div class="frag">';
          html += '<div>' + copyId(req.custom_id) + '</div>';
          html += '<div class="frag-text">' + esc(req.text || '') + '</div>';
          html += '</div>';
        } else {
          html += '<div class="frag frag-kind-' + esc(kindClass(kind)) + '">';
          html += '<div class="frag-meta">' + kindBadge(kind);
          if (req.seq !== null && req.seq !== undefined) {
            html += '<span class="frag-seq">seq ' + esc(req.seq) + '</span>';
          }
          html += '</div>';
          html += '<div>' + copyId(req.custom_id) + '</div>';
          html += '<div class="frag-text">' + esc(req.text || '') + '</div>';
          html += '</div>';
        }
      });
      html += '</div>';
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
  }

  async function runDetect() {
    if (!store.doc || store.busy.detect) return;
    clearAlert('batch');
    store.busy.detect = true;
    render();

    try {
      const payload = await apiJson('/api/detect', 'POST', { doc_id: store.doc.doc_id });
      store.detect = payload;
      store.detectDocId = store.doc.doc_id;
      // Detection changes the stored entities, so cached derivatives are stale.
      store.redaction = null;
      store.redactionDocId = null;
      store.report = null;
      const results = safeArray(payload && payload.results);
      const bad = results.filter((r) => r.status !== 'ok').length;
      toast(bad ? 'info' : 'ok', 'Detection finished — ' + results.length + ' results' +
        (bad ? ', ' + bad + ' not ok' : ''));
    } catch (err) {
      fail('batch', err);
    } finally {
      store.busy.detect = false;
      render();
    }
  }

  // ------------------------------------------------------- view: llm i/o

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
    html += '<p class="page-sub">Exactly what was sent and what came back for the last detection run: the shared payload template, then one row per fragment with its status, latency and parsed entities.</p>';
    html += '</div>';

    html += alertHtml('llm');

    if (!store.detect) {
      root.innerHTML = html + emptyState('No detection run yet',
        'Run detection on the <a href="#batch">Batch</a> step first.');
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
    html += '<div class="stat"><div class="stat-k">Requests</div><div class="stat-v">' + fmtInt(rows.length) + '</div></div>';
    html += '<div class="stat"><div class="stat-k">ok</div><div class="stat-v stat-v-ok">' + fmtInt(counts.ok) + '</div></div>';
    html += '<div class="stat"><div class="stat-k">refusal</div><div class="stat-v' + (counts.refusal ? ' stat-v-warn' : '') + '">' + fmtInt(counts.refusal) + '</div></div>';
    html += '<div class="stat"><div class="stat-k">error</div><div class="stat-v" style="' + (counts.error ? 'color:var(--error)' : '') + '">' + fmtInt(counts.error) + '</div></div>';
    html += '<div class="stat"><div class="stat-k">Total latency</div><div class="stat-v">' + fmtMs(totalLatency) + '</div></div>';
    html += '<div class="stat"><div class="stat-k">Avg latency</div><div class="stat-v">' + (latencies.length ? fmtMs(avgLatency) : '—') + '</div></div>';

    const recall = detect.honeytoken_recall;
    if (recall && typeof recall === 'object') {
      html += '<div class="stat"><div class="stat-k">Honeytoken recall</div><div class="stat-v">' + fmtPct(recall.recall) +
        '</div><div class="mono" style="color:var(--fg-dim);font-size:11px">' +
        fmtInt(recall.found) + ' / ' + fmtInt(recall.planted) + ' found</div></div>';
    }
    html += '</div>';

    // Shared payload template
    const template = detect.payload_template && typeof detect.payload_template === 'object' ? detect.payload_template : null;
    html += '<details class="payload"><summary><span class="io-caret">&#9662;</span><strong>Shared payload template</strong>' +
      '<span style="color:var(--fg-dim);font-size:12px">— identical for every fragment</span></summary>';
    html += '<div class="payload-body">';
    if (!template) {
      html += '<div class="empty-hint">No payload template returned.</div>';
    } else {
      html += '<div class="chips" style="margin-bottom:12px">';
      if (template.model) html += '<span class="chip"><span class="chip-k">model</span><span class="chip-v">' + esc(template.model) + '</span></span>';
      if (template.max_tokens !== undefined && template.max_tokens !== null) {
        html += '<span class="chip"><span class="chip-k">max_tokens</span><span class="chip-v">' + fmtInt(template.max_tokens) + '</span></span>';
      }
      if (store.settings && store.settings.effort) {
        html += '<span class="chip"><span class="chip-k">effort</span><span class="chip-v">' + esc(store.settings.effort) + '</span></span>';
      }
      html += '</div>';
      html += '<h4 style="margin:0 0 6px;font-size:10.5px;text-transform:uppercase;letter-spacing:0.08em;color:var(--fg-dim)">System prompt</h4>';
      html += '<pre class="code">' + esc(typeof template.system === 'string' ? template.system : pretty(template.system)) + '</pre>';
      html += '<h4 style="margin:14px 0 6px;font-size:10.5px;text-transform:uppercase;letter-spacing:0.08em;color:var(--fg-dim)">Output schema</h4>';
      html += '<pre class="code code-nowrap">' + esc(pretty(template.output_config)) + '</pre>';
    }
    html += '</div></details>';

    html += '<h3 class="section-title">Per-fragment exchange</h3>';
    if (!rows.length) {
      html += emptyState('No requests', 'The run produced no request rows.');
    } else {
      html += '<div class="table-wrap" style="overflow-x:hidden">';
      rows.forEach((row) => {
        const req = row.request;
        const result = row.result;
        const status = String((result && result.status) || 'unknown').toLowerCase();
        const kind = String((req && req.kind) || (result && result.kind) || 'unknown');
        const customId = (req && req.custom_id) || (result && result.custom_id) || '';
        const rowCls = status === 'refusal' ? ' is-refusal' : (status === 'error' ? ' is-error' : '');

        html += '<details class="io-row' + rowCls + '">';
        html += '<summary class="io-summary">';
        html += '<span class="io-caret">&#9656;</span>';
        html += '<span class="mono" style="color:var(--fg-muted)">' + esc(customId) + '</span>';
        html += kindBadge(kind);
        html += statusPill(status);
        html += '<span class="io-latency">' + (result && result.latency_ms !== undefined && result.latency_ms !== null ? fmtInt(result.latency_ms) + ' ms' : '—') + '</span>';
        html += '</summary>';
        html += '<div class="io-detail">';

        html += '<h4>Request text</h4>';
        html += '<pre class="code">' + esc(req ? (req.text || '') : '(no matching request)') + '</pre>';

        if (result) {
          const entities = safeArray(result.entities);
          html += '<h4>Parsed entities (' + fmtInt(entities.length) + ')</h4>';
          if (entities.length) {
            html += '<div class="ent-list">' + entities.map(entityHtml).join('') + '</div>';
          } else {
            html += '<div class="empty-hint">No entities returned.</div>';
          }
          if (result.raw_text) {
            html += '<h4>Raw response</h4><pre class="code">' + esc(result.raw_text) + '</pre>';
          }
          if (result.detail) {
            html += '<h4>Detail</h4><div class="alert alert-' + (status === 'error' ? 'error' : 'warn') +
              '" style="margin:0"><div class="alert-body">' + esc(result.detail) + '</div></div>';
          }
        } else {
          html += '<h4>Response</h4><div class="empty-hint">No result returned for this custom_id.</div>';
        }

        html += '</div></details>';
      });
      html += '</div>';
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
    html += '<p class="page-sub">The document reassembled from its chunks with every detected name replaced by a placeholder.</p></div>';
    html += '<button type="button" class="btn" id="refresh-redaction"' +
      (store.busy.redaction || !store.doc ? ' disabled' : '') + '>' +
      (store.busy.redaction ? '<span class="spinner"></span>Loading…' : 'Refresh') + '</button>';
    html += '</div></div>';

    html += alertHtml('redaction');

    if (!store.doc) {
      root.innerHTML = html + emptyState('No active document',
        'Upload a document first on the <a href="#document">Document</a> step.');
      return;
    }

    if (!store.redaction) {
      root.innerHTML = html + emptyState(store.busy.redaction ? 'Loading redaction…' : 'No redaction yet',
        store.busy.redaction ? 'Fetching the reassembled document.'
          : 'Run detection on the <a href="#batch">Batch</a> step, then refresh.');
      wireRedaction(root);
      return;
    }

    const data = store.redaction;
    const entities = safeArray(data.entities);

    html += '<div class="diff-grid">';
    html += '<div class="pane"><div class="pane-head">Original</div><pre class="code">' +
      esc(data.original || '(empty)') + '</pre></div>';
    html += '<div class="pane"><div class="pane-head">Redacted</div><pre class="code">' +
      highlightRedacted(data.redacted || '(empty)') + '</pre></div>';
    html += '</div>';

    html += '<h3 class="section-title">Detected entities (' + fmtInt(entities.length) + ')</h3>';
    if (!entities.length) {
      html += '<div class="empty"><div class="empty-title">No entities</div><div class="empty-hint">Nothing was redacted in this document.</div></div>';
    } else {
      html += '<div class="card"><div class="ent-list">' + entities.map(entityHtml).join('') + '</div></div>';
    }

    root.innerHTML = html;
    wireRedaction(root);
  }

  function wireRedaction(root) {
    const button = root.querySelector('#refresh-redaction');
    if (button) button.addEventListener('click', fetchRedaction);
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

  // --------------------------------------------------------- view: report

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
    html += '<p class="page-sub">Synthetic-fragment inventory, honeytoken recall per batch, and canary probe history — the evidence that the defences are actually working.</p></div>';
    html += '<div style="display:flex;gap:8px;flex-wrap:wrap">';
    html += '<button type="button" class="btn" id="refresh-report"' + (store.busy.report ? ' disabled' : '') + '>' +
      (store.busy.report ? '<span class="spinner"></span>Loading…' : 'Refresh') + '</button>';
    html += '<button type="button" class="btn btn-primary" id="run-probe"' + (store.busy.probe ? ' disabled' : '') + '>' +
      (store.busy.probe ? '<span class="spinner"></span>Probing…' : 'Run canary probe') + '</button>';
    html += '</div></div></div>';

    html += alertHtml('report');

    if (!store.report) {
      root.innerHTML = html + emptyState(store.busy.report ? 'Loading report…' : 'No report loaded',
        store.busy.report ? 'Fetching synthetic-fragment statistics.' : 'Press Refresh to load the synthetic report.');
      wireReport(root);
      return;
    }

    const report = store.report;
    const counts = report.counts && typeof report.counts === 'object' ? report.counts : {};
    const stats = safeArray(report.honeytoken_stats);
    const probes = safeArray(report.canary_probes);

    html += '<h3 class="section-title">Synthetic fragments by kind</h3>';
    const countKeys = Object.keys(counts);
    if (!countKeys.length) {
      html += '<div class="empty"><div class="empty-title">No counts</div><div class="empty-hint">No synthetic fragments recorded yet.</div></div>';
    } else {
      html += '<div class="comp-legend" style="margin-top:0">';
      countKeys.forEach((key) => {
        const normalized = String(key).toLowerCase().replace(/s$/, '');
        const dot = KINDS.includes(normalized) ? ' legend-dot-' + normalized : '';
        html += '<span class="legend-item"><span class="legend-dot' + dot + '"' +
          (dot ? '' : ' style="background:var(--fg-dim)"') + '></span>' + esc(key) +
          ' <span class="legend-n">' + fmtInt(counts[key]) + '</span></span>';
      });
      html += '</div>';
    }

    html += '<h3 class="section-title">Honeytoken recall per batch</h3>';
    if (!stats.length) {
      html += '<div class="empty"><div class="empty-title">No batches scored</div><div class="empty-hint">Run detection to plant and score honeytokens.</div></div>';
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
      html += '<div class="empty"><div class="empty-title">No probes yet</div><div class="empty-hint">Run a canary probe to test whether a model has memorised a planted fact.</div></div>';
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
        html += '<td><div class="excerpt">' + esc(probe.response_excerpt || '—') + '</div></td>';
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
        setAlert('report', 'error', tripped + ' of ' + total + ' canaries TRIPPED — a probed model reproduced a planted fact.');
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

  // ------------------------------------------------------- view: settings

  function renderSettings(root) {
    const settings = store.settings;

    let html = '';
    html += '<div class="page-head">';
    html += '<h2 class="page-title">Settings</h2>';
    html += '<p class="page-sub">Credentials and model configuration. The API key is never echoed back — only a masked form of it.</p>';
    html += '</div>';

    html += alertHtml('settings');

    if (!settings) {
      html += emptyState('Settings not loaded', 'Could not read the current configuration.');
      html += '<div style="margin-top:12px"><button type="button" class="btn" id="reload-settings">Reload settings</button></div>';
      root.innerHTML = html;
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
      return;
    }

    const keyPlaceholder = settings.has_api_key
      ? (settings.anthropic_api_key_masked || 'key set')
      : 'not set — paste a key';

    const efforts = EFFORTS.slice();
    if (settings.effort && !efforts.includes(settings.effort)) efforts.unshift(settings.effort);

    html += '<form class="card" id="settings-form" autocomplete="off">';
    html += '<div class="form-grid">';

    html += '<div class="field" style="grid-column:1/-1">';
    html += '<label for="f-key">Anthropic API key</label>';
    html += '<input type="password" id="f-key" name="anthropic_api_key" placeholder="' + esc(keyPlaceholder) + '" autocomplete="new-password">';
    html += '<div class="hint">' + (settings.has_api_key
      ? 'A key is stored. Leave empty to keep it unchanged.'
      : 'No key stored. Detection will fail until a key is set.') + '</div>';
    html += '</div>';

    html += '<div class="field"><label for="f-model">Model</label>';
    html += '<input type="text" id="f-model" name="model" value="' + esc(settings.model || '') + '"></div>';

    html += '<div class="field"><label for="f-effort">Effort</label><select id="f-effort" name="effort">';
    efforts.forEach((value) => {
      html += '<option value="' + esc(value) + '"' + (settings.effort === value ? ' selected' : '') + '>' + esc(value) + '</option>';
    });
    html += '</select></div>';

    html += '<div class="field"><label for="f-base">LLM base URL</label>';
    html += '<input type="text" id="f-base" name="llm_base_url" value="' + esc(settings.llm_base_url || '') + '"></div>';

    html += '<div class="field"><label for="f-llm-model">LLM model</label>';
    html += '<input type="text" id="f-llm-model" name="llm_model" value="' + esc(settings.llm_model || '') + '"></div>';

    html += '</div>';

    html += '<div style="display:flex;gap:9px;flex-wrap:wrap;margin-top:6px">';
    html += '<button type="submit" class="btn btn-primary" id="save-settings"' + (store.busy.settings ? ' disabled' : '') + '>' +
      (store.busy.settings ? '<span class="spinner"></span>Saving…' : 'Save settings') + '</button>';
    html += '<button type="button" class="btn btn-danger" id="clear-key"' +
      (store.busy.settings || !settings.has_api_key ? ' disabled' : '') + '>Clear key</button>';
    html += '</div>';
    html += '</form>';

    html += '<h3 class="section-title">Pipeline parameters (read-only)</h3>';
    html += '<div class="chips">';
    html += '<span class="chip"><span class="chip-k">chunk_size_tokens</span><span class="chip-v">' + esc(settings.chunk_size_tokens) + '</span></span>';
    html += '<span class="chip"><span class="chip-k">chaff_ratio</span><span class="chip-v">' + esc(settings.chaff_ratio) + '</span></span>';
    html += '<span class="chip"><span class="chip-k">honeytoken_rate</span><span class="chip-v">' + esc(settings.honeytoken_rate) + '</span></span>';
    html += '<span class="chip"><span class="chip-k">canaries_per_batch</span><span class="chip-v">' + esc(settings.canaries_per_batch) + '</span></span>';
    html += '</div>';

    root.innerHTML = html;
    wireSettings(root);
  }

  function wireSettings(root) {
    const form = root.querySelector('#settings-form');
    if (form) {
      form.addEventListener('submit', (event) => {
        event.preventDefault();
        const body = {
          model: (root.querySelector('#f-model').value || '').trim(),
          effort: root.querySelector('#f-effort').value,
          llm_base_url: (root.querySelector('#f-base').value || '').trim(),
          llm_model: (root.querySelector('#f-llm-model').value || '').trim(),
        };
        const key = root.querySelector('#f-key').value;
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
  }

  // ------------------------------------------------------- global wiring

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
    const target = event.target.closest ? event.target.closest('[data-copy]') : null;
    if (!target) return;
    event.preventDefault();
    const value = target.getAttribute('data-copy');
    const ok = await copyToClipboard(value);
    if (!ok) {
      toast('error', 'Could not copy to clipboard');
      return;
    }
    const original = target.textContent;
    target.classList.add('copied');
    target.textContent = 'copied';
    window.setTimeout(() => {
      target.classList.remove('copied');
      target.textContent = original;
    }, 900);
  });

  window.addEventListener('hashchange', onRoute);

  async function init() {
    if (!location.hash || !VIEWS.includes(location.hash.replace(/^#/, ''))) {
      location.hash = '#document';
    }

    updateSidebarDoc();
    render();

    const tasks = [
      loadSettings().catch((err) => { setAlert('settings', 'error', err.message); }),
      loadDocs().catch((err) => { setAlert('document', 'error', err.message); }),
    ];
    await Promise.all(tasks);
    onRoute();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
