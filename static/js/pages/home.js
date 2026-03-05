import { api } from '../api.js';
import * as state from '../state.js';
import { navigate } from '../router.js';
import { connectSSE } from '../sse.js';

export async function renderHome(el) {
  el.innerHTML = `
    <!-- Hero Nav Cards -->
    <div class="home-hero">
      <div class="hero-card" id="hero-tableone">
        <div class="hero-card-body">
          <h3>Table One Results</h3>
          <p>Explore cohorts, demographics, medications, ventilation, outcomes</p>
        </div>
        <span class="hero-arrow" aria-hidden="true">&rarr;</span>
      </div>
      <div class="hero-card" id="hero-validation">
        <div class="hero-card-body">
          <h3>Validation</h3>
          <p>Review DQA validation status for all CLIF tables</p>
        </div>
        <span class="hero-arrow" aria-hidden="true">&rarr;</span>
      </div>
    </div>

    <!-- AI All-Tables Interpretation -->
    <div id="ai-all-section" style="display:none;margin-bottom:28px;">
      <div class="card ai-card">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
          <div class="loading-spinner" id="ai-all-spinner"></div>
          <span style="font-size:0.9rem;font-weight:600;">Validation Summary</span>
          <span id="ai-model-badge" style="font-size:0.68rem;color:var(--text-muted);display:none;"></span>
          <button class="btn btn-sm btn-secondary" id="btn-ai-expand" style="margin-left:auto;display:none;">Expand</button>
          <a href="#" id="btn-ai-reinterpret" style="margin-left:8px;font-size:0.8rem;display:none;">Re-interpret</a>
        </div>
        <div id="ai-all-text"></div>
      </div>
    </div>

    <!-- Analyze All -->
    <div id="home-actions">
      <h3>Analyze All Tables</h3>
      <div style="display:flex;gap:12px;align-items:center;margin-bottom:12px;">
        <label><input type="checkbox" id="chk-aggregates"> Generate Summary Aggregates</label>
      </div>
      <div style="margin-bottom:12px;">
        <a href="#" id="toggle-table-picker" style="font-size:0.85rem;">Select Tables &#9662;</a>
        <div id="table-picker" style="display:none;margin-top:8px;padding:12px;border:1px solid var(--border);border-radius:6px;">
          <div style="margin-bottom:8px;font-size:0.8rem;">
            <a href="#" id="picker-all">All</a> &middot; <a href="#" id="picker-none">None</a>
          </div>
          <div id="picker-grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:4px 16px;"></div>
        </div>
      </div>
      <button class="btn btn-primary" id="btn-analyze-all">Analyze All Tables</button>
      <button class="btn btn-secondary" id="btn-regenerate" style="margin-left:8px;">Regenerate All Reports</button>
      <div id="bulk-progress" style="display:none;margin-top:16px;">
        <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
        <div id="progress-text" style="margin-top:8px;font-size:0.85rem;opacity:0.7;"></div>
      </div>
      <div id="bulk-results"></div>
    </div>
    <div id="download-section" style="margin-top:16px;display:none;">
      <h3>Download Reports</h3>
      <a class="btn btn-secondary" id="dl-pdf" style="text-decoration:none;display:inline-block;">Download PDF Report</a>
      <a class="btn btn-secondary" id="dl-csv" style="text-decoration:none;display:inline-block;margin-left:8px;">Download CSV Summary</a>
    </div>

    <hr>

    <!-- FAQ -->
    <div class="faq-section">
      <h3>How to Use This App</h3>
      <details>
        <summary>What does this app do?</summary>
        <p>This app validates your CLIF 2.1 data against the official spec. It flags issues across conformance, completeness, and plausibility — but some "errors" might be totally fine for your site. Your data might not use a certain drug or device, and that's okay.</p>
      </details>
      <details>
        <summary>1. Review Each Table</summary>
        <p>Click through the tables on the Validation page. Check out the validation results and summary stats. See what's flagged.</p>
      </details>
      <details>
        <summary>2. Mark Site-Specific Stuff</summary>
        <p>In the Validation tab, enable "Review Status-Affecting Errors". For each error, decide:</p>
        <ul>
          <li><strong>Accept</strong> — Legit issue, needs fixing</li>
          <li><strong>Reject</strong> — Not a problem for your site (add a reason why)</li>
          <li><strong>Pending</strong> — You'll deal with it later</li>
        </ul>
        <p>Hit "Save Feedback" when done.</p>
      </details>
      <details>
        <summary>3. Regenerate the Combined Report</summary>
        <p>Once you've reviewed, come back to this page and click "Regenerate All Reports". This creates a fresh combined report with your feedback applied.</p>
      </details>
      <details>
        <summary>4. Need to Update Just One Table?</summary>
        <p>Select that table from the table picker above, then click "Analyze All Tables". Just those selected tables get refreshed.</p>
      </details>
      <details>
        <summary>5. Check Out Table One</summary>
        <p>Click "Table One Results" above to explore cohorts, demographics, meds, ventilation, and outcomes — all in one place.</p>
      </details>
      <details>
        <summary>Questions?</summary>
        <p>Hit me up on CLIF Slack — Kaveri Chhikara.</p>
      </details>
    </div>
  `;

  // Hero card navigation
  document.getElementById('hero-tableone').addEventListener('click', () => navigate('tableone'));
  document.getElementById('hero-validation').addEventListener('click', () => navigate('validation'));

  // AI section — only show if LLM available
  initAiAll();

  // Check downloads
  checkDownloads();

  // Table picker
  initTablePicker();

  // Analyze all button
  document.getElementById('btn-analyze-all').addEventListener('click', async () => {
    const genAgg = document.getElementById('chk-aggregates').checked;
    const selected = getSelectedTables();
    const payload = { generate_aggregates: genAgg };
    if (selected) payload.tables = selected;
    try {
      const { task_id } = await api.analyzeAll(payload);
      showProgress(task_id);
    } catch (e) {
      alert('Failed to start analysis: ' + e.message);
    }
  });

  // Regenerate button
  document.getElementById('btn-regenerate').addEventListener('click', async () => {
    const btn = document.getElementById('btn-regenerate');
    btn.disabled = true;
    btn.textContent = 'Regenerating...';
    try {
      await api.regenerateReports();
      btn.textContent = 'Done!';
      setTimeout(() => { btn.textContent = 'Regenerate All Reports'; btn.disabled = false; }, 2000);
      checkDownloads();
    } catch (e) {
      alert('Failed: ' + e.message);
      btn.textContent = 'Regenerate All Reports';
      btn.disabled = false;
    }
  });
}

async function initTablePicker() {
  const toggle = document.getElementById('toggle-table-picker');
  const picker = document.getElementById('table-picker');
  const grid = document.getElementById('picker-grid');

  toggle.addEventListener('click', (e) => {
    e.preventDefault();
    const open = picker.style.display !== 'none';
    picker.style.display = open ? 'none' : 'block';
    toggle.innerHTML = open ? 'Select Tables &#9662;' : 'Select Tables &#9652;';
  });

  try {
    const { tables } = await api.getTables();
    for (const [key, info] of Object.entries(tables)) {
      const label = document.createElement('label');
      label.style.fontSize = '0.85rem';
      label.innerHTML = `<input type="checkbox" class="table-pick" value="${key}" checked> ${info.display_name}`;
      grid.appendChild(label);
    }
  } catch (_) { return; }

  document.getElementById('picker-all').addEventListener('click', (e) => {
    e.preventDefault();
    grid.querySelectorAll('.table-pick').forEach(cb => cb.checked = true);
  });
  document.getElementById('picker-none').addEventListener('click', (e) => {
    e.preventDefault();
    grid.querySelectorAll('.table-pick').forEach(cb => cb.checked = false);
  });
}

/** Returns null if all checked (send nothing = backend default), or array of selected names. */
function getSelectedTables() {
  const boxes = document.querySelectorAll('#picker-grid .table-pick');
  if (!boxes.length) return null;
  const checked = [...boxes].filter(cb => cb.checked).map(cb => cb.value);
  return checked.length === boxes.length ? null : checked;
}

async function initAiAll() {
  const section = document.getElementById('ai-all-section');
  if (!section) return;

  let modelName = '';
  try {
    const { available, model } = await api.getLlmStatus();
    if (!available) return;
    modelName = model || '';
  } catch (e) { return; }

  section.style.display = 'block';

  // Show model badge
  const badge = document.getElementById('ai-model-badge');
  if (badge && modelName) {
    badge.textContent = `via ${modelName} (Ollama)`;
    badge.style.display = 'inline';
  }

  // Restore cached interpretation or fetch fresh
  const cached = state.get('aiInterpretation');
  if (cached) {
    restoreAiResult(cached);
  } else {
    startAiStream();
  }

  // Re-interpret link — always fetches fresh
  document.getElementById('btn-ai-reinterpret').addEventListener('click', (e) => {
    e.preventDefault();
    startAiStream();
  });

  // Expand button — fullscreen overlay
  document.getElementById('btn-ai-expand').addEventListener('click', () => {
    const html = document.getElementById('ai-all-text').innerHTML;
    const overlay = document.createElement('div');
    overlay.className = 'ai-fullpage-overlay';
    overlay.innerHTML = `
      <button class="btn btn-secondary close-btn" id="ai-overlay-close">Close</button>
      <h2>Validation Summary — All Tables</h2>
      <div class="card ai-card"><div class="ai-prose">${html}</div></div>
    `;
    document.body.appendChild(overlay);
    overlay.querySelector('#ai-overlay-close').addEventListener('click', () => overlay.remove());
  });
}

function restoreAiResult(raw) {
  const textEl = document.getElementById('ai-all-text');
  const spinner = document.getElementById('ai-all-spinner');
  const expandBtn = document.getElementById('btn-ai-expand');
  const reinterpretBtn = document.getElementById('btn-ai-reinterpret');

  textEl.innerHTML = typeof marked !== 'undefined' ? marked.parse(raw) : raw;
  spinner.style.display = 'none';
  expandBtn.style.display = 'inline-flex';
  reinterpretBtn.style.display = 'inline';
}

function startAiStream() {
  const textEl = document.getElementById('ai-all-text');
  const spinner = document.getElementById('ai-all-spinner');
  const expandBtn = document.getElementById('btn-ai-expand');
  const reinterpretBtn = document.getElementById('btn-ai-reinterpret');

  textEl.innerHTML = '';
  spinner.style.display = '';
  expandBtn.style.display = 'none';
  reinterpretBtn.style.display = 'none';

  let raw = '';
  api.streamInterpretationAll({
    onChunk(text) {
      raw += text;
      textEl.innerHTML = typeof marked !== 'undefined' ? marked.parse(raw) : raw;
      textEl.scrollTop = textEl.scrollHeight;
    },
    onDone() {
      spinner.style.display = 'none';
      expandBtn.style.display = 'inline-flex';
      reinterpretBtn.style.display = 'inline';
      state.set('aiInterpretation', raw);
    },
    onError(msg) {
      spinner.style.display = 'none';
      if (msg && (msg.includes('404') || msg.toLowerCase().includes('no tables'))) {
        textEl.innerHTML = '<em>No tables analyzed yet. Run analysis first to generate a validation summary.</em>';
      } else {
        textEl.innerHTML = '<em>Error: ' + escapeHtml(msg) + '</em>';
      }
      reinterpretBtn.style.display = 'inline';
    },
  });
}

function escapeHtml(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

function showProgress(taskId) {
  const prog = document.getElementById('bulk-progress');
  prog.style.display = 'block';
  const fill = document.getElementById('progress-fill');
  const text = document.getElementById('progress-text');
  const results = document.getElementById('bulk-results');

  connectSSE(taskId, {
    onProgress(data) {
      fill.style.width = (data.pct || 0) + '%';
      text.textContent = data.message || '';
    },
    onComplete(data) {
      fill.style.width = '100%';
      text.textContent = 'Complete!';
      if (data.results) {
        const r = data.results;
        results.innerHTML = `
          <div style="margin-top:16px;">
            <span class="badge badge-success">${r.success?.length || 0} Successful</span>
            <span class="badge badge-danger" style="margin-left:8px;">${r.failed?.length || 0} Failed</span>
            <span class="badge badge-info" style="margin-left:8px;">${r.skipped?.length || 0} Skipped</span>
          </div>
        `;
      }
      // Refresh table data in state
      api.getTables().then(({ tables }) => {
        state.set('tables', tables);
      });
      checkDownloads();
    },
    onError(data) {
      text.textContent = 'Error: ' + (data.message || 'Unknown error');
      fill.style.backgroundColor = 'var(--danger)';
    },
  });
}

function checkDownloads() {
  const section = document.getElementById('download-section');
  if (!section) return;
  fetch('/api/reports/download/pdf', { method: 'HEAD' }).then(r => {
    if (r.ok) {
      section.style.display = 'block';
      document.getElementById('dl-pdf').href = '/api/reports/download/pdf';
      document.getElementById('dl-csv').href = '/api/reports/download/csv';
    }
  }).catch(() => {});
}
