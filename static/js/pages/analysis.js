import { api } from '../api.js';
import * as state from '../state.js';
import { getParams, navigate } from '../router.js';
import { connectSSE } from '../sse.js';
import { TABLE_DISPLAY_NAMES } from '../components/navbar.js';
import { renderTabs } from '../components/tabs.js';

export async function renderAnalysis(el, params) {
  const table = params.table || state.get('selectedTable') || 'patient';
  state.set('selectedTable', table);

  const displayName = TABLE_DISPLAY_NAMES[table] || table;

  el.innerHTML = `
    <button class="btn btn-back" id="btn-back-validation">&larr; Back to Validation</button>
    <h1>${displayName} Analysis</h1>
    <div style="margin-bottom:16px;display:flex;gap:12px;align-items:center;">
      <button class="btn btn-primary" id="btn-run-analysis">Run Analysis</button>
      <a class="btn btn-outline" id="btn-view-pdf" href="${api.tableReport(table)}" target="_blank" style="display:none;">View PDF Report</a>
      <div id="single-progress" style="display:none;flex:1;">
        <div class="progress-bar"><div class="progress-fill" id="single-progress-fill"></div></div>
        <span id="single-progress-text" style="font-size:0.8rem;opacity:0.7;"></span>
      </div>
    </div>
    <div id="analysis-tabs"></div>
    <div id="analysis-content"></div>
  `;

  // Back to validation
  document.getElementById('btn-back-validation').addEventListener('click', () => navigate('validation'));

  // Run analysis button
  document.getElementById('btn-run-analysis').addEventListener('click', async () => {
    try {
      const { task_id } = await api.analyze(table, { generate_aggregates: true });
      const prog = document.getElementById('single-progress');
      prog.style.display = 'block';
      const fill = document.getElementById('single-progress-fill');
      const text = document.getElementById('single-progress-text');

      connectSSE(task_id, {
        onProgress(data) {
          fill.style.width = (data.pct || 0) + '%';
          text.textContent = data.message || '';
        },
        onComplete() {
          fill.style.width = '100%';
          text.textContent = 'Complete!';
          const pdfBtn = document.getElementById('btn-view-pdf');
          if (pdfBtn) pdfBtn.style.display = '';
          // Reload analysis data
          loadAnalysisData(table, el);
        },
        onError(data) {
          text.textContent = 'Error: ' + (data.message || 'Unknown');
        },
      });
    } catch (e) {
      alert('Failed: ' + e.message);
    }
  });

  // Load existing data
  loadAnalysisData(table, el);
}

async function loadAnalysisData(table, el) {
  const tabsContainer = document.getElementById('analysis-tabs');
  const contentContainer = document.getElementById('analysis-content');

  const tabs = [
    { id: 'validation', label: 'Validation' },
    { id: 'mcide', label: 'MCIDE' },
    { id: 'summary', label: 'Summary' },
  ];

  renderTabs(tabsContainer, contentContainer, tabs, async (tabId, panel) => {
    if (tabId === 'validation') await renderValidation(table, panel);
    else if (tabId === 'mcide') await renderMcide(table, panel);
    else if (tabId === 'summary') await renderSummary(table, panel);
  });
}

async function renderValidation(table, panel) {
  try {
    const data = await api.getValidation(table);

    // Show PDF button if validation data exists
    const pdfBtn = document.getElementById('btn-view-pdf');
    if (pdfBtn) pdfBtn.style.display = '';

    // DQA Score hero
    let html = `
      <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:8px;">
        <span style="font-size:2.2rem;font-weight:700;">${data.overall_pct}%</span>
        <span style="font-size:0.85rem;opacity:0.5;">${data.total_passed}/${data.total_checks} checks passed</span>
      </div>
      <div class="progress-bar" style="margin-bottom:16px;"><div class="progress-fill" style="width:${data.overall_pct}%"></div></div>
    `;

    // Metric cards
    html += `<div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap;">`;
    html += `<div class="card metric-card"><div class="metric-label">Errors</div><div class="metric-value" style="color:${data.error_count > 0 ? 'var(--danger)' : 'var(--success)'}">${data.error_count}</div></div>`;
    html += `<div class="card metric-card"><div class="metric-label">Warnings</div><div class="metric-value" style="color:${data.warning_count > 0 ? 'var(--warning)' : 'var(--success)'}">${data.warning_count}</div></div>`;

    for (const [cat, scores] of Object.entries(data.category_scores)) {
      const pct = scores.total ? Math.round(scores.passed / scores.total * 100) : 100;
      const color = pct >= 85 ? 'var(--success)' : pct >= 60 ? 'var(--warning)' : 'var(--danger)';
      html += `<div class="card metric-card"><div class="metric-label">${cat.charAt(0).toUpperCase() + cat.slice(1)}</div><div class="metric-value" style="color:${color}">${scores.passed}/${scores.total}</div></div>`;
    }
    html += `</div>`;

    // Review status placeholder (populated after feedback loads)
    html += `<div id="review-status"></div>`;

    // Issues table (errors)
    const errors = data.issues.filter(i => i.severity === 'error');
    if (errors.length > 0) {
      html += `<div class="card" style="margin-bottom:16px;">
        <h3>Review & Resolve (${errors.length} error(s))</h3>
        <div style="display:flex;justify-content:space-evenly;font-size:0.8rem;margin:0 0 12px;padding:8px 12px;background:var(--bg-secondary,#f5f5f7);border-radius:6px;">
          <span><strong style="color:var(--text);">Pending</strong> <span style="opacity:0.6;">— not reviewed</span></span>
          <span><strong style="color:var(--success);">Accepted</strong> <span style="opacity:0.6;">— known issue, acknowledged</span></span>
          <span><strong style="color:var(--danger);">Rejected</strong> <span style="opacity:0.6;">— incorrect or not applicable</span></span>
        </div>
        <div id="feedback-container"></div>
        <table class="data-table" id="errors-table">
          <thead><tr><th>Feedback</th><th class="reason-col" style="display:none;">Reason</th><th>Category</th><th>Check</th><th>Column</th><th>Message</th></tr></thead>
          <tbody>`;

      for (const issue of errors) {
        const eid = issue.error_id || '';
        html += `<tr data-eid="${eid}">
          <td>
            <select class="feedback-select" data-eid="${eid}">
              <option value="pending">Pending</option>
              <option value="accepted">Accepted</option>
              <option value="rejected">Rejected</option>
            </select>
          </td>
          <td class="reason-col" style="display:none;">
            <textarea class="feedback-reason" data-eid="${eid}" placeholder="Reason..." rows="1" style="display:none;"></textarea>
          </td>
          <td>${issue.category || ''}</td>
          <td>${issue.rule_description || issue.check_type || ''}</td>
          <td>${issue.column_field || 'N/A'}</td>
          <td>${issue.message || ''}</td>
        </tr>`;
      }

      html += `</tbody></table>
        <button class="btn btn-primary" id="btn-save-feedback" style="margin-top:12px;">Save Feedback</button>
      </div>`;
    } else {
      html += `<div class="card" style="border-left:4px solid var(--success);"><p>No validation issues found!</p></div>`;
    }

    // Warnings — grouped by rule_description with sparklines
    const warnings = data.issues.filter(i => i.severity === 'warning');
    if (warnings.length > 0) {
      const groups = {};
      for (const w of warnings) {
        const key = w.rule_description || w.check_type || 'Other';
        (groups[key] = groups[key] || []).push(w);
      }
      html += `<div class="card" style="margin-top:12px;">
        <h3>Warnings (${warnings.length})</h3>`;
      let first = true;
      for (const [groupName, items] of Object.entries(groups)) {
        html += `<details${first ? ' open' : ''} style="margin-bottom:8px;">
          <summary style="cursor:pointer;font-weight:600;padding:6px 0;">${groupName} <span style="background:var(--warning);color:#000;border-radius:10px;padding:1px 8px;font-size:0.85em;font-weight:500;margin-left:6px;">${items.length}</span></summary>
          <table class="data-table" style="margin:4px 0 8px 16px;">
            <thead><tr><th>Column</th><th>Finding</th></tr></thead>
            <tbody>`;
        for (const w of items) {
          const finding = w.finding || w.message || '';
          const yearly = w.details && w.details.yearly_counts;
          let sparkHtml = '';
          if (yearly) {
            const years = Object.keys(yearly).sort();
            const maxCount = Math.max(...Object.values(yearly)) || 1;
            sparkHtml = `<div class="spark-bar" title="${years.map(y => y + ': ' + yearly[y]).join(', ')}">`;
            for (const y of years) {
              const count = yearly[y];
              const pct = count > 0 ? Math.max(10, Math.round((count / maxCount) * 100)) : 100;
              const color = count > 0 ? 'var(--info)' : 'var(--danger)';
              sparkHtml += `<span class="spark-bar-col" style="height:${pct}%;background:${color};"></span>`;
            }
            sparkHtml += `<span class="spark-bar-labels"><span>${years[0]}</span><span>${years[years.length - 1]}</span></span></div>`;
          }
          html += `<tr><td>${w.column_field || 'N/A'}</td><td>${finding}${sparkHtml}</td></tr>`;
        }
        html += `</tbody></table></details>`;
        first = false;
      }
      html += `</div>`;
    }

    panel.innerHTML = html;

    // Show/hide the Reason column based on whether any row is rejected
    const toggleReasonColumn = () => {
      const hasRejected = [...panel.querySelectorAll('.feedback-select')].some(s => s.value === 'rejected');
      const display = hasRejected ? '' : 'none';
      panel.querySelectorAll('.reason-col').forEach(el => el.style.display = display);
    };

    // Load existing feedback decisions and reasons
    try {
      const feedback = await api.getFeedback(table);
      if (feedback && feedback.user_decisions) {
        for (const [eid, info] of Object.entries(feedback.user_decisions)) {
          const sel = panel.querySelector(`.feedback-select[data-eid="${eid}"]`);
          if (sel) sel.value = info.decision || 'pending';
          const reasonInput = panel.querySelector(`.feedback-reason[data-eid="${eid}"]`);
          if (reasonInput) {
            if (info.decision === 'rejected') {
              reasonInput.style.display = 'block';
              reasonInput.value = info.reason || '';
              reasonInput.style.height = 'auto';
              reasonInput.style.height = reasonInput.scrollHeight + 'px';
            }
          }
        }
        updateStatsFromFeedback(panel, data, feedback);
      }

      // Render review status
      const statusEl = panel.querySelector('#review-status');
      if (statusEl && feedback) {
        const hasDecisions = feedback.accepted_count > 0 || feedback.rejected_count > 0;
        if (hasDecisions && feedback.timestamp) {
          const ts = new Date(feedback.timestamp);
          const dateStr = ts.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
          const timeStr = ts.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
          const parts = [];
          if (feedback.accepted_count) parts.push(`${feedback.accepted_count} accepted`);
          if (feedback.rejected_count) parts.push(`${feedback.rejected_count} rejected`);
          if (feedback.pending_count) parts.push(`${feedback.pending_count} pending`);
          statusEl.innerHTML = `<div class="review-status-bar">
            <span>Last reviewed ${dateStr} at ${timeStr}</span>
            <span class="review-status-counts">${parts.join(' · ')}</span>
          </div>`;
        }
      }

      toggleReasonColumn();
    } catch (e) { /* no feedback yet */ }

    // Feedback change handlers — update stats live on change
    panel.querySelectorAll('.feedback-select').forEach(sel => {
      sel.addEventListener('change', async () => {
        const eid = sel.dataset.eid;
        const reasonInput = panel.querySelector(`.feedback-reason[data-eid="${eid}"]`);
        if (sel.value === 'rejected') {
          reasonInput.style.display = 'block';
        } else {
          reasonInput.style.display = 'none';
          reasonInput.value = '';
        }
        toggleReasonColumn();
        try {
          await api.putFeedback(table, eid, sel.value, reasonInput.value);
          const fb = await api.getFeedback(table);
          updateStatsFromFeedback(panel, data, fb);
        } catch (e) { console.error(e); }
      });
    });

    // Reason textarea handlers — auto-resize, save on blur or Enter
    const autoResize = (el) => { el.style.height = 'auto'; el.style.height = el.scrollHeight + 'px'; };
    panel.querySelectorAll('.feedback-reason').forEach(input => {
      const save = async () => {
        const eid = input.dataset.eid;
        const sel = panel.querySelector(`.feedback-select[data-eid="${eid}"]`);
        try {
          await api.putFeedback(table, eid, sel.value, input.value);
        } catch (e) { console.error(e); }
      };
      input.addEventListener('input', () => autoResize(input));
      input.addEventListener('blur', save);
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); input.blur(); }
      });
    });

    // Save feedback button
    const saveBtn = panel.querySelector('#btn-save-feedback');
    if (saveBtn) {
      saveBtn.addEventListener('click', async () => {
        try {
          await api.saveFeedback(table);
          saveBtn.textContent = 'Saved!';
          setTimeout(() => saveBtn.textContent = 'Save Feedback', 2000);
          // Refresh review status bar
          const fb = await api.getFeedback(table);
          const statusEl = panel.querySelector('#review-status');
          if (statusEl && fb) {
            const now = new Date();
            const dateStr = now.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
            const timeStr = now.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
            const parts = [];
            if (fb.accepted_count) parts.push(`${fb.accepted_count} accepted`);
            if (fb.rejected_count) parts.push(`${fb.rejected_count} rejected`);
            if (fb.pending_count) parts.push(`${fb.pending_count} pending`);
            statusEl.innerHTML = `<div class="review-status-bar">
              <span>Last reviewed ${dateStr} at ${timeStr}</span>
              <span class="review-status-counts">${parts.join(' · ')}</span>
            </div>`;
          }
        } catch (e) { alert('Save failed: ' + e.message); }
      });
    }

  } catch (e) {
    panel.innerHTML = `<p>No validation results available. Click "Run Analysis" to validate this table.</p>`;
  }
}

function updateStatsFromFeedback(panel, data, feedback) {
  if (!feedback || !feedback.user_decisions) return;

  const decisions = feedback.user_decisions;
  // Rejected = "not a real error" → resolved, counts as passed
  // Accepted = "confirmed error" → still an error
  // Pending = unresolved → still an error
  const rejectedIds = new Set(
    Object.entries(decisions)
      .filter(([, d]) => d.decision === 'rejected')
      .map(([eid]) => eid)
  );

  // Recount errors/warnings excluding rejected
  const adjErrors = data.issues.filter(i => i.severity === 'error' && !rejectedIds.has(i.error_id));
  const adjWarnings = data.issues.filter(i => i.severity === 'warning' && !rejectedIds.has(i.error_id));
  const rejectedCount = data.issues.filter(i => i.severity === 'error' && rejectedIds.has(i.error_id)).length;

  // Adjusted scores: rejected errors become "passed"
  const adjPassed = data.total_passed + rejectedCount;
  const adjPct = data.total_checks ? Math.round(adjPassed / data.total_checks * 1000) / 10 : 100;

  // Update hero stats
  const metricCards = panel.querySelectorAll('.metric-card');
  const heroLine = panel.querySelector('[style*="font-size:2.2rem"]');
  if (heroLine) {
    heroLine.parentElement.innerHTML = `
      <span style="font-size:2.2rem;font-weight:700;">${adjPct}%</span>
      <span style="font-size:0.85rem;opacity:0.5;">${adjPassed}/${data.total_checks} checks passed</span>
    `;
  }

  // Update progress bar
  const bar = panel.querySelector('.progress-fill');
  if (bar) bar.style.width = adjPct + '%';

  // Update error/warning metric cards
  if (metricCards.length >= 2) {
    const errVal = metricCards[0].querySelector('.metric-value');
    if (errVal) {
      errVal.textContent = adjErrors.length;
      errVal.style.color = adjErrors.length > 0 ? 'var(--danger)' : 'var(--success)';
    }
    const warnVal = metricCards[1].querySelector('.metric-value');
    if (warnVal) {
      warnVal.textContent = adjWarnings.length;
      warnVal.style.color = adjWarnings.length > 0 ? 'var(--warning)' : 'var(--success)';
    }
  }

  // Update per-category cards
  let cardIdx = 2;
  for (const [cat, scores] of Object.entries(data.category_scores)) {
    const catRejected = data.issues.filter(
      i => i.category === cat && i.severity === 'error' && rejectedIds.has(i.error_id)
    ).length;
    const adjCatPassed = scores.passed + catRejected;
    const pct = scores.total ? Math.round(adjCatPassed / scores.total * 100) : 100;
    const color = pct >= 85 ? 'var(--success)' : pct >= 60 ? 'var(--warning)' : 'var(--danger)';
    if (metricCards[cardIdx]) {
      const val = metricCards[cardIdx].querySelector('.metric-value');
      if (val) {
        val.textContent = `${adjCatPassed}/${scores.total}`;
        val.style.color = color;
      }
    }
    cardIdx++;
  }
}

async function renderMcide(table, panel) {
  try {
    const data = await api.getMcide(table);
    let html = '';

    if (data.mcide_files && data.mcide_files.length > 0) {
      html += `<h3>MCIDE Value Counts</h3>`;
      for (const file of data.mcide_files) {
        if (file.error) {
          html += `<p class="text-danger">Error: ${file.error}</p>`;
          continue;
        }
        html += `<div class="card" style="margin-bottom:12px;">
          <h4>${file.name.replace(/_/g, ' ')}</h4>
          <div style="display:flex;gap:16px;margin-bottom:8px;">
            <span>Unique: ${file.row_count}</span>
            ${file.total_n !== null ? `<span>Total N: ${file.total_n.toLocaleString()}</span>` : ''}
          </div>
          <div class="table-wrapper" id="mcide-${file.name}"></div>
        </div>`;
      }
    }

    if (data.stats_files && data.stats_files.length > 0) {
      html += `<h3>Summary Statistics</h3>`;
      for (const sf of data.stats_files) {
        if (sf.error) continue;
        html += `<div class="card" style="margin-bottom:12px;">
          <h4>${sf.name.replace(/_/g, ' ')}</h4>
          <div class="table-wrapper" id="stats-${sf.name}"></div>
        </div>`;
      }
    }

    if (!html) html = '<p>No MCIDE data available. Run analysis first.</p>';
    panel.innerHTML = html;

    // Initialize DataTables for MCIDE files
    if (typeof jQuery !== 'undefined' && jQuery.fn.DataTable) {
      for (const file of (data.mcide_files || [])) {
        if (file.error || !file.data || file.data.length === 0) continue;
        const container = panel.querySelector(`#mcide-${file.name}`);
        if (!container) continue;
        const tableEl = document.createElement('table');
        tableEl.className = 'display';
        tableEl.style.width = '100%';
        container.appendChild(tableEl);
        jQuery(tableEl).DataTable({
          data: file.data,
          columns: file.columns.map(c => ({ data: c, title: c })),
          pageLength: 20,
          order: file.columns.includes('N') ? [[file.columns.indexOf('N'), 'desc']] : [],
          scrollX: true,
        });
      }
      for (const sf of (data.stats_files || [])) {
        if (sf.error || !sf.data || !Array.isArray(sf.data) || sf.data.length === 0) continue;
        const container = panel.querySelector(`#stats-${sf.name}`);
        if (!container) continue;
        const cols = Object.keys(sf.data[0]);
        const tableEl = document.createElement('table');
        tableEl.className = 'display';
        tableEl.style.width = '100%';
        container.appendChild(tableEl);
        jQuery(tableEl).DataTable({
          data: sf.data,
          columns: cols.map(c => ({ data: c, title: c })),
          pageLength: 20,
          scrollX: true,
        });
      }
    }
  } catch (e) {
    panel.innerHTML = '<p>No MCIDE data available.</p>';
  }
}

async function renderSummary(table, panel) {
  try {
    const summary = await api.getSummary(table);
    let html = '';

    // Data overview
    const info = summary.data_info || {};
    html += `<h3>Data Overview</h3>
      <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;">`;

    if (info.row_count !== undefined) html += `<div class="card metric-card"><div class="metric-label">Total Rows</div><div class="metric-value">${info.row_count.toLocaleString()}</div></div>`;
    if (info.unique_hospitalizations !== undefined) html += `<div class="card metric-card"><div class="metric-label">Unique Hospitalizations</div><div class="metric-value">${info.unique_hospitalizations.toLocaleString()}</div></div>`;
    if (info.unique_patients !== undefined && info.unique_patients > 0) html += `<div class="card metric-card"><div class="metric-label">Unique Patients</div><div class="metric-value">${info.unique_patients.toLocaleString()}</div></div>`;
    if (info.column_count !== undefined) html += `<div class="card metric-card"><div class="metric-label">Total Columns</div><div class="metric-value">${info.column_count}</div></div>`;
    html += `</div>`;

    // Missingness
    const miss = summary.missingness || {};
    if (miss.total_columns) {
      html += `<h3>Missingness Analysis</h3>
        <div style="display:flex;gap:12px;margin-bottom:16px;">
          <div class="card metric-card"><div class="metric-label">Complete Columns</div><div class="metric-value">${miss.complete_columns_count || 0}/${miss.total_columns}</div></div>
          <div class="card metric-card"><div class="metric-label">Overall Missing %</div><div class="metric-value">${(miss.overall_missing_percentage || 0).toFixed(2)}%</div></div>
          <div class="card metric-card"><div class="metric-label">Complete Rows %</div><div class="metric-value">${(miss.complete_rows_percentage || 0).toFixed(2)}%</div></div>
        </div>`;

      html += `<div id="missingness-chart" style="margin-bottom:16px;"></div>`;
    }

    // Distribution charts placeholder
    html += `<div id="distribution-charts"></div>`;

    panel.innerHTML = html;

    // Render missingness chart
    if (miss.columns_with_missing && miss.columns_with_missing.length > 0 && typeof Plotly !== 'undefined') {
      try {
        const chartData = await api.getChart(table, 'missingness');
        if (chartData.data && chartData.data.length > 0) {
          Plotly.newPlot('missingness-chart', chartData.data, chartData.layout, { responsive: true });
        }
      } catch (e) { /* no chart */ }
    }

    // Render distribution charts
    const distContainer = panel.querySelector('#distribution-charts');
    if (distContainer && typeof Plotly !== 'undefined') {
      try {
        const charts = await api.getChart(table, 'distribution');
        if (charts && typeof charts === 'object') {
          for (const [key, chart] of Object.entries(charts)) {
            const div = document.createElement('div');
            div.style.marginBottom = '16px';
            distContainer.appendChild(div);
            Plotly.newPlot(div, chart.data, chart.layout, { responsive: true });
          }
        }
      } catch (e) { /* no distributions */ }
    }

  } catch (e) {
    panel.innerHTML = '<p>No summary data available. Click "Run Analysis" to generate.</p>';
  }
}
