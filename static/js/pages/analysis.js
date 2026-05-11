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
    <h1>${displayName} Validation</h1>
    <div style="margin-bottom:16px;display:flex;gap:12px;align-items:center;">
      <button class="btn btn-primary" id="btn-run-analysis">Run Validation</button>
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

    // Always render all three category cards so absent tables show N/A
    // for Completeness / Plausibility instead of hiding the cards.
    const CATS = ['conformance', 'completeness', 'plausibility'];
    for (const cat of CATS) {
      const scores = (data.category_scores || {})[cat];
      const label = cat.charAt(0).toUpperCase() + cat.slice(1);
      if (!scores || scores.total == null) {
        html += `<div class="card metric-card"><div class="metric-label">${label}</div><div class="metric-value" style="color:var(--text-muted,#888);">N/A</div></div>`;
        continue;
      }
      const pct = scores.total ? Math.round(scores.passed / scores.total * 100) : 100;
      const color = pct >= 85 ? 'var(--success)' : pct >= 60 ? 'var(--warning)' : 'var(--danger)';
      html += `<div class="card metric-card"><div class="metric-label">${label}</div><div class="metric-value" style="color:${color}">${scores.passed}/${scores.total}</div></div>`;
    }
    html += `</div>`;

    // Review status placeholder (populated after feedback loads)
    html += `<div id="review-status"></div>`;

    // Issues table (errors). For absent tables the single table_presence
    // error is a factual state (site didn't submit the table), not an
    // actionable finding — hide the feedback dropdown, legend, and
    // Save Feedback button; the row just stays visible.
    const errors = data.issues.filter(i => i.severity === 'error');
    const errorAtoms = errors.reduce((s, i) => s + (i.atomic_count ?? 1), 0);
    if (errors.length > 0) {
      const isAbsent = !!data.absent;
      html += `<div class="card" style="margin-bottom:16px;">
        <h3>Review & Resolve (${errorAtoms} error(s))</h3>`;
      if (!isAbsent) {
        html += `<div style="display:flex;justify-content:space-evenly;font-size:0.8rem;margin:0 0 12px;padding:8px 12px;background:var(--bg-secondary,#f5f5f7);border-radius:6px;">
          <span><strong style="color:var(--text);">Pending</strong> <span style="opacity:0.6;">— not reviewed</span></span>
          <span><strong style="color:var(--success);">Accepted</strong> <span style="opacity:0.6;">— known issue, acknowledged</span></span>
          <span><strong style="color:var(--danger);">Rejected</strong> <span style="opacity:0.6;">— incorrect or not applicable</span></span>
        </div>`;
      }
      html += `<div id="feedback-container"></div>`;
      html += `<table class="data-table" id="errors-table">
          <thead><tr>
            <th>Feedback</th><th class="reason-col" style="display:none;">Reason</th><th>Category</th><th>Check</th><th>Column</th><th>Message</th><th style="text-align:right;">Checks</th>
          </tr></thead>
          <tbody>`;

      for (const issue of errors) {
        const eid = issue.error_id || '';
        const checks = issue.atomic_count ?? 1;
        const checksDisplay = checks === 0 ? '—' : checks;
        const subValues = issue.details && Array.isArray(issue.details.missing_values)
          ? issue.details.missing_values : null;
        const isMulti = !isAbsent
          && issue.check_type === 'mcide_value_coverage'
          && subValues && subValues.length > 1;
        const feedbackCell = isAbsent
          ? `<td style="color:var(--text-muted,#888);">—</td>`
          : isMulti
            ? `<td>
                <select class="mcide-action-select" data-eid="${eid}">
                  <option value="">— Action —</option>
                  <option value="accepted">Accept selected</option>
                  <option value="rejected">Reject selected</option>
                  <option value="pending">Set to pending</option>
                </select>
                <div style="margin-top:6px;"><span class="parent-summary" data-eid="${eid}" style="font-size:0.85em;"></span></div>
               </td>
               <td class="reason-col" style="display:none;"></td>`
            : `<td>
              <select class="feedback-select" data-eid="${eid}">
                <option value="pending">Pending</option>
                <option value="accepted">Accepted</option>
                <option value="rejected">Rejected</option>
              </select>
            </td>
            <td class="reason-col" style="display:none;">
              <textarea class="feedback-reason" data-eid="${eid}" placeholder="Reason..." rows="1" style="display:none;"></textarea>
            </td>`;
        let messageCell;
        if (isMulti) {
          // Render the missing values as wrapping chips inside the Message
          // cell. Each chip is a checkbox + label with the value text.
          // Decision state is shown via chip classes (decision-rejected /
          // decision-accepted) applied on feedback load + after bulk apply.
          const intro = String(issue.message || '').split(':')[0] + ':';
          let chipsHtml = '';
          for (const v of subValues) {
            const vAttr = String(v).replace(/"/g, '&quot;').replace(/</g, '&lt;');
            const vText = String(v).replace(/</g, '&lt;');
            chipsHtml += `<label class="mcide-chip" data-eid="${eid}" data-value-key="${vAttr}"
                style="display:inline-flex;align-items:center;gap:6px;padding:2px 10px;border-radius:12px;
                       background:rgba(127,127,127,0.1);cursor:pointer;font-size:0.88em;line-height:1.5;">
              <input type="checkbox" class="bulk-select" data-eid="${eid}" data-value-key="${vAttr}" style="margin:0;">
              <span class="chip-label">${vText}</span>
            </label>`;
          }
          messageCell = `<td>
            <div style="margin-bottom:8px;">
              <strong>${intro}</strong>
              <a href="#" class="mcide-select-all" data-eid="${eid}" style="margin-left:10px;font-size:0.85em;">select all</a>
              <span style="opacity:0.4;font-size:0.85em;"> · </span>
              <a href="#" class="mcide-deselect-all" data-eid="${eid}" style="font-size:0.85em;">deselect all</a>
            </div>
            <div class="mcide-chips" data-eid="${eid}" style="display:flex;flex-wrap:wrap;gap:6px 8px;">
              ${chipsHtml}
            </div>
          </td>`;
        } else {
          messageCell = `<td>${issue.message || ''}</td>`;
        }
        html += `<tr data-eid="${eid}"${isMulti ? ' data-multi="1"' : ''}>
          ${feedbackCell}
          <td>${issue.category || ''}</td>
          <td>${issue.rule_description || issue.check_type || ''}</td>
          <td>${issue.column_field || 'N/A'}</td>
          ${messageCell}
          <td style="text-align:right;">${checksDisplay}</td>
        </tr>`;
      }

      html += `</tbody></table>`;
      if (!isAbsent) {
        html += `<button class="btn btn-primary" id="btn-save-feedback" style="margin-top:12px;">Save Feedback</button>`;
      }
      html += `</div>`;
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
      const warningAtoms = warnings.reduce((s, w) => s + (w.atomic_count ?? 1), 0);
      html += `<div class="card" style="margin-top:12px;">
        <h3>Warnings (${warningAtoms})</h3>`;
      let first = true;
      for (const [groupName, items] of Object.entries(groups)) {
        const groupChecks = items.reduce((sum, w) => sum + (w.atomic_count ?? 1), 0);
        html += `<details${first ? ' open' : ''} style="margin-bottom:8px;">
          <summary style="cursor:pointer;font-weight:600;padding:6px 0;">${groupName} <span style="background:var(--warning);color:#000;border-radius:10px;padding:1px 8px;font-size:0.85em;font-weight:500;margin-left:6px;">${groupChecks}</span></summary>
          <table class="data-table" style="margin:4px 0 8px 16px;">
            <thead><tr><th>Column</th><th>Finding</th><th style="text-align:right;">Checks</th></tr></thead>
            <tbody>`;
        for (const w of items) {
          const finding = w.finding || w.message || '';
          const checks = w.atomic_count ?? 1;
          const checksDisplay = checks === 0 ? '—' : checks;
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
          html += `<tr><td>${w.column_field || 'N/A'}</td><td>${finding}${sparkHtml}</td><td style="text-align:right;">${checksDisplay}</td></tr>`;
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

    // "Unsaved changes" indicator on the Save Feedback button.
    // Tracked per page-view: any successful putFeedback marks dirty;
    // saveFeedback clears it. Refreshing the page resets the indicator
    // (session pending_feedback may still hold unsaved changes though —
    // that's a narrow gap, fine for a UX hint).
    const markDirty = () => {
      const btn = panel.querySelector('#btn-save-feedback');
      if (!btn || btn.dataset.dirty === '1') return;
      btn.dataset.dirty = '1';
      btn.textContent = '● Save Feedback';
      btn.style.background = 'var(--warning, #f59e0b)';
      btn.style.color = '#000';
      btn.title = 'Unsaved changes — click to persist and regenerate the PDF report';
    };
    const markClean = () => {
      const btn = panel.querySelector('#btn-save-feedback');
      if (!btn) return;
      delete btn.dataset.dirty;
      btn.style.background = '';
      btn.style.color = '';
      btn.title = '';
      // Text is reset by the saveBtn click handler itself (to "Saved!" then "Save Feedback")
    };

    // Find a feedback-select or reason textarea by (eid, value_key). Using
    // dataset lookups (instead of attribute selectors) avoids the need to
    // escape arbitrary MCIDE value strings.
    const findFeedbackEl = (cls, eid, valueKey) => {
      for (const el of panel.querySelectorAll(`.${cls}`)) {
        if (el.dataset.eid !== eid) continue;
        const elKey = el.dataset.valueKey || null;
        if (elKey === (valueKey || null)) return el;
      }
      return null;
    };

    // Update the parent aggregate summary for a multi-value MCIDE row.
    // Reads chip classes (decision-rejected / decision-accepted) since
    // chips are now the source of truth for per-value state.
    const updateParentSummary = (parentEid) => {
      const summaryEl = panel.querySelector(`.parent-summary[data-eid="${parentEid}"]`);
      if (!summaryEl) return;
      const chips = panel.querySelectorAll(`.mcide-chip[data-eid="${parentEid}"]`);
      let accepted = 0, rejected = 0, pending = 0;
      chips.forEach(c => {
        if (c.classList.contains('decision-rejected')) rejected++;
        else if (c.classList.contains('decision-accepted')) accepted++;
        else pending++;
      });
      const total = accepted + rejected + pending;
      summaryEl.innerHTML =
        `<span style="color:var(--danger);">${rejected}</span> rejected · ` +
        `<span style="color:var(--success);">${accepted}</span> accepted · ` +
        `<span style="opacity:0.6;">${pending}</span> pending ` +
        `<span style="opacity:0.5;">(of ${total})</span>`;
    };

    // Apply decision classes + tooltips to every chip from a feedback dict.
    const paintChipDecisions = (feedback) => {
      const decisions = (feedback && feedback.user_decisions) || {};
      panel.querySelectorAll('.mcide-chip').forEach(chip => {
        const eid = chip.dataset.eid;
        const valueKey = chip.dataset.valueKey;
        const entry = decisions[eid];
        const sub = entry && entry.value_decisions && entry.value_decisions[valueKey];
        const decision = (sub && sub.decision) || 'pending';
        const reason = (sub && sub.reason) || '';
        chip.classList.remove('decision-rejected', 'decision-accepted');
        const label = chip.querySelector('.chip-label');
        if (decision === 'rejected') {
          chip.classList.add('decision-rejected');
          if (label) {
            label.style.color = 'var(--danger)';
            label.style.textDecoration = 'line-through';
          }
          chip.title = reason ? `Rejected: ${reason}` : 'Rejected';
        } else if (decision === 'accepted') {
          chip.classList.add('decision-accepted');
          if (label) {
            label.style.color = 'var(--success)';
            label.style.textDecoration = '';
          }
          chip.title = reason ? `Accepted: ${reason}` : 'Accepted';
        } else {
          if (label) {
            label.style.color = '';
            label.style.textDecoration = '';
          }
          chip.title = '';
        }
      });
      // Refresh parent summaries for all multi-value rows
      panel.querySelectorAll('.parent-summary').forEach(el => {
        updateParentSummary(el.dataset.eid);
      });
    };

    // Select-all / deselect-all links for each multi-value MCIDE row
    panel.querySelectorAll('.mcide-select-all').forEach(a => {
      a.addEventListener('click', (ev) => {
        ev.preventDefault();
        const eid = a.dataset.eid;
        panel.querySelectorAll(`.mcide-chip[data-eid="${eid}"] input.bulk-select`)
          .forEach(cb => { cb.checked = true; });
      });
    });
    panel.querySelectorAll('.mcide-deselect-all').forEach(a => {
      a.addEventListener('click', (ev) => {
        ev.preventDefault();
        const eid = a.dataset.eid;
        panel.querySelectorAll(`.mcide-chip[data-eid="${eid}"] input.bulk-select`)
          .forEach(cb => { cb.checked = false; });
      });
    });

    // Load existing feedback decisions and reasons
    try {
      const feedback = await api.getFeedback(table);
      if (feedback && feedback.user_decisions) {
        for (const [eid, info] of Object.entries(feedback.user_decisions)) {
          const hasSubs = info.value_decisions && typeof info.value_decisions === 'object'
            && Object.keys(info.value_decisions).length > 0;
          if (hasSubs) continue;  // Multi-value chips handled by paintChipDecisions below
          const sel = findFeedbackEl('feedback-select', eid, null);
          if (sel) sel.value = info.decision || 'pending';
          const reasonInput = findFeedbackEl('feedback-reason', eid, null);
          if (reasonInput && info.decision === 'rejected') {
            reasonInput.style.display = 'block';
            reasonInput.value = info.reason || '';
          }
        }
        // Apply chip classes + tooltips for all multi-value MCIDE rows,
        // then refresh parent aggregate summaries.
        paintChipDecisions(feedback);
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

      // Reveal the Reason column first, then auto-size textareas. Measuring
      // scrollHeight while the parent <td> is still display:none returns 0
      // and collapses the textarea to a sliver.
      toggleReasonColumn();
      panel.querySelectorAll('.feedback-reason').forEach(el => {
        if (el.style.display !== 'none' && el.value) {
          el.style.height = 'auto';
          el.style.height = el.scrollHeight + 'px';
        }
      });
    } catch (e) { /* no feedback yet */ }

    // Feedback change handlers — fire for single-value / non-MCIDE errors
    // (multi-value MCIDE decisions come via chip bulk actions).
    panel.querySelectorAll('.feedback-select').forEach(sel => {
      sel.addEventListener('change', async () => {
        const eid = sel.dataset.eid;
        const valueKey = sel.dataset.valueKey || null;
        const reasonInput = findFeedbackEl('feedback-reason', eid, valueKey);
        if (reasonInput) {
          if (sel.value === 'rejected') {
            reasonInput.style.display = 'block';
          } else {
            reasonInput.style.display = 'none';
            reasonInput.value = '';
          }
        }
        toggleReasonColumn();
        try {
          await api.putFeedback(
            table, eid, sel.value,
            reasonInput ? reasonInput.value : '', valueKey,
          );
          markDirty();
          if (valueKey) updateParentSummary(eid);
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
        const valueKey = input.dataset.valueKey || null;
        const sel = findFeedbackEl('feedback-select', eid, valueKey);
        try {
          await api.putFeedback(
            table, eid, sel ? sel.value : 'pending', input.value, valueKey,
          );
          markDirty();
        } catch (e) { console.error(e); }
      };
      input.addEventListener('input', () => autoResize(input));
      input.addEventListener('blur', save);
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); input.blur(); }
      });
    });

    // Per-row MCIDE action dropdown. Changing it applies the chosen decision
    // to every chip currently checked under this parent error_id.
    panel.querySelectorAll('.mcide-action-select').forEach(sel => {
      sel.addEventListener('change', async () => {
        const decision = sel.value;
        if (!decision) return;
        const eid = sel.dataset.eid;
        const checked = [...panel.querySelectorAll(
          `.mcide-chip[data-eid="${eid}"] input.bulk-select:checked`
        )];
        if (checked.length === 0) {
          sel.value = '';
          return;
        }
        try {
          await Promise.all(checked.map(cb =>
            api.putFeedback(table, eid, decision, '', cb.dataset.valueKey)
          ));
          markDirty();
          const fb = await api.getFeedback(table);
          paintChipDecisions(fb);
          updateStatsFromFeedback(panel, data, fb);
          checked.forEach(cb => { cb.checked = false; });
        } catch (e) {
          console.error(e);
          alert('Update failed: ' + e.message);
        }
        sel.value = '';
      });
    });

    // Save feedback button
    const saveBtn = panel.querySelector('#btn-save-feedback');
    if (saveBtn) {
      saveBtn.addEventListener('click', async () => {
        try {
          await api.saveFeedback(table);
          markClean();
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
    panel.innerHTML = `<p>No validation results available. Click "Run Validation" to validate this table.</p>`;
  }
}

function updateStatsFromFeedback(panel, data, feedback) {
  if (!feedback || !feedback.user_decisions) return;

  const decisions = feedback.user_decisions;

  // Atom-level rejection count for one issue. Multi-value MCIDE errors may
  // be partially rejected (e.g. 15 of 21 missing languages) — in that case
  // only the rejected sub-values count as "passed".
  const issueRejectedAtoms = (issue) => {
    const entry = decisions[issue.error_id];
    if (!entry) return 0;
    const subs = entry.value_decisions;
    const vals = issue.details && issue.details.missing_values;
    const isMulti = subs && typeof subs === 'object'
      && Array.isArray(vals) && vals.length > 1
      && issue.check_type === 'mcide_value_coverage';
    if (isMulti) {
      let n = 0;
      for (const v of vals) {
        if (subs[v] && subs[v].decision === 'rejected') n++;
      }
      return n;
    }
    return entry.decision === 'rejected' ? (issue.atomic_count ?? 1) : 0;
  };

  // Recount errors/warnings using atom-level partial credit
  const errorIssues = data.issues.filter(i => i.severity === 'error');
  const warningIssues = data.issues.filter(i => i.severity === 'warning');
  const rejectedAtoms = errorIssues.reduce((s, i) => s + issueRejectedAtoms(i), 0);
  const adjErrorAtoms = errorIssues.reduce(
    (s, i) => s + Math.max(0, (i.atomic_count ?? 1) - issueRejectedAtoms(i)), 0,
  );
  const adjWarningAtoms = warningIssues.reduce(
    (s, i) => s + Math.max(0, (i.atomic_count ?? 1) - issueRejectedAtoms(i)), 0,
  );

  // Adjusted scores: rejected errors become "passed"
  const adjPassed = data.total_passed + rejectedAtoms;
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
      errVal.textContent = adjErrorAtoms;
      errVal.style.color = adjErrorAtoms > 0 ? 'var(--danger)' : 'var(--success)';
    }
    const warnVal = metricCards[1].querySelector('.metric-value');
    if (warnVal) {
      warnVal.textContent = adjWarningAtoms;
      warnVal.style.color = adjWarningAtoms > 0 ? 'var(--warning)' : 'var(--success)';
    }
  }

  // Update per-category cards. Iterate the same 3-category sequence as
  // the initial render so N/A cards line up with their metricCards slot.
  let cardIdx = 2;
  const CATS = ['conformance', 'completeness', 'plausibility'];
  for (const cat of CATS) {
    const scores = (data.category_scores || {})[cat];
    if (!scores || scores.total == null) {
      cardIdx += 1;  // N/A card — skip, nothing to adjust
      continue;
    }
    const catRejectedAtoms = data.issues
      .filter(i => i.category === cat && i.severity === 'error')
      .reduce((s, i) => s + issueRejectedAtoms(i), 0);
    const adjCatPassed = scores.passed + catRejectedAtoms;
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
    panel.innerHTML = '<p>No summary data available. Click "Run Validation" to generate.</p>';
  }
}
