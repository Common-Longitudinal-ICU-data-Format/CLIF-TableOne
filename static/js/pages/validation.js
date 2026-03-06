import { api } from '../api.js';
import * as state from '../state.js';
import { navigate } from '../router.js';
import { renderStatusGrid } from '../components/status-grid.js';

export async function renderValidation(el) {
  el.innerHTML = `
    <button class="btn btn-back" id="btn-back-home">&larr; Back to Home</button>
    <h1>Table Validation Status</h1>
    <p>Click any table to view detailed validation results.</p>
    <div id="dqa-report-card"></div>
    <div class="status-grid" id="status-grid"></div>
  `;

  document.getElementById('btn-back-home').addEventListener('click', () => navigate('home'));

  try {
    const [{ tables }, summary] = await Promise.all([
      api.getTables(),
      api.getValidationSummary(),
    ]);
    state.set('tables', tables);
    renderReportCard(summary);
    renderStatusGrid('status-grid', tables);
  } catch (e) {
    document.getElementById('status-grid').innerHTML = '<p>Could not load table status.</p>';
  }
}

function renderReportCard(data) {
  const container = document.getElementById('dqa-report-card');
  if (!data || data.tables_analyzed === 0) return;

  const overallColor = data.overall_pct >= 85 ? 'var(--success)' : data.overall_pct >= 60 ? 'var(--warning)' : 'var(--danger)';

  let html = `<div class="card" style="margin-bottom:20px;padding:20px;">`;
  html += `<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px;">
    <span style="font-size:2.2rem;font-weight:700;color:${overallColor}">${data.overall_pct}%</span>
    <span style="font-size:0.85rem;opacity:0.5;">${data.total_passed}/${data.total_checks} checks passed across ${data.tables_analyzed} of ${data.tables_total} tables</span>
  </div>`;
  html += `<div class="progress-bar" style="margin-bottom:16px;"><div class="progress-fill" style="width:${data.overall_pct}%"></div></div>`;

  const catInfo = {
    conformance: {
      short: 'Structural correctness',
      detail: 'Checks that data matches the CLIF specification — expected fields present, correct data types, valid mCIDE values, UTC datetimes, correct lab units, and derived fields match calculations.',
    },
    completeness: {
      short: 'Data coverage',
      detail: 'Checks that required data elements are present — mandatory fields NOT NULL, conditional fields appear when expected, related records exist across tables, and all mCIDE categories have at least one record.',
    },
    plausibility: {
      short: 'Clinical validity',
      detail: 'Checks that values make clinical sense — no duplicate keys, physiologically plausible ranges, no overlapping intervals, stable distributions, expected cross-table references, and appropriate frequency.',
    },
  };

  html += `<div style="display:flex;gap:12px;flex-wrap:wrap;">`;
  for (const [cat, scores] of Object.entries(data.categories)) {
    const pct = scores.total ? Math.round(scores.passed / scores.total * 100) : 100;
    const color = pct >= 85 ? 'var(--success)' : pct >= 60 ? 'var(--warning)' : 'var(--danger)';
    const info = catInfo[cat] || { short: '', detail: '' };
    const infoBtn = info.detail ? `<span class="dqa-info-btn" data-cat="${cat}" style="display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px;font-size:10px;border-radius:50%;border:1px solid rgba(128,128,128,0.4);color:rgba(128,128,128,0.6);cursor:pointer;margin-left:4px;vertical-align:middle;line-height:1;">i</span>` : '';
    html += `<div class="card metric-card"><div class="metric-label">${cat.charAt(0).toUpperCase() + cat.slice(1)}${infoBtn}</div><div class="metric-value" style="color:${color}">${scores.passed}/${scores.total}</div><div style="font-size:0.7rem;opacity:0.5;margin-top:2px;">${info.short}</div></div>`;
  }
  html += `<div class="card metric-card"><div class="metric-label">Errors</div><div class="metric-value" style="color:${data.total_errors > 0 ? 'var(--danger)' : 'var(--success)'}">${data.total_errors}</div></div>`;
  html += `<div class="card metric-card"><div class="metric-label">Warnings</div><div class="metric-value" style="color:${data.total_warnings > 0 ? 'var(--warning)' : 'var(--success)'}">${data.total_warnings}</div></div>`;
  html += `</div></div>`;

  container.innerHTML = html;

  // Info button popups
  container.querySelectorAll('.dqa-info-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const cat = btn.dataset.cat;
      const info = catInfo[cat];
      if (!info) return;

      // Remove any existing popup
      document.querySelectorAll('.dqa-info-popup').forEach(p => p.remove());

      const popup = document.createElement('div');
      popup.className = 'dqa-info-popup';
      popup.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;display:flex;align-items:center;justify-content:center;z-index:1000;background:rgba(0,0,0,0.3);';
      popup.innerHTML = `
        <div style="background:var(--bg-card,#fff);border-radius:10px;padding:24px;max-width:420px;width:90%;box-shadow:0 8px 32px rgba(0,0,0,0.2);">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <strong style="font-size:1rem;">${cat.charAt(0).toUpperCase() + cat.slice(1)}</strong>
            <span class="dqa-popup-close" style="cursor:pointer;font-size:1.2rem;opacity:0.5;line-height:1;">&times;</span>
          </div>
          <p style="font-size:0.85rem;line-height:1.5;margin:0;opacity:0.8;">${info.detail}</p>
        </div>
      `;
      document.body.appendChild(popup);
      popup.querySelector('.dqa-popup-close').addEventListener('click', () => popup.remove());
      popup.addEventListener('click', (ev) => { if (ev.target === popup) popup.remove(); });
    });
  });
}
