import { api } from '../api.js';
import { renderTabs } from '../components/tabs.js';

export async function renderTableone(el) {
  // Check availability first
  try {
    const { available } = await api.getTableone('available');
    if (!available) {
      el.innerHTML = `<h1>Table One Results</h1><p>Results not yet generated. Run: <code>uv run run_tableone.py</code></p>`;
      return;
    }
  } catch (e) {
    el.innerHTML = `<h1>Table One Results</h1><p>Could not check availability.</p>`;
    return;
  }

  el.innerHTML = `
    <h1>Table One Results</h1>
    <p style="opacity:0.6;">To update, run: <code>uv run run_tableone.py</code></p>
    <div id="t1-tabs"></div>
    <div id="t1-content"></div>
  `;

  const tabs = [
    { id: 'cohort', label: 'Cohort' },
    { id: 'demographics', label: 'Demographics' },
    { id: 'medications', label: 'Medications' },
    { id: 'imv', label: 'IMV' },
    { id: 'sofa_cci', label: 'SOFA & CCI' },
    { id: 'outcomes', label: 'Outcomes' },
  ];

  renderTabs(
    document.getElementById('t1-tabs'),
    document.getElementById('t1-content'),
    tabs,
    async (tabId, panel) => {
      try {
        const data = await api.getTableoneTab(tabId);
        renderTabContent(tabId, data, panel);
      } catch (e) {
        panel.innerHTML = `<p>No data for this tab.</p>`;
      }
    }
  );
}

function imgTag(filename) {
  return `<img src="/api/tableone/images/${filename}" style="width:100%;border-radius:8px;margin:8px 0;" loading="lazy">`;
}

function renderTabContent(tabId, data, panel) {
  let html = '';

  if (tabId === 'cohort') {
    if (data.consort) html += `<h3>CONSORT Flow Diagram</h3>${imgTag(data.consort)}`;
    if (data.upset) html += `<h3>Cohort Intersections (UpSet Plot)</h3>${imgTag(data.upset)}`;
    if (data.venn) html += `<h3>Cohort Venn Diagram</h3>${imgTag(data.venn)}`;
    if (data.code_status) html += `<h3>Code Status Distribution</h3>${imgTag(data.code_status)}`;
    if (data.sankeys && data.sankeys.length > 0) {
      html += `<h3>Patient Flow Sankey Diagrams</h3><div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">`;
      for (const s of data.sankeys) {
        html += `<div><p style="font-weight:600;">${s.label}</p>${imgTag(s.filename)}</div>`;
      }
      html += `</div>`;
    }
  }

  else if (tabId === 'demographics') {
    if (data.tables) {
      for (const [key, tbl] of Object.entries(data.tables)) {
        html += `<h3>Table One - ${key === 'by_year' ? 'Stratified by Year' : 'Overall'}</h3>`;
        html += `<div id="demo-table-${key}"></div>`;
      }
    }
  }

  else if (tabId === 'medications') {
    if (data.html_plots) {
      for (const plot of data.html_plots) {
        html += `<h4>${plot.label}</h4><iframe src="/api/tableone/images/${plot.filename}" style="width:100%;height:520px;border:none;border-radius:8px;margin-bottom:16px;"></iframe>`;
      }
    }
    if (data.csv_files) {
      for (const f of data.csv_files) {
        html += `<h4>${f.label}</h4><div id="meds-table-${f.label.replace(/\s/g, '_')}"></div>`;
      }
    }
  }

  else if (tabId === 'imv') {
    if (data.images) {
      for (const img of data.images) html += `<h3>${img.label}</h3>${imgTag(img.filename)}`;
    }
    if (data.csv_files) {
      for (const f of data.csv_files) html += `<h4>${f.label}</h4><div id="imv-table-${f.label.replace(/\s/g, '_')}"></div>`;
    }
  }

  else if (tabId === 'sofa_cci') {
    if (data.images) {
      for (const img of data.images) html += `<h3>${img.label}</h3>${imgTag(img.filename)}`;
    }
    if (data.csv_files) {
      for (const f of data.csv_files) html += `<h4>${f.label}</h4><div id="sofa-table-${f.label.replace(/\s/g, '_')}"></div>`;
    }
  }

  else if (tabId === 'outcomes') {
    if (data.images) {
      for (const img of data.images) html += `<h3>${img.label}</h3>${imgTag(img.filename)}`;
    }
    if (!data.images || data.images.length === 0) html += '<p>No outcome data available.</p>';
  }

  if (!html) html = '<p>No data available for this tab.</p>';
  panel.innerHTML = html;

  // Initialize DataTables for any table data
  if (typeof jQuery !== 'undefined' && jQuery.fn.DataTable) {
    if (data.tables) {
      for (const [key, tbl] of Object.entries(data.tables)) {
        const container = panel.querySelector(`#demo-table-${key}`);
        if (!container || !tbl.data || tbl.data.length === 0) continue;
        const tableEl = document.createElement('table');
        tableEl.className = 'display';
        tableEl.style.width = '100%';
        container.appendChild(tableEl);
        jQuery(tableEl).DataTable({
          data: tbl.data,
          columns: tbl.columns.map(c => ({ data: c, title: c })),
          pageLength: 25,
          scrollX: true,
        });
      }
    }
    if (data.csv_files) {
      for (const f of data.csv_files) {
        if (!f.data || f.data.length === 0) continue;
        const container = panel.querySelector(`[id$="${f.label.replace(/\s/g, '_')}"]`);
        if (!container) continue;
        const tableEl = document.createElement('table');
        tableEl.className = 'display';
        tableEl.style.width = '100%';
        container.appendChild(tableEl);
        jQuery(tableEl).DataTable({
          data: f.data,
          columns: f.columns.map(c => ({ data: c, title: c })),
          pageLength: 25,
          scrollX: true,
        });
      }
    }
  }
}
