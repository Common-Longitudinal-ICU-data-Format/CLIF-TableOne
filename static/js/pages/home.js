import { api } from '../api.js';
import * as state from '../state.js';
import { navigate } from '../router.js';
import { connectSSE } from '../sse.js';
import { TABLE_DISPLAY_NAMES, updateSidebarStatus } from '../components/sidebar.js';

const STATUS_COLORS = {
  complete: 'var(--success)',
  partial: 'var(--warning)',
  incomplete: 'var(--danger)',
  not_analyzed: '#555',
};

const STATUS_ICONS = {
  complete: '\u2705', partial: '\u26a0\ufe0f', incomplete: '\u274c', not_analyzed: '\u2b55',
};

export async function renderHome(el) {
  el.innerHTML = `
    <h1>CLIF Validation Status Overview</h1>
    <h3>Table Validation Status</h3>
    <div class="status-grid" id="status-grid"></div>
    <hr>
    <div id="home-actions">
      <h3>Analyze All Tables</h3>
      <div style="display:flex;gap:12px;align-items:center;margin-bottom:12px;">
        <label><input type="checkbox" id="chk-sample" checked> Use 1k ICU Sample (Recommended)</label>
        <label><input type="checkbox" id="chk-aggregates"> Generate Summary Aggregates</label>
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
  `;

  // Load table statuses
  try {
    const { tables } = await api.getTables();
    state.set('tables', tables);
    renderGrid(tables);
    // Show download buttons if reports exist
    checkDownloads();
  } catch (e) {
    document.getElementById('status-grid').innerHTML = '<p>Could not load table status.</p>';
  }

  // Analyze all button
  document.getElementById('btn-analyze-all').addEventListener('click', async () => {
    const useSample = document.getElementById('chk-sample').checked;
    const genAgg = document.getElementById('chk-aggregates').checked;
    try {
      const { task_id } = await api.analyzeAll({ use_sample: useSample, generate_aggregates: genAgg });
      showProgress(task_id);
    } catch (e) {
      alert('Failed to start analysis: ' + e.message);
    }
  });

  // Regenerate button
  document.getElementById('btn-regenerate').addEventListener('click', async () => {
    document.getElementById('btn-regenerate').disabled = true;
    document.getElementById('btn-regenerate').textContent = 'Regenerating...';
    try {
      await api.regenerateReports();
      document.getElementById('btn-regenerate').textContent = 'Done!';
      setTimeout(() => {
        document.getElementById('btn-regenerate').textContent = 'Regenerate All Reports';
        document.getElementById('btn-regenerate').disabled = false;
      }, 2000);
      checkDownloads();
    } catch (e) {
      alert('Failed: ' + e.message);
      document.getElementById('btn-regenerate').textContent = 'Regenerate All Reports';
      document.getElementById('btn-regenerate').disabled = false;
    }
  });
}

function renderGrid(tables) {
  const grid = document.getElementById('status-grid');
  grid.innerHTML = Object.entries(tables).map(([name, info]) => {
    const color = STATUS_COLORS[info.status] || STATUS_COLORS.not_analyzed;
    const ts = info.timestamp ? new Date(info.timestamp).toLocaleString() : '';
    const statusLabel = info.status === 'not_analyzed' ? 'Not analyzed' : info.status.toUpperCase();
    const badgeClass = info.status === 'complete' ? 'badge-success' :
                       info.status === 'partial' ? 'badge-warning' :
                       info.status === 'incomplete' ? 'badge-danger' : 'badge-info';
    return `
      <div class="status-card" data-table="${name}" style="border-left-color:${color};">
        <div class="card-title">${info.display_name}</div>
        <div class="card-status"><span class="badge ${badgeClass}">${statusLabel}</span></div>
        ${ts ? `<div class="card-ts">${ts}</div>` : ''}
      </div>
    `;
  }).join('');

  grid.querySelectorAll('.status-card').forEach(card => {
    card.addEventListener('click', () => {
      const table = card.dataset.table;
      state.set('selectedTable', table);
      navigate('analysis', { table });
    });
  });
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
      // Refresh grid and sidebar
      api.getTables().then(({ tables }) => {
        state.set('tables', tables);
        renderGrid(tables);
        updateSidebarStatus(tables);
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
  // Simple check: try fetching report HEAD
  fetch('/api/reports/download/pdf', { method: 'HEAD' }).then(r => {
    if (r.ok) {
      section.style.display = 'block';
      document.getElementById('dl-pdf').href = '/api/reports/download/pdf';
      document.getElementById('dl-csv').href = '/api/reports/download/csv';
    }
  }).catch(() => {});
}
