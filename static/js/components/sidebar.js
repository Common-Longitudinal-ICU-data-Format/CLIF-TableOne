import * as state from '../state.js';
import { navigate } from '../router.js';
import { api } from '../api.js';

const TABLE_DISPLAY_NAMES = {
  patient: 'Patient',
  hospitalization: 'Hospitalization',
  adt: 'ADT',
  code_status: 'Code Status',
  crrt_therapy: 'CRRT Therapy',
  ecmo_mcs: 'ECMO/MCS',
  hospital_diagnosis: 'Hospital Diagnosis',
  labs: 'Labs',
  medication_admin_continuous: 'Medication Admin (Continuous)',
  medication_admin_intermittent: 'Medication Admin (Intermittent)',
  microbiology_culture: 'Microbiology Culture',
  microbiology_nonculture: 'Microbiology Non-Culture',
  microbiology_susceptibility: 'Microbiology Susceptibility',
  patient_assessments: 'Patient Assessments',
  patient_procedures: 'Patient Procedures',
  position: 'Position',
  respiratory_support: 'Respiratory Support',
  vitals: 'Vitals',
};

export { TABLE_DISPLAY_NAMES };

const SUN_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`;
const MOON_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;

export function renderSidebar(el) {
  const config = state.get('config') || {};
  const theme = document.documentElement.getAttribute('data-theme') || 'dark';

  el.innerHTML = `
    <div class="sidebar-inner">
      <div class="sidebar-logo">
        <img src="/images/clif_logo_v1_white.png" alt="CLIF" onerror="this.style.display='none'">
        <span>CLIF 2.1</span>
      </div>

      <div class="sidebar-config">
        <div class="site-name">${config.site_name || 'Not loaded'}</div>
        <div class="site-path">${config.tables_path || ''}</div>
      </div>

      <div class="sidebar-section">
        <div class="sidebar-label">Navigation</div>
        <button class="nav-btn" data-page="home">Home</button>
        <button class="nav-btn" data-page="instructions">Instructions</button>
        <button class="nav-btn" data-page="tableone">Table One Results</button>
      </div>

      <div class="sidebar-section">
        <div class="sidebar-label">Select Table</div>
        <select id="table-select">
          ${Object.entries(TABLE_DISPLAY_NAMES).map(([k, v]) =>
            `<option value="${k}">${v}</option>`
          ).join('')}
        </select>
      </div>

      <div class="sidebar-section" id="sidebar-status">
        <div class="sidebar-label">Table Status</div>
        <div id="status-list">
          <div style="padding:8px;color:var(--text-muted);font-size:0.8rem;">
            <span class="loading-spinner" style="width:14px;height:14px;margin-right:6px;vertical-align:middle;"></span>
            Loading...
          </div>
        </div>
      </div>

      <div class="sidebar-footer">
        <button class="theme-toggle" id="theme-toggle">
          ${theme === 'dark' ? SUN_ICON : MOON_ICON}
          ${theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
        </button>
      </div>
    </div>
  `;

  // Nav button click handlers
  el.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => navigate(btn.dataset.page));
  });

  // Table select handler
  const select = el.querySelector('#table-select');
  select.addEventListener('change', () => {
    state.set('selectedTable', select.value);
    navigate('analysis', { table: select.value });
  });

  // Highlight active nav
  state.on('currentPage', (page) => {
    el.querySelectorAll('.nav-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.page === page);
    });
  });

  // Theme toggle
  document.getElementById('theme-toggle').addEventListener('click', toggleTheme);

  // Fetch and render table statuses
  loadSidebarStatus();
}

function toggleTheme() {
  const html = document.documentElement;
  const current = html.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  localStorage.setItem('clif-theme', next);

  // Update toggle button
  const btn = document.getElementById('theme-toggle');
  if (btn) {
    btn.innerHTML = `${next === 'dark' ? SUN_ICON : MOON_ICON} ${next === 'dark' ? 'Light Mode' : 'Dark Mode'}`;
  }

  // Update logo for light/dark
  const logoImg = document.querySelector('.sidebar-logo img');
  if (logoImg) {
    logoImg.src = next === 'dark'
      ? '/images/clif_logo_v1_white.png'
      : '/images/clif_logo_red_2.png';
  }
}

async function loadSidebarStatus() {
  try {
    const { tables } = await api.getTables();
    state.set('tables', tables);
    updateSidebarStatus(tables);
  } catch (e) {
    const list = document.getElementById('status-list');
    if (list) {
      list.innerHTML = `<div style="padding:8px;color:var(--text-muted);font-size:0.78rem;">Could not load status</div>`;
    }
  }
}

export function updateSidebarStatus(tables) {
  const list = document.getElementById('status-list');
  if (!list) return;

  list.innerHTML = Object.entries(tables).map(([name, info]) => {
    const status = info.status || 'not_analyzed';
    const display = TABLE_DISPLAY_NAMES[name] || name;
    return `
      <div class="sidebar-status-item" data-table="${name}">
        <span class="status-dot ${status}"></span>
        <span>${display}</span>
      </div>
    `;
  }).join('');

  // Click to navigate to table analysis
  list.querySelectorAll('.sidebar-status-item').forEach(item => {
    item.style.cursor = 'pointer';
    item.addEventListener('click', () => {
      const table = item.dataset.table;
      state.set('selectedTable', table);
      const select = document.getElementById('table-select');
      if (select) select.value = table;
      navigate('analysis', { table });
    });
  });
}
