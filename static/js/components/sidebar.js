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
const HOME_ICON = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>`;

export function renderSidebar(el) {
  const config = state.get('config') || {};
  const theme = document.documentElement.getAttribute('data-theme') || 'dark';

  el.innerHTML = `
    <div class="sidebar-inner">
      <div class="sidebar-logo">
        <img src="${theme === 'dark' ? '/images/clif_logo_v1_white.png' : '/images/clif_logo_red_2.png'}" alt="CLIF" onerror="this.style.display='none'">
        <span class="version-badge">v2.1</span>
      </div>

      <div class="sidebar-config">
        <div class="site-name">${config.site_name || 'Not loaded'}</div>
        <div class="site-path">${config.tables_path || ''}</div>
      </div>

      <div style="margin-bottom:16px;">
        <button class="theme-toggle" id="theme-toggle">
          ${theme === 'dark' ? SUN_ICON : MOON_ICON}
          ${theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
        </button>
      </div>

      <div class="sidebar-section" style="margin-bottom:12px;">
        <button class="nav-btn" data-page="home">${HOME_ICON} Home</button>
      </div>

      <div class="sidebar-section" id="sidebar-status">
        <div class="sidebar-label">Tables</div>
        <div id="status-list">
          <div style="padding:8px;color:var(--text-muted);font-size:0.8rem;">
            <span class="loading-spinner" style="width:14px;height:14px;margin-right:6px;vertical-align:middle;"></span>
            Loading...
          </div>
        </div>
      </div>

      <div class="sidebar-footer"></div>
    </div>
  `;

  // Nav button click handlers
  el.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => navigate(btn.dataset.page));
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

  const STATUS_ICONS = {
    pass: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--success)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`,
    fail: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--danger)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`,
    warning: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--warning)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>`,
    not_analyzed: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="8" y1="12" x2="16" y2="12"/></svg>`,
  };

  list.innerHTML = Object.entries(tables).map(([name, info]) => {
    const status = info.status || 'not_analyzed';
    const display = TABLE_DISPLAY_NAMES[name] || name;
    const full = info.display_name || name;
    const icon = STATUS_ICONS[status] || STATUS_ICONS.not_analyzed;
    return `
      <div class="sidebar-status-item ${status}" data-table="${name}" title="${full}">
        <span class="status-icon">${icon}</span>
        <span class="status-label">${display}</span>
      </div>
    `;
  }).join('');

  // Click to navigate to table analysis
  list.querySelectorAll('.sidebar-status-item').forEach(item => {
    item.style.cursor = 'pointer';
    item.addEventListener('click', () => {
      const table = item.dataset.table;
      state.set('selectedTable', table);
      navigate('analysis', { table });
    });
  });
}
