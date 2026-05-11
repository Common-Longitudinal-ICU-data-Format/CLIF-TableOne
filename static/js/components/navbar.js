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

export function renderNavbar(el) {
  const config = state.get('config') || {};
  const theme = document.documentElement.getAttribute('data-theme') || 'dark';
  const currentPage = state.get('currentPage') || 'home';

  el.innerHTML = `
    <div class="navbar-left">
      <img class="navbar-logo" src="${theme === 'dark' ? '/images/clif_logo_v1_white.png' : '/images/clif_logo_red_2.png'}" alt="CLIF" onerror="this.style.display='none'">
      <span class="version-badge">v2.1</span>
    </div>
    <div class="navbar-center">
      <a class="navbar-link${currentPage === 'home' ? ' active' : ''}" data-page="home">Home</a>
      <a class="navbar-link${currentPage === 'validation' ? ' active' : ''}" data-page="validation">Validation</a>
      <a class="navbar-link${currentPage === 'tableone' ? ' active' : ''}" data-page="tableone">Table One</a>
    </div>
    <div class="navbar-right">
      <span class="navbar-site">${config.site_name || ''}</span>
      <button class="navbar-theme-toggle" id="theme-toggle" title="${theme === 'dark' ? 'Light Mode' : 'Dark Mode'}">
        ${theme === 'dark' ? SUN_ICON : MOON_ICON}
      </button>
    </div>
  `;

  // Nav link click handlers
  el.querySelectorAll('.navbar-link').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      navigate(link.dataset.page);
    });
  });

  // Update active link on page change
  state.on('currentPage', (page) => {
    el.querySelectorAll('.navbar-link').forEach(link => {
      link.classList.toggle('active', link.dataset.page === page);
    });
  });

  // Theme toggle
  document.getElementById('theme-toggle').addEventListener('click', toggleTheme);

  // Load table data into state
  loadTableData();
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
    btn.innerHTML = next === 'dark' ? SUN_ICON : MOON_ICON;
    btn.title = next === 'dark' ? 'Light Mode' : 'Dark Mode';
  }

  // Update logo
  const logoImg = document.querySelector('.navbar-logo');
  if (logoImg) {
    logoImg.src = next === 'dark'
      ? '/images/clif_logo_v1_white.png'
      : '/images/clif_logo_red_2.png';
  }
}

async function loadTableData() {
  try {
    const { tables } = await api.getTables();
    state.set('tables', tables);
  } catch (e) {
    console.warn('Could not load table status:', e);
  }
}
