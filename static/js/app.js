import * as router from './router.js';
import * as state from './state.js';
import { api } from './api.js';
import { renderSidebar } from './components/sidebar.js';
import { renderHome } from './pages/home.js';
import { renderAnalysis } from './pages/analysis.js';
import { renderTableone } from './pages/tableone.js';
import { renderInstructions } from './pages/instructions.js';

// Register routes
router.register('home', renderHome);
router.register('analysis', renderAnalysis);
router.register('tableone', renderTableone);
router.register('instructions', renderInstructions);

// Initialize theme from localStorage
function initTheme() {
  const saved = localStorage.getItem('clif-theme') || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
}

// Initialize
async function init() {
  initTheme();

  try {
    const config = await api.getConfig();
    state.set('config', config);
  } catch (e) {
    console.warn('Config not loaded:', e);
  }

  renderSidebar(document.getElementById('sidebar'));
  router.init();
}

document.addEventListener('DOMContentLoaded', init);
