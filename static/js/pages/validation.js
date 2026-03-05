import { api } from '../api.js';
import * as state from '../state.js';
import { renderStatusGrid } from '../components/status-grid.js';

export async function renderValidation(el) {
  el.innerHTML = `
    <h1>Table Validation Status</h1>
    <p>Click any table to view detailed validation results.</p>
    <div class="status-grid" id="status-grid"></div>
  `;

  try {
    const { tables } = await api.getTables();
    state.set('tables', tables);
    renderStatusGrid('status-grid', tables);
  } catch (e) {
    document.getElementById('status-grid').innerHTML = '<p>Could not load table status.</p>';
  }
}
