import { api } from '../api.js';
import * as state from '../state.js';
import { navigate } from '../router.js';
import { renderStatusGrid } from '../components/status-grid.js';

export async function renderValidation(el) {
  el.innerHTML = `
    <button class="btn btn-back" id="btn-back-home">&larr; Back to Home</button>
    <h1>Table Validation Status</h1>
    <p>Click any table to view detailed validation results.</p>
    <div class="status-grid" id="status-grid"></div>

    <hr>

    <div class="faq-section">
      <h3>DQA Check Reference</h3>

      <details>
        <summary>Completeness</summary>
        <p>Required data elements are present — mandatory fields are NOT NULL, conditionally required fields appear when expected, records exist in related tables where logically expected, and all standardized mCIDE values have at least one observed record.</p>
      </details>

      <details>
        <summary>Conformance</summary>
        <p>Data values match the CLIF spec — all expected fields are present, data types are correct, mCIDE permissible values are valid, datetimes use UTC format (+00:00), lab reference units are correct, and derived fields match expected calculations.</p>
      </details>

      <details>
        <summary>Plausibility</summary>
        <p>Values make clinical and operational sense — no duplicate records on composite keys, values are physiologically plausible, time intervals don't overlap inappropriately, categorical distributions are stable over time, expected data exists across related tables, and measurement frequency is appropriate.</p>
      </details>
    </div>
  `;

  document.getElementById('btn-back-home').addEventListener('click', () => navigate('home'));

  try {
    const { tables } = await api.getTables();
    state.set('tables', tables);
    renderStatusGrid('status-grid', tables);
  } catch (e) {
    document.getElementById('status-grid').innerHTML = '<p>Could not load table status.</p>';
  }
}
