import * as state from '../state.js';
import { navigate } from '../router.js';

export const STATUS_COLORS = {
  complete: 'var(--success)',
  partial: 'var(--warning)',
  incomplete: 'var(--danger)',
  not_analyzed: '#555',
};

export const STATUS_ICONS = {
  complete: '\u2705', partial: '\u26a0\ufe0f', incomplete: '\u274c', not_analyzed: '\u2b55',
};

export function renderStatusGrid(containerId, tables) {
  const grid = document.getElementById(containerId);
  if (!grid) return;

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
