export function renderTabs(tabsContainer, contentContainer, tabs, onTabChange) {
  // Render tab buttons
  tabsContainer.innerHTML = `<div class="tabs">
    ${tabs.map((t, i) => `<button class="tab${i === 0 ? ' active' : ''}" data-tab="${t.id}">${t.label}</button>`).join('')}
  </div>`;

  // Create panels
  contentContainer.innerHTML = tabs.map((t, i) =>
    `<div class="tab-content${i === 0 ? '' : ' hidden'}" data-panel="${t.id}"></div>`
  ).join('');

  const loaded = new Set();

  function activateTab(tabId) {
    tabsContainer.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
    contentContainer.querySelectorAll('.tab-content').forEach(p => p.classList.toggle('hidden', p.dataset.panel !== tabId));

    if (!loaded.has(tabId)) {
      loaded.add(tabId);
      const panel = contentContainer.querySelector(`[data-panel="${tabId}"]`);
      panel.innerHTML = '<div class="loading-spinner"></div>';
      onTabChange(tabId, panel);
    }
  }

  tabsContainer.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => activateTab(btn.dataset.tab));
  });

  // Activate first tab
  if (tabs.length > 0) activateTab(tabs[0].id);
}
