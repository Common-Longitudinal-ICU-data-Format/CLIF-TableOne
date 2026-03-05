import * as state from './state.js';

const routes = {};

export function register(hash, renderFn) {
  routes[hash] = renderFn;
}

export function navigate(hash, params = {}) {
  const qs = new URLSearchParams(params).toString();
  window.location.hash = qs ? `${hash}?${qs}` : hash;
}

export function getParams() {
  const [, qs] = window.location.hash.split('?');
  return Object.fromEntries(new URLSearchParams(qs || ''));
}

function handleRoute() {
  const [hash] = window.location.hash.replace('#', '').split('?');
  const page = hash || 'home';
  state.set('currentPage', page);
  const content = document.getElementById('content');
  if (!content) return;
  const renderFn = routes[page];
  if (renderFn) {
    content.innerHTML = '';
    renderFn(content, getParams());
  } else {
    content.innerHTML = '<h2>Page not found</h2>';
  }
}

export function init() {
  window.addEventListener('hashchange', handleRoute);
  handleRoute();
}
