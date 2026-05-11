// Simple pub/sub state manager
const _state = {
  config: null,
  currentPage: 'home',
  selectedTable: null,
  tables: {},        // table statuses
  analysisResults: {},  // cached analysis per table
};

const _listeners = {};

export function get(key) {
  return _state[key];
}

export function set(key, value) {
  _state[key] = value;
  (_listeners[key] || []).forEach(fn => fn(value));
}

export function on(key, fn) {
  (_listeners[key] = _listeners[key] || []).push(fn);
}

export function off(key, fn) {
  _listeners[key] = (_listeners[key] || []).filter(f => f !== fn);
}
