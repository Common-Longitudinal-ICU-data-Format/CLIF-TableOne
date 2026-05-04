const BASE = '/api';

async function request(method, path, body = null) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(BASE + path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export const api = {
  getConfig: () => request('GET', '/config'),
  putConfig: (path) => request('PUT', '/config', { config_path: path }),
  getTables: () => request('GET', '/tables'),
  analyze: (name, opts) => request('POST', `/analyze/${name}`, opts),
  analyzeAll: (opts) => request('POST', '/analyze-all', opts),
  getValidationSummary: () => request('GET', '/validation/summary'),
  getValidation: (name) => request('GET', `/validation/${name}`),
  getSummary: (name) => request('GET', `/summary/${name}`),
  getChart: (name, type) => request('GET', `/summary/${name}/charts/${type}`),
  getFeedback: (name) => request('GET', `/feedback/${name}`),
  putFeedback: (name, errorId, decision, reason, valueKey = null) =>
    request('PUT', `/feedback/${name}`, {
      error_id: errorId,
      decision,
      reason,
      ...(valueKey !== null && valueKey !== undefined ? { value_key: valueKey } : {}),
    }),
  saveFeedback: (name) => request('POST', `/feedback/${name}/save`),
  clearAllFeedback: () => request('DELETE', '/feedback'),
  downloadReport: (type) => `${BASE}/reports/download/${type}`,  // returns URL string
  tableReport: (name) => `${BASE}/reports/table/${name}`,  // returns URL string
  // Table One — legacy (cohort='ci' implied), kept for any caller that hasn't migrated.
  getTableone: (key) => request('GET', `/tableone/${key}`),
  getTableoneTab: (tab) => request('GET', `/tableone/data/${tab}`),
  getTableoneImage: (filename) => `${BASE}/tableone/images/${filename}`,  // returns URL string
  // Table One — cohort-aware (cohort ∈ 'ci' | 'ward')
  getTableoneAvailable: (cohort) => request('GET', `/tableone/${cohort}/available`),
  getTableoneStrobe: (cohort) => request('GET', `/tableone/${cohort}/strobe`),
  getTableoneTabFor: (cohort, tab) => request('GET', `/tableone/${cohort}/data/${tab}`),
  getTableoneImageFor: (cohort, filename) => `${BASE}/tableone/${cohort}/images/${filename}`,
  // Strata: lazy per-stratum fetch.  /strata returns the list; /strata/{stratum} returns its tables.
  getTableoneStrataList: (cohort) => request('GET', `/tableone/${cohort}/strata`),
  getTableoneStratum: (cohort, stratum) => request('GET', `/tableone/${cohort}/strata/${stratum}`),
  getMcide: (name) => request('GET', `/mcide/${name}`),
};
