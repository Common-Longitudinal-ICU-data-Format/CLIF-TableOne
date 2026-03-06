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
  putFeedback: (name, errorId, decision, reason) =>
    request('PUT', `/feedback/${name}`, { error_id: errorId, decision, reason }),
  saveFeedback: (name) => request('POST', `/feedback/${name}/save`),
  regenerateReports: () => request('POST', '/reports/regenerate'),
  downloadReport: (type) => `${BASE}/reports/download/${type}`,  // returns URL string
  getTableone: (key) => request('GET', `/tableone/${key}`),
  getTableoneTab: (tab) => request('GET', `/tableone/data/${tab}`),
  getTableoneImage: (filename) => `${BASE}/tableone/images/${filename}`,  // returns URL string
  getMcide: (name) => request('GET', `/mcide/${name}`),
  getLlmStatus: () => request('GET', '/llm/status'),
  streamInterpretation(name, callbacks) {
    return _streamSSE(`${BASE}/llm/interpret/${name}`, callbacks);
  },
  streamInterpretationAll(callbacks) {
    return _streamSSE(`${BASE}/llm/interpret-all`, callbacks);
  },
};

function _streamSSE(url, { onChunk, onDone, onError }) {
  const ctrl = new AbortController();
  fetch(url, { method: 'POST', signal: ctrl.signal }).then(async (res) => {
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      if (onError) onError(err.detail || res.statusText);
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const msg = JSON.parse(line.slice(6));
          if (msg.error) { if (onError) onError(msg.error); return; }
          if (msg.text && onChunk) onChunk(msg.text);
          if (msg.done && onDone) onDone();
        } catch (_) { /* ignore parse errors */ }
      }
    }
  }).catch((e) => {
    if (e.name !== 'AbortError' && onError) onError(e.message);
  });
  return ctrl;
}
