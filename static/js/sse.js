export function connectSSE(taskId, { onProgress, onComplete, onError }) {
  const es = new EventSource(`/api/sse/progress/${taskId}`);

  es.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.status === 'progress' && onProgress) onProgress(data);
    if (data.status === 'complete' && onComplete) { onComplete(data); es.close(); }
    if (data.status === 'error' && onError) { onError(data); es.close(); }
  };

  es.onerror = () => {
    if (onError) onError({ message: 'Connection lost' });
    es.close();
  };

  return es;
}
