/* API client for the backend */
const API_BASE = 'http://localhost:8000/api';

export async function uploadFile(file, onProgress) {
  const formData = new FormData();
  formData.append('file', file);

  const xhr = new XMLHttpRequest();

  return new Promise((resolve, reject) => {
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        reject(new Error(JSON.parse(xhr.responseText)?.detail || 'Upload failed'));
      }
    });

    xhr.addEventListener('error', () => reject(new Error('Network error')));
    xhr.open('POST', `${API_BASE}/upload`);
    xhr.send(formData);
  });
}

export async function getDocuments() {
  const res = await fetch(`${API_BASE}/documents`);
  if (!res.ok) throw new Error('Failed to fetch documents');
  return res.json();
}

export async function getDocument(id) {
  const res = await fetch(`${API_BASE}/documents/${id}`);
  if (!res.ok) throw new Error('Document not found');
  return res.json();
}

export async function deleteDocument(id) {
  const res = await fetch(`${API_BASE}/documents/${id}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to delete document');
  return res.json();
}

export async function getSummary(documentId) {
  const res = await fetch(`${API_BASE}/documents/${documentId}/summary`);
  if (!res.ok) throw new Error('Failed to get summary');
  return res.json();
}

export async function sendChatMessage(documentId, question) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document_id: documentId, question }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Chat request failed');
  }
  return res.json();
}

export function streamChat(documentId, question, onChunk, onDone, onError) {
  const url = `${API_BASE}/chat/stream?document_id=${encodeURIComponent(documentId)}&question=${encodeURIComponent(question)}`;
  const eventSource = new EventSource(url);

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'chunk') {
        onChunk(data.content);
      } else if (data.type === 'done') {
        onDone(data.timestamps || []);
        eventSource.close();
      }
    } catch (e) {
      console.error('SSE parse error:', e);
    }
  };

  eventSource.onerror = (err) => {
    onError(err);
    eventSource.close();
  };

  return eventSource;
}

export async function getChatHistory(documentId) {
  const res = await fetch(`${API_BASE}/chat/history/${documentId}`);
  if (!res.ok) throw new Error('Failed to get chat history');
  return res.json();
}

export async function clearChatHistory(documentId) {
  const res = await fetch(`${API_BASE}/chat/history/${documentId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to clear chat history');
  return res.json();
}

export function getMediaUrl(documentId) {
  return `${API_BASE}/media/${documentId}`;
}

export function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

export function formatDuration(seconds) {
  if (!seconds) return '';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export function formatTimestamp(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}
