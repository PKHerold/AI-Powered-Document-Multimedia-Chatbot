import { formatFileSize } from '../services/api';

export default function DocumentList({ documents, selectedId, onSelect, onDelete }) {
  if (!documents || documents.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-icon">📂</div>
        <h3>No documents yet</h3>
        <p>Upload a PDF, audio, or video file to get started with AI-powered Q&A</p>
      </div>
    );
  }

  const typeIcons = { pdf: '📄', audio: '🎵', video: '🎬' };
  const typeColors = { pdf: 'pdf', audio: 'audio', video: 'video' };

  return (
    <div className="documents-grid" id="documents-grid">
      {documents.map((doc) => (
        <div
          key={doc.id}
          className={`document-card ${selectedId === doc.id ? 'active' : ''}`}
          onClick={() => onSelect(doc)}
          id={`doc-${doc.id}`}
        >
          <div className="doc-card-header">
            <div className={`doc-type-icon ${typeColors[doc.file_type]}`}>
              {typeIcons[doc.file_type] || '📄'}
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              <span className={`doc-status ${doc.status}`}>
                {doc.status === 'completed' ? '✓' : doc.status === 'processing' ? '⏳' : '✕'}{' '}
                {doc.status}
              </span>
              <button
                className="btn-icon"
                onClick={(e) => { e.stopPropagation(); onDelete(doc.id); }}
                title="Delete"
                id={`delete-${doc.id}`}
                style={{ width: 24, height: 24, fontSize: 12 }}
              >
                🗑
              </button>
            </div>
          </div>
          <div className="doc-card-title">{doc.original_filename}</div>
          <div className="doc-card-meta">
            <span>{formatFileSize(doc.file_size)}</span>
            <span>{doc.chunk_count} chunks</span>
            {doc.duration && <span>{Math.round(doc.duration)}s</span>}
          </div>
        </div>
      ))}
    </div>
  );
}
