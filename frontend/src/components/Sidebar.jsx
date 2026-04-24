export default function Sidebar({ activeView, onViewChange, documents, selectedDocId, onSelectDoc }) {
  const typeIcons = { pdf: '📄', audio: '🎵', video: '🎬' };

  return (
    <aside className="sidebar" id="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <div className="logo-icon">🧠</div>
          <h1>DocQ&A</h1>
        </div>
      </div>

      <nav className="sidebar-nav">
        <button
          className={`nav-item ${activeView === 'upload' ? 'active' : ''}`}
          onClick={() => onViewChange('upload')}
          id="nav-upload"
        >
          <span className="nav-icon">📤</span>
          <span>Upload</span>
        </button>
        <button
          className={`nav-item ${activeView === 'documents' ? 'active' : ''}`}
          onClick={() => onViewChange('documents')}
          id="nav-documents"
        >
          <span className="nav-icon">📁</span>
          <span>Documents</span>
        </button>
        <button
          className={`nav-item ${activeView === 'chat' ? 'active' : ''}`}
          onClick={() => onViewChange('chat')}
          id="nav-chat"
        >
          <span className="nav-icon">💬</span>
          <span>Chat</span>
        </button>
        <button
          className={`nav-item ${activeView === 'summary' ? 'active' : ''}`}
          onClick={() => onViewChange('summary')}
          id="nav-summary"
        >
          <span className="nav-icon">📋</span>
          <span>Summary</span>
        </button>
      </nav>

      <div className="sidebar-documents">
        <div className="sidebar-section-title">Your Documents</div>
        {documents.length === 0 ? (
          <div style={{ padding: '12px', color: 'var(--text-muted)', fontSize: 12 }}>
            No documents uploaded
          </div>
        ) : (
          documents.map((doc) => (
            <div
              key={doc.id}
              className={`doc-item ${selectedDocId === doc.id ? 'active' : ''}`}
              onClick={() => onSelectDoc(doc)}
              title={doc.original_filename}
            >
              <span className="doc-icon">{typeIcons[doc.file_type] || '📄'}</span>
              <span className="doc-name">{doc.original_filename}</span>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
