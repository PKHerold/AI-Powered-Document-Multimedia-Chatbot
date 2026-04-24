import { useState } from 'react';
import { getSummary } from '../services/api';

export default function Summary({ document }) {
  const [summary, setSummary] = useState(document?.summary || '');
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {
    if (!document) return;
    setLoading(true);
    try {
      const result = await getSummary(document.id);
      setSummary(result.summary);
    } catch (err) {
      setSummary('Error generating summary: ' + err.message);
    }
    setLoading(false);
  };

  // Update summary when document changes
  if (document?.summary && !summary) {
    setSummary(document.summary);
  }

  if (!document) {
    return (
      <div className="empty-state">
        <div className="empty-icon">📋</div>
        <h3>No document selected</h3>
        <p>Select a document to view its summary</p>
      </div>
    );
  }

  return (
    <div className="summary-card" id="summary-card">
      <h3>📋 Summary — {document.original_filename}</h3>

      {loading ? (
        <div>
          <div className="skeleton skeleton-text" style={{ width: '100%' }} />
          <div className="skeleton skeleton-text" style={{ width: '90%' }} />
          <div className="skeleton skeleton-text" style={{ width: '95%' }} />
          <div className="skeleton skeleton-text" style={{ width: '60%' }} />
        </div>
      ) : summary ? (
        <div className="summary-text">{summary}</div>
      ) : (
        <div style={{ textAlign: 'center', padding: 20 }}>
          <p style={{ color: 'var(--text-secondary)', marginBottom: 12 }}>
            No summary generated yet
          </p>
          <button className="btn btn-primary" onClick={handleGenerate} id="generate-summary-btn">
            ✨ Generate Summary
          </button>
        </div>
      )}

      {summary && (
        <button
          className="btn btn-secondary"
          onClick={handleGenerate}
          style={{ marginTop: 16 }}
          id="regenerate-summary-btn"
        >
          🔄 Regenerate
        </button>
      )}
    </div>
  );
}
