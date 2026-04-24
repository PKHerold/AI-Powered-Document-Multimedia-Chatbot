import { useState, useEffect, useCallback } from 'react';
import './index.css';
import Sidebar from './components/Sidebar';
import FileUpload from './components/FileUpload';
import DocumentList from './components/DocumentList';
import ChatBot from './components/ChatBot';
import Summary from './components/Summary';
import MediaPlayer from './components/MediaPlayer';
import { getDocuments, deleteDocument } from './services/api';

function App() {
  const [activeView, setActiveView] = useState('upload');
  const [documents, setDocuments] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [toast, setToast] = useState(null);

  const showToast = (message, type = 'info') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 4000);
  };

  const fetchDocuments = useCallback(async () => {
    try {
      const docs = await getDocuments();
      setDocuments(docs);
    } catch (err) {
      console.error('Failed to fetch documents:', err);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
    const interval = setInterval(fetchDocuments, 5000);
    return () => clearInterval(interval);
  }, [fetchDocuments]);

  const handleUploadComplete = (result) => {
    showToast(`${result.filename} uploaded successfully!`, 'success');
    fetchDocuments();
  };

  const handleSelectDoc = (doc) => {
    setSelectedDoc(doc);
    setActiveView('chat');
  };

  const handleDeleteDoc = async (docId) => {
    try {
      await deleteDocument(docId);
      showToast('Document deleted', 'info');
      if (selectedDoc?.id === docId) setSelectedDoc(null);
      fetchDocuments();
    } catch (err) {
      showToast('Failed to delete: ' + err.message, 'error');
    }
  };

  const viewTitles = {
    upload: 'Upload Files',
    documents: 'Your Documents',
    chat: selectedDoc ? `Chat — ${selectedDoc.original_filename}` : 'Chat',
    summary: selectedDoc ? `Summary — ${selectedDoc.original_filename}` : 'Summary',
  };

  return (
    <div className="app-layout">
      <Sidebar
        activeView={activeView}
        onViewChange={setActiveView}
        documents={documents}
        selectedDocId={selectedDoc?.id}
        onSelectDoc={handleSelectDoc}
      />

      <main className="main-content">
        <header className="top-header">
          <h2>{viewTitles[activeView]}</h2>
          {selectedDoc && (activeView === 'chat' || activeView === 'summary') && (
            <div className="header-actions">
              <button className="btn btn-secondary" onClick={() => setActiveView('documents')}>
                📁 All Documents
              </button>
            </div>
          )}
        </header>

        {activeView === 'chat' ? (
          <div style={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
            {selectedDoc && (selectedDoc.file_type === 'audio' || selectedDoc.file_type === 'video') && (
              <div style={{ padding: '16px 32px 0' }}>
                <MediaPlayer document={selectedDoc} />
              </div>
            )}
            <ChatBot document={selectedDoc} />
          </div>
        ) : (
          <div className="content-area">
            {activeView === 'upload' && <FileUpload onUploadComplete={handleUploadComplete} />}
            {activeView === 'documents' && (
              <DocumentList
                documents={documents}
                selectedId={selectedDoc?.id}
                onSelect={handleSelectDoc}
                onDelete={handleDeleteDoc}
              />
            )}
            {activeView === 'summary' && <Summary document={selectedDoc} />}
          </div>
        )}
      </main>

      {toast && (
        <div className={`toast ${toast.type}`} id="toast">
          {toast.message}
        </div>
      )}
    </div>
  );
}

export default App;
