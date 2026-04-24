import { useState, useRef } from 'react';
import { uploadFile } from '../services/api';

export default function FileUpload({ onUploadComplete }) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => { e.preventDefault(); setDragging(true); };
  const handleDragLeave = () => setDragging(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) handleUpload(file);
  };

  const handleUpload = async (file) => {
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
      setStatus('File too large (max 50MB)');
      return;
    }

    setUploading(true);
    setProgress(0);
    setStatus(`Uploading ${file.name}...`);

    try {
      const result = await uploadFile(file, (pct) => setProgress(pct));
      setStatus(`✅ ${file.name} uploaded! Processing...`);
      setProgress(100);
      if (onUploadComplete) onUploadComplete(result);
      setTimeout(() => { setUploading(false); setProgress(0); setStatus(''); }, 3000);
    } catch (err) {
      setStatus(`❌ Error: ${err.message}`);
      setUploading(false);
      setProgress(0);
    }
  };

  return (
    <div>
      <div
        className={`upload-zone ${dragging ? 'dragging' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        id="upload-dropzone"
      >
        <div className="upload-icon">📄</div>
        <div className="upload-title">Drop files here or click to upload</div>
        <div className="upload-subtitle">
          Upload PDF documents, audio, or video files for AI-powered Q&A
        </div>
        <div className="upload-formats">
          <span className="format-badge pdf">PDF</span>
          <span className="format-badge audio">MP3</span>
          <span className="format-badge audio">WAV</span>
          <span className="format-badge video">MP4</span>
          <span className="format-badge video">MKV</span>
          <span className="format-badge video">MOV</span>
        </div>

        {uploading && (
          <div className="upload-progress">
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>
            <div className="progress-text">{status}</div>
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.mp3,.wav,.ogg,.flac,.m4a,.mp4,.avi,.mkv,.mov,.webm"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
          id="file-input"
        />
      </div>

      {!uploading && status && (
        <div className="progress-text" style={{ marginTop: 12, textAlign: 'center' }}>
          {status}
        </div>
      )}
    </div>
  );
}
