import { useState, useRef, useEffect } from 'react';
import { sendChatMessage, formatTimestamp } from '../services/api';

export default function ChatBot({ document }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Clear chat when document changes
  useEffect(() => { setMessages([]); }, [document?.id]);
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || !document || loading) return;
    const question = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: question }]);
    setLoading(true);

    try {
      const result = await sendChatMessage(document.id, question);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: result.answer, timestamps: result.timestamps || [] },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `❌ Error: ${err.message}` },
      ]);
    }
    setLoading(false);
    inputRef.current?.focus();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const handleTimestampClick = (seconds) => {
    if (window.__mediaPlayer) window.__mediaPlayer.jumpToTimestamp(seconds);
  };

  if (!document) {
    return (
      <div className="chat-container">
        <div className="empty-state" style={{ flex: 1 }}>
          <div className="empty-icon">💬</div>
          <h3>Start a conversation</h3>
          <p>Select a document from the sidebar to ask questions about it</p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-container" id="chat-container">
      <div className="chat-messages" id="chat-messages">
        {messages.length === 0 && (
          <div className="empty-state" style={{ flex: 1 }}>
            <div className="empty-icon">🤖</div>
            <h3>Ask anything about this document</h3>
            <p>{document.original_filename}</p>
            <div style={{ display: 'flex', gap: 8, marginTop: 16, flexWrap: 'wrap', justifyContent: 'center' }}>
              {['What is this about?', 'Summarize the key points', 'What are the main topics?'].map((q) => (
                <button
                  key={q}
                  className="btn btn-secondary"
                  onClick={() => { setInput(q); }}
                  style={{ fontSize: 12 }}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`chat-message ${msg.role}`}>
            <div className={`message-avatar ${msg.role === 'user' ? 'user-avatar' : 'ai-avatar'}`}>
              {msg.role === 'user' ? '👤' : '🤖'}
            </div>
            <div>
              <div className="message-bubble">{msg.content}</div>
              {msg.timestamps?.length > 0 && (
                <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {msg.timestamps.map((ts, j) => (
                    <button
                      key={j}
                      className="timestamp-chip"
                      onClick={() => handleTimestampClick(ts.start)}
                      id={`chat-ts-${i}-${j}`}
                    >
                      ▶ {formatTimestamp(ts.start)}
                      {ts.end ? ` - ${formatTimestamp(ts.end)}` : ''}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="chat-message assistant">
            <div className="message-avatar ai-avatar">🤖</div>
            <div className="message-bubble">
              <div className="typing-indicator">
                <span /><span /><span />
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-area">
        <div className="chat-input-wrapper">
          <textarea
            ref={inputRef}
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={`Ask about ${document.original_filename}...`}
            rows={1}
            disabled={loading || document.status !== 'completed'}
            id="chat-input"
          />
          <button
            className="send-btn"
            onClick={handleSend}
            disabled={!input.trim() || loading}
            id="send-btn"
          >
            ➤
          </button>
        </div>
      </div>
    </div>
  );
}
