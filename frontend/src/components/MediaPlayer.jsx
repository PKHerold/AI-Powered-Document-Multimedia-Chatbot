import { useRef, useEffect } from 'react';
import { getMediaUrl, formatTimestamp } from '../services/api';

export default function MediaPlayer({ document, timestamps = [], onTimeUpdate }) {
  const mediaRef = useRef(null);

  const isVideo = document?.file_type === 'video';
  const isMedia = document?.file_type === 'audio' || isVideo;

  const jumpToTimestamp = (seconds) => {
    if (mediaRef.current) {
      mediaRef.current.currentTime = seconds;
      mediaRef.current.play();
    }
  };

  // Expose jumpToTimestamp via window for chat component
  useEffect(() => {
    window.__mediaPlayer = { jumpToTimestamp };
    return () => { delete window.__mediaPlayer; };
  }, []);

  if (!document || !isMedia) return null;

  const mediaUrl = getMediaUrl(document.id);

  return (
    <div className="media-player" id="media-player">
      {isVideo ? (
        <video
          ref={mediaRef}
          src={mediaUrl}
          controls
          preload="metadata"
          onTimeUpdate={() => onTimeUpdate?.(mediaRef.current?.currentTime)}
        />
      ) : (
        <audio
          ref={mediaRef}
          src={mediaUrl}
          controls
          preload="metadata"
          style={{ width: '100%' }}
          onTimeUpdate={() => onTimeUpdate?.(mediaRef.current?.currentTime)}
        />
      )}

      {timestamps.length > 0 && (
        <div className="timestamp-markers">
          {timestamps.map((ts, i) => (
            <button
              key={i}
              className="timestamp-chip"
              onClick={() => jumpToTimestamp(ts.start)}
              title={ts.text || ''}
              id={`timestamp-${i}`}
            >
              ▶ {formatTimestamp(ts.start)}
              {ts.end ? ` - ${formatTimestamp(ts.end)}` : ''}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
