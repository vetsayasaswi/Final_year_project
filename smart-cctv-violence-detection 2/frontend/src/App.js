import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import './Auth.css';
import LoginPage from './LoginPage';
import AdminDashboard from './AdminDashboard';

const API_URL = 'http://localhost:8000';

function App() {
  // ── Auth state (no persistence — fresh login every session) ─────────────────
  const [auth, setAuth] = useState(null);
  const [page, setPage] = useState('app'); // 'app' | 'admin'

  // ── Detection state ──────────────────────────────────────────────────────────
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [status, setStatus]         = useState('Monitoring...');
  const [confidence, setConfidence] = useState(0);
  const [personCount, setPersonCount] = useState(0);
  const [alerts, setAlerts]         = useState([]);
  const [uploadedVideo, setUploadedVideo] = useState(null);
  const [alertStats, setAlertStats] = useState(null);
  const [isViolenceActive, setIsViolenceActive] = useState(false);
  const [notificationsEnabled, setNotificationsEnabled] = useState(false);

  // ── Refs ─────────────────────────────────────────────────────────────────────
  const videoRef        = useRef(null);
  const fileInputRef    = useRef(null);
  const streamRef       = useRef(null);
  const wsRef           = useRef(null);
  const violenceTimerRef = useRef(null);
  const audioCtxRef     = useRef(null);
  const lastAlertTimeRef = useRef(0);
  const locationRef      = useRef(null);

  // ── Auth handlers ─────────────────────────────────────────────────────────────
  const handleLogin = (token, user) => {
    setAuth({ token, user });
  };

  const handleLogout = () => {
    stopWebcam();
    setAuth(null);
    setPage('app');
  };

  // ── On-mount effects ──────────────────────────────────────────────────────────
  useEffect(() => {
    // Browser notifications permission
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission().then(p => setNotificationsEnabled(p === 'granted'));
    } else if (Notification.permission === 'granted') {
      setNotificationsEnabled(true);
    }

    // Geolocation (ask once, send to backend)
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          const loc = {
            lat: pos.coords.latitude,
            lng: pos.coords.longitude,
            accuracy: pos.coords.accuracy
          };
          locationRef.current = loc;
          fetch(`${API_URL}/update-location`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(loc)
          }).catch(() => {});
        },
        () => {} // silently ignore if denied
      );
    }

    fetchAlerts();
    fetchAlertStats();

    return () => {
      stopWebcam();
      if (wsRef.current) wsRef.current.close();
      if (violenceTimerRef.current) clearTimeout(violenceTimerRef.current);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Backend data fetchers ─────────────────────────────────────────────────────
  const fetchAlertStats = async () => {
    try {
      const res = await fetch(`${API_URL}/alerts/statistics`);
      if (res.ok) setAlertStats(await res.json());
    } catch {}
  };

  const fetchAlerts = async () => {
    try {
      const res = await fetch(`${API_URL}/alerts`);
      if (res.ok) {
        const data = await res.json();
        const loaded = data.alerts.slice(0, 10).map(a => ({
          id: a.id || String(a.timestamp),
          time: new Date((a.timestamp || 0) * 1000).toLocaleTimeString(),
          confidence: a.confidence,
          source: a.source || 'system',
          screenshot: null,
          location: a.location || null
        }));
        setAlerts(loaded);
      }
    } catch {}
  };

  // ── Audio beep ────────────────────────────────────────────────────────────────
  const playAlertSound = () => {
    try {
      if (!audioCtxRef.current) audioCtxRef.current = new window.AudioContext();
      const ctx = audioCtxRef.current;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type = 'square';
      osc.frequency.setValueAtTime(880, ctx.currentTime);
      gain.gain.setValueAtTime(0.3, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.5);
    } catch {}
  };

  // ── Violence flash on video container ─────────────────────────────────────────
  const triggerViolenceFlash = () => {
    setIsViolenceActive(true);
    if (violenceTimerRef.current) clearTimeout(violenceTimerRef.current);
    violenceTimerRef.current = setTimeout(() => setIsViolenceActive(false), 3000);
  };

  // ── Webcam ────────────────────────────────────────────────────────────────────
  const startWebcam = async () => {
    try {
      setUploadedVideo(null);
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });

      // videoRef is always in the DOM — assign stream immediately
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;

        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play()
            .then(() => {
              setIsWebcamActive(true);
              setTimeout(() => connectWebSocket(), 1000);
            })
            .catch(err => console.error('Video play error:', err));
        };
      }
    } catch (error) {
      console.error('Webcam error:', error);
    }
  };

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
    setIsWebcamActive(false);
    setStatus('Monitoring...');
    setConfidence(0);
    setPersonCount(0);
    setIsViolenceActive(false);
  };

  // ── WebSocket ─────────────────────────────────────────────────────────────────
  const connectWebSocket = () => {
    const ws = new WebSocket(`ws://localhost:8000/ws`);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setTimeout(() => captureAndSendFrame(), 500);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        // Backend confirms violence only after 3 consecutive frames above threshold,
        // then immediately resets the frame buffer and enters a ~5 s lockout to
        // prevent buffer contamination from causing sticky false positives.
        if (data.violence_detected) {
          setStatus('Violence Detected');
          setConfidence(data.confidence);
          addAlert(data.confidence, 'webcam');
          triggerViolenceFlash();
        } else {
          setStatus('Monitoring...');
          setConfidence(data.confidence);
          // Let the 3-second visual timer (triggerViolenceFlash) clear the overlay
          // naturally — don't force-clear it here so the red flash remains visible.
        }
        setPersonCount(data.person_count || 0);
        if (data.alert_triggered) fetchAlertStats();
        setTimeout(() => captureAndSendFrame(), 200);
      } catch {}
    };

    ws.onerror = () => console.error('WebSocket error');
    ws.onclose = () => console.log('WebSocket closed');
    wsRef.current = ws;
  };

  // ── Frame capture & send ──────────────────────────────────────────────────────
  const captureAndSendFrame = () => {
    if (!videoRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    const canvas = document.createElement('canvas');
    canvas.width  = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    if (canvas.width === 0 || canvas.height === 0) {
      setTimeout(() => captureAndSendFrame(), 100);
      return;
    }
    canvas.getContext('2d').drawImage(videoRef.current, 0, 0);
    canvas.toBlob((blob) => {
      if (blob && wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send(blob);
    }, 'image/jpeg', 0.8);
  };

  const captureScreenshot = () => {
    if (!videoRef.current || videoRef.current.videoWidth === 0) return null;
    const canvas = document.createElement('canvas');
    canvas.width  = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext('2d').drawImage(videoRef.current, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.9);
  };

  // ── Alert helpers ─────────────────────────────────────────────────────────────
  const addAlert = (conf, source = 'webcam') => {
    const now = Date.now();
    if (now - lastAlertTimeRef.current < 5000) return; // 5-second frontend cooldown
    lastAlertTimeRef.current = now;

    const screenshot = captureScreenshot();
    setAlerts(prev => [{
      id: now,
      time: new Date(now).toLocaleTimeString(),
      confidence: conf,
      source,
      screenshot,
      location: locationRef.current
    }, ...prev].slice(0, 10));

    playAlertSound();
    if (notificationsEnabled) {
      new Notification('Violence Detected!', {
        body: `Confidence: ${(conf * 100).toFixed(1)}%  •  ${new Date(now).toLocaleTimeString()}`,
        requireInteraction: true
      });
    }
  };

  // ── Video file upload ─────────────────────────────────────────────────────────
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setUploadedVideo(URL.createObjectURL(file));
    setIsWebcamActive(false);

    const formData = new FormData();
    formData.append('file', file);

    try {
      setStatus('Processing...');
      const res = await fetch(`${API_URL}/upload-video`, { method: 'POST', body: formData });
      if (!res.ok) { setStatus('Monitoring...'); return; }
      const data = await res.json();
      setStatus(data.violence_detected ? 'Violence Detected' : 'Monitoring...');
      setConfidence(data.confidence);
      if (data.violence_detected) {
        addAlert(data.confidence, 'video-upload');
        triggerViolenceFlash();
        // Backend already logged the alert — refresh stats so the panel updates
        fetchAlertStats();
      }
    } catch {
      setStatus('Monitoring...');
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // ── Page routing ──────────────────────────────────────────────────────────────
  if (!auth) {
    return <LoginPage onLogin={handleLogin} />;
  }

  if (page === 'admin') {
    return (
      <AdminDashboard
        auth={auth}
        onBack={() => setPage('app')}
        onLogout={handleLogout}
      />
    );
  }

  // ── Main detection view ───────────────────────────────────────────────────────
  return (
    <div className="app">
      {/* Header */}
      <div className="header">
        <div className="header-row">
          <div>
            <h1>Violence Detection System</h1>
            <p>Real-time monitoring and video analysis</p>
          </div>
          <div className="header-nav">
            <span className="header-user">
              {auth.user.username} &nbsp;<em>({auth.user.role})</em>
            </span>
            {auth.user.role === 'admin' && (
              <button className="btn btn-secondary" onClick={() => setPage('admin')}>
                Admin Dashboard
              </button>
            )}
            <button className="btn btn-secondary" onClick={handleLogout}>
              Logout
            </button>
          </div>
        </div>
      </div>

      <div className="main-content">
        {/* Video section */}
        <div className="video-section">
          <div className={`video-container${isViolenceActive ? ' violence-active' : ''}`}>
            {isViolenceActive && (
              <div className="violence-overlay">
                <span>VIOLENCE DETECTED</span>
              </div>
            )}

            {/*
              Always rendered — this is the webcam feed.
              Hidden via CSS when not active so videoRef.current is always accessible.
            */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{
                display: isWebcamActive ? 'block' : 'none',
                width: '100%', height: '100%', objectFit: 'contain'
              }}
            />

            {/* Uploaded video */}
            {!isWebcamActive && uploadedVideo && (
              <video
                key={uploadedVideo}
                src={uploadedVideo}
                controls
                autoPlay
                loop
                muted
                playsInline
                style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
              />
            )}

            {/* Placeholder */}
            {!isWebcamActive && !uploadedVideo && (
              <div className="placeholder">
                Camera feed or uploaded video will appear here
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="controls">
            <button
              className="btn btn-primary"
              onClick={startWebcam}
              disabled={isWebcamActive}
            >
              Start Camera
            </button>
            <button
              className="btn btn-secondary"
              onClick={stopWebcam}
              disabled={!isWebcamActive}
            >
              Stop Camera
            </button>
            <button
              className="btn btn-primary"
              onClick={() => fileInputRef.current.click()}
              disabled={isWebcamActive}
            >
              Upload Video
            </button>
            <input
              ref={fileInputRef}
              type="file"
              className="file-input"
              accept="video/*,.avi,.mp4,.mov,.mkv,.flv,.wmv"
              onChange={handleFileUpload}
            />
          </div>
        </div>

        {/* Status panel */}
        <div className="status-section">
          <h2>Status</h2>

          <div className="status-item">
            <label>Detection Status</label>
            <div className={`value ${status === 'Violence Detected' ? 'status-warning' : 'status-safe'}`}>
              {status}
            </div>
          </div>

          <div className="status-item">
            <label>Confidence</label>
            <div className="value">{(confidence * 100).toFixed(1)}%</div>
            <div className="confidence-bar">
              <div
                className={`confidence-fill ${
                  confidence >= 0.7 ? 'high' : confidence >= 0.4 ? 'medium' : 'low'
                }`}
                style={{ width: `${(confidence * 100).toFixed(1)}%` }}
              />
            </div>
          </div>

          <div className="status-item">
            <label>Persons Detected</label>
            <div className="value">{personCount}</div>
          </div>

          {locationRef.current && (
            <div className="status-item">
              <label>Camera Location</label>
              <div style={{ fontSize: 13, color: '#555', marginTop: 4 }}>
                <a
                  href={`https://www.google.com/maps?q=${locationRef.current.lat},${locationRef.current.lng}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="loc-link"
                >
                  {Number(locationRef.current.lat).toFixed(4)}°,{' '}
                  {Number(locationRef.current.lng).toFixed(4)}°
                </a>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Alerts section */}
      <div className="alerts-section">
        <div className="alerts-header">
          <h2>
            Recent Alerts
            {alerts.length > 0 && (
              <span className="alert-badge">{alerts.length}</span>
            )}
          </h2>
          {alerts.length > 0 && (
            <button className="btn btn-danger" onClick={() => setAlerts([])}>
              Clear
            </button>
          )}
        </div>

        {/* Summary stats */}
        {alertStats && (
          <div className="alert-stats">
            <div className="stat-card">
              <label>Total Alerts</label>
              <div className="stat-value">{alertStats.total_alerts}</div>
            </div>
            <div className="stat-card">
              <label>Last 24h</label>
              <div className="stat-value">{alertStats.recent_alerts_24h}</div>
            </div>
            <div className="stat-card">
              <label>Avg Confidence</label>
              <div className="stat-value">
                {alertStats.average_confidence > 0
                  ? `${(alertStats.average_confidence * 100).toFixed(1)}%`
                  : '—'}
              </div>
            </div>
          </div>
        )}

        {alerts.length === 0 ? (
          <div className="no-alerts">No alerts yet</div>
        ) : (
          alerts.map(alert => (
            <div key={alert.id} className="alert-item">
              <div className="alert-item-body">
                {alert.screenshot && (
                  <img
                    src={alert.screenshot}
                    alt="Alert screenshot"
                    className="alert-thumbnail"
                  />
                )}
                <div className="alert-info">
                  <div className="time">{alert.time}</div>
                  <div className="message">Violence Detected</div>
                  <div className="confidence">
                    Confidence: {(alert.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="alert-source">
                    Source: {(alert.source || 'webcam').replace('-', ' ')}
                    {alert.location && alert.location.lat && (
                      <> &nbsp;&bull;&nbsp;
                        <a
                          href={`https://www.google.com/maps?q=${alert.location.lat},${alert.location.lng}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="loc-link"
                        >
                          View location
                        </a>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default App;
