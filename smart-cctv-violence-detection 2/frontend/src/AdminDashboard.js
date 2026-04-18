import React, { useState, useEffect, useRef } from 'react';
import './Auth.css';

const API_URL = 'http://localhost:8000';

function AdminDashboard({ auth, onBack, onLogout }) {
  const [alerts, setAlerts]         = useState([]);
  const [stats, setStats]           = useState(null);
  const [sysStatus, setSysStatus]   = useState(null);
  const [loading, setLoading]       = useState(true);
  const [error, setError]           = useState('');
  const [testEmailMsg, setTestEmailMsg] = useState('');
  const [testAlertMsg, setTestAlertMsg] = useState('');
  const [testEmailLoading, setTestEmailLoading] = useState(false);
  const [testAlertLoading, setTestAlertLoading] = useState(false);

  // Keep headers in a ref so the fetch callbacks always see the latest value
  const headersRef = useRef({ Authorization: `Bearer ${auth.token}` });

  // ── Fetch all data ────────────────────────────────────────────────────────────
  const fetchData = async () => {
    setLoading(true);
    setError('');
    try {
      const [alertsRes, statsRes, statusRes] = await Promise.all([
        fetch(`${API_URL}/admin/alerts`,      { headers: headersRef.current }),
        fetch(`${API_URL}/alerts/statistics`, { headers: headersRef.current }),
        fetch(`${API_URL}/debug/status`)          // no auth needed
      ]);

      // Admin alerts
      if (alertsRes.status === 403) {
        setError('Access denied: your account does not have Admin role. Please register again as Admin.');
      } else if (alertsRes.status === 401) {
        setError('Session expired. Please log out and log in again.');
      } else if (!alertsRes.ok) {
        setError(`Server error (${alertsRes.status}) — is the backend running?`);
      } else {
        const data = await alertsRes.json();
        // Newest alert first
        const sorted = (data.alerts || []).slice().reverse();
        setAlerts(sorted);
      }

      // Stats (public endpoint — always try)
      if (statsRes.ok) setStats(await statsRes.json());

      // System status (diagnostic endpoint)
      if (statusRes.ok) setSysStatus(await statusRes.json());

    } catch (e) {
      setError('Cannot reach the backend. Make sure the server is running on port 8000.');
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Send test email ────────────────────────────────────────────────────────────
  const handleTestEmail = async () => {
    setTestEmailLoading(true);
    setTestEmailMsg('');
    try {
      const res = await fetch(`${API_URL}/test-email`, {
        method: 'POST',
        headers: headersRef.current
      });
      const data = await res.json();
      if (!res.ok) {
        setTestEmailMsg(`Failed: ${data.detail || 'Unknown error'}`);
      } else if (data.success) {
        setTestEmailMsg(`Email sent to: ${(data.attempted_to || []).join(', ') || 'no recipients'}`);
      } else {
        setTestEmailMsg('Email failed — check SMTP settings in backend/.env');
      }
    } catch {
      setTestEmailMsg('Cannot reach backend.');
    }
    setTestEmailLoading(false);
    setTimeout(() => setTestEmailMsg(''), 6000);
  };

  // ── Create test alert ─────────────────────────────────────────────────────────
  const handleTestAlert = async () => {
    setTestAlertLoading(true);
    setTestAlertMsg('');
    try {
      const res = await fetch(`${API_URL}/debug/create-test-alert`, {
        method: 'POST',
        headers: headersRef.current
      });
      const data = await res.json();
      if (!res.ok) {
        setTestAlertMsg(`Failed: ${data.detail || 'Unknown error'}`);
      } else {
        setTestAlertMsg('Test alert created — email sent, refreshing...');
        setTimeout(fetchData, 1500);
      }
    } catch {
      setTestAlertMsg('Cannot reach backend.');
    }
    setTestAlertLoading(false);
    setTimeout(() => setTestAlertMsg(''), 6000);
  };

  // ── Helpers ───────────────────────────────────────────────────────────────────
  const confidenceClass = (c) => {
    if (c >= 0.8) return 'conf-high';
    if (c >= 0.7) return 'conf-medium';
    return 'conf-low';
  };

  const formatDT = (isoString) => {
    if (!isoString) return { date: '—', time: '—' };
    const d = new Date(isoString);
    return { date: d.toLocaleDateString('en-GB'), time: d.toLocaleTimeString() };
  };

  const StatusDot = ({ ok }) => (
    <span style={{
      display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
      background: ok ? '#43a047' : '#e53935', marginRight: 6, verticalAlign: 'middle'
    }} />
  );

  return (
    <div className="dash-page">

      {/* ── Header ── */}
      <div className="dash-header">
        <div>
          <h1 className="dash-title">Admin Dashboard</h1>
          <p className="dash-subtitle">
            Welcome, <strong>{auth.user.username}</strong> &mdash; Alert monitoring panel
          </p>
        </div>
        <div className="dash-header-actions">
          <button className="btn btn-secondary" onClick={onBack}>Back to Monitor</button>
          <button className="btn btn-danger"    onClick={onLogout}>Logout</button>
        </div>
      </div>

      {/* ── Error banner ── */}
      {error && (
        <div className="dash-error-banner">
          {error}
        </div>
      )}

      {/* ── System status row ── */}
      {sysStatus && (
        <div className="dash-section" style={{ marginBottom: 20 }}>
          <div className="dash-section-header" style={{ marginBottom: 12 }}>
            <h2>System Status</h2>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              {/* Test email button */}
              <button
                className="btn btn-secondary"
                style={{ fontSize: 13, padding: '6px 14px' }}
                onClick={handleTestEmail}
                disabled={testEmailLoading}
              >
                {testEmailLoading ? 'Sending...' : 'Send Test Email'}
              </button>
              {/* Test alert button */}
              <button
                className="btn btn-primary"
                style={{ fontSize: 13, padding: '6px 14px' }}
                onClick={handleTestAlert}
                disabled={testAlertLoading}
              >
                {testAlertLoading ? 'Creating...' : 'Create Test Alert'}
              </button>
            </div>
          </div>

          {/* Feedback messages */}
          {testEmailMsg && (
            <div className={`dash-feedback ${testEmailMsg.startsWith('Email sent') ? 'ok' : 'err'}`}>
              {testEmailMsg}
            </div>
          )}
          {testAlertMsg && (
            <div className={`dash-feedback ${testAlertMsg.startsWith('Test alert') ? 'ok' : 'err'}`}>
              {testAlertMsg}
            </div>
          )}

          <div className="sys-status-grid">
            <div className="sys-status-item">
              <StatusDot ok={sysStatus.model_loaded} />
              <span>Detection Model</span>
              <strong>{sysStatus.model_loaded ? 'Loaded' : 'NOT LOADED — no detection'}</strong>
            </div>
            <div className="sys-status-item">
              <StatusDot ok={sysStatus.email_sender_configured} />
              <span>Email (SMTP)</span>
              <strong>{sysStatus.email_sender_configured ? sysStatus.sender_email : 'Not configured'}</strong>
            </div>
            <div className="sys-status-item">
              <StatusDot ok={sysStatus.total_admins_in_db > 0} />
              <span>Admin accounts</span>
              <strong>
                {sysStatus.total_admins_in_db} admin
                {sysStatus.total_admins_in_db !== 1 ? 's' : ''}
                {sysStatus.admin_emails.length > 0 && (
                  <span style={{ fontWeight: 400, color: '#888', fontSize: 12 }}>
                    {' '}({sysStatus.admin_emails.join(', ')})
                  </span>
                )}
              </strong>
            </div>
            <div className="sys-status-item">
              <StatusDot ok={sysStatus.total_alerts_logged > 0} />
              <span>Alerts logged</span>
              <strong>{sysStatus.total_alerts_logged} total</strong>
            </div>
          </div>
        </div>
      )}

      {/* ── Stats cards ── */}
      {stats && (
        <div className="dash-stats">
          <div className="dash-stat-card">
            <div className="dash-stat-value">{stats.total_alerts}</div>
            <div className="dash-stat-label">Total Alerts</div>
          </div>
          <div className="dash-stat-card">
            <div className="dash-stat-value">{stats.recent_alerts_24h}</div>
            <div className="dash-stat-label">Last 24 Hours</div>
          </div>
          <div className="dash-stat-card">
            <div className="dash-stat-value">
              {stats.average_confidence > 0
                ? `${(stats.average_confidence * 100).toFixed(1)}%`
                : '—'}
            </div>
            <div className="dash-stat-label">Avg Confidence</div>
          </div>
          <div className="dash-stat-card">
            <div className="dash-stat-value">
              {stats.most_recent
                ? new Date(stats.most_recent.datetime).toLocaleDateString('en-GB')
                : '—'}
            </div>
            <div className="dash-stat-label">Last Alert Date</div>
          </div>
        </div>
      )}

      {/* ── Alert table ── */}
      <div className="dash-section">
        <div className="dash-section-header">
          <h2>Alert Log</h2>
          <button
            className="btn btn-secondary"
            style={{ fontSize: 13, padding: '6px 14px' }}
            onClick={fetchData}
          >
            Refresh
          </button>
        </div>

        {loading ? (
          <div className="no-alerts">Loading...</div>
        ) : error ? (
          <div className="no-alerts" style={{ color: '#c62828' }}>
            Could not load alerts. See the error above.
          </div>
        ) : alerts.length === 0 ? (
          <div className="no-alerts">
            No alerts recorded yet.
            {sysStatus && !sysStatus.model_loaded && (
              <div style={{ color: '#e65100', marginTop: 8, fontSize: 13 }}>
                The detection model is not loaded — place <code>best_model.pth</code> in <code>backend/models/</code> and restart the server.
              </div>
            )}
          </div>
        ) : (
          <div className="dash-table-wrapper">
            <table className="dash-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Date</th>
                  <th>Time</th>
                  <th>Confidence</th>
                  <th>Source</th>
                  <th>Location</th>
                </tr>
              </thead>
              <tbody>
                {alerts.map((alert, idx) => {
                  const a   = alert.to_dict ? alert.to_dict() : alert;
                  const dt  = formatDT(a.datetime);
                  const conf = a.confidence || 0;
                  const loc  = a.location;

                  return (
                    <tr key={a.id || idx}>
                      <td className="dash-td-num">{idx + 1}</td>
                      <td>{dt.date}</td>
                      <td style={{ color: '#555' }}>{dt.time}</td>
                      <td>
                        <span className={`conf-badge ${confidenceClass(conf)}`}>
                          {(conf * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td style={{ color: '#555', textTransform: 'capitalize' }}>
                        {(a.source || 'webcam').replace(/-/g, ' ')}
                      </td>
                      <td>
                        {loc && loc.lat ? (
                          <a
                            href={`https://www.google.com/maps?q=${loc.lat},${loc.lng}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="loc-link"
                          >
                            {Number(loc.lat).toFixed(4)}°, {Number(loc.lng).toFixed(4)}°
                          </a>
                        ) : (
                          <span style={{ color: '#bbb' }}>Not available</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default AdminDashboard;
