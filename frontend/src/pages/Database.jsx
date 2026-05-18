import React, { useEffect, useState } from 'react';
import { api } from '../api.js';
import DigitPreview from '../components/DigitPreview.jsx';

export default function Database() {
  const [stats, setStats] = useState(null);
  const [samples, setSamples] = useState(null);
  const [feedbackRows, setFeedbackRows] = useState([]);
  const [feedbackStatus, setFeedbackStatus] = useState('pending');
  const [label, setLabel] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [page, setPage] = useState(0);
  const [message, setMessage] = useState('');

  async function load(nextPage = page) {
    const [s, rows, feedback] = await Promise.all([
      api.dbStats(),
      api.dbSamples({ page: nextPage, size: 20, label, status: statusFilter }),
      api.listFeedback(feedbackStatus),
    ]);
    setStats(s);
    setSamples(rows);
    setFeedbackRows(feedback);
    setPage(nextPage);
  }

  useEffect(() => { load(0).catch(err => setMessage(err.message)); }, []);

  async function search() {
    await load(0);
  }

  async function updateLabel(id) {
    const value = prompt('New true label 0-9:');
    if (value === null) return;
    await api.updateSample(id, Number(value));
    setMessage(`Updated sample ${id}`);
    await load();
  }

  async function deleteSample(id) {
    if (!confirm(`Delete sample ${id}?`)) return;
    await api.deleteSample(id);
    setMessage(`Deleted sample ${id}`);
    await load();
  }

  return (
    <div className="stack">
      <section className="card">
        <h2>Database Stats</h2>
        {stats ? (
          <>
            <div className="metrics">
              <Metric label="Total" value={stats.totalSamples} />
              <Metric label="Accepted" value={stats.acceptedSamples} />
              <Metric label="Pending feedback" value={stats.pendingFeedback} />
              <Metric label="Dataset version" value={stats.datasetVersion} />
            </div>
            <h3>Distribution</h3>
            <div className="bars">
              {stats.distribution.map(row => <div key={row.label}><span>{row.label}</span><div><i style={{ width: `${Math.min(100, row.count / Math.max(1, stats.acceptedSamples) * 1000)}%` }} /></div><b>{row.count}</b></div>)}
            </div>
          </>
        ) : <p>Loading...</p>}
      </section>

      <section className="card">
        <h2>Query Samples</h2>
        <div className="form-row">
          <label>Label</label>
          <select value={label} onChange={e => setLabel(e.target.value)}>
            <option value="">All</option>
            {[0,1,2,3,4,5,6,7,8,9].map(n => <option key={n} value={n}>{n}</option>)}
          </select>
          <label>Status</label>
          <select value={statusFilter} onChange={e => setStatusFilter(e.target.value)}>
            <option value="accepted">accepted</option>
            <option value="pending">pending</option>
            <option value="rejected">rejected</option>
            <option value="">all</option>
          </select>
          <button onClick={search}>Search</button>
        </div>

        {samples && <p>Total matched: {samples.total}</p>}
        {samples && (
          <div className="sample-grid">
            {samples.rows.map(row => (
              <div className="sample-card" key={row.id}>
                <DigitPreview pixels={row.pixels} size={84} />
                <p>ID #{row.id}</p>
                <p>Label: {row.label}</p>
                <p>{row.source} · {row.status}</p>
                <button onClick={() => updateLabel(row.id)}>Edit label</button>
                <button className="danger" onClick={() => deleteSample(row.id)}>Delete</button>
              </div>
            ))}
          </div>
        )}
        <div className="pager">
          <button disabled={page <= 0} onClick={() => load(page - 1)}>Prev</button>
          <span>Page {page}</span>
          <button disabled={!samples || (page + 1) * samples.size >= samples.total} onClick={() => load(page + 1)}>Next</button>
        </div>
      </section>

      <section className="card">
        <h2>Feedback Submissions</h2>
        <div className="form-row">
          <label>Status</label>
          <select value={feedbackStatus} onChange={e => setFeedbackStatus(e.target.value)}>
            <option value="pending">pending</option>
            <option value="rejected">rejected</option>
            <option value="accepted">accepted</option>
          </select>
          <button onClick={() => load(0)}>Load feedback</button>
        </div>

        {feedbackRows.length === 0 ? (
          <p>No {feedbackStatus} feedback submissions.</p>
        ) : (
          <div className="feedback-grid">
            {feedbackRows.map(row => (
              <div className="feedback-card" key={row.id}>
                <DigitPreview pixels={row.pixels} size={84} />
                <p>ID #{row.id}</p>
                <p>Predicted: {row.predictedLabel ?? 'N/A'}</p>
                <p>True: {row.trueLabel ?? 'N/A'}</p>
                <p>Status: {row.status}</p>
              </div>
            ))}
          </div>
        )}
      </section>
      <p className="status">{message}</p>
    </div>
  );
}

function Metric({ label, value }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}
