import React, { useEffect, useState } from 'react';
import { api } from '../api.js';
import DigitPreview from '../components/DigitPreview.jsx';

export default function Model() {
  const [dashboard, setDashboard] = useState(null);
  const [feedback, setFeedback] = useState([]);
  const [sampleCount, setSampleCount] = useState(500);
  const [method, setMethod] = useState('kd_tree');
  const [kText, setKText] = useState('1,3,5,7');
  const [tuneResult, setTuneResult] = useState(null);
  const [status, setStatus] = useState('');

  async function load() {
    const [dash, fb] = await Promise.all([api.dashboard(), api.listFeedback('pending')]);
    setDashboard(dash);
    setFeedback(fb);
  }

  useEffect(() => { load().catch(err => setStatus(err.message)); }, []);

  async function runTune() {
    setStatus('Tuning...');
    try {
      const kValues = kText.split(',').map(v => Number(v.trim())).filter(v => Number.isInteger(v) && v > 0);
      const result = await api.tune({ sampleCount: Number(sampleCount), method, kValues });
      setTuneResult(result);
      setStatus('Tune done.');
      await load();
    } catch (err) {
      setStatus(`Tune failed: ${err.message}`);
    }
  }

  async function activate(row) {
    try {
      await api.activateTune({ jobId: tuneResult.jobId, k: row.k, method: row.method });
      setStatus(`Activated k=${row.k}, method=${row.method}`);
      await load();
    } catch (err) {
      setStatus(`Activate failed: ${err.message}`);
    }
  }

  async function accept(id) {
    await api.acceptFeedback(id);
    await load();
  }

  async function reject(id) {
    await api.rejectFeedback(id);
    await load();
  }

  const grouped = feedback.reduce((acc, item) => {
    const key = item.trueLabel;
    acc[key] = acc[key] || [];
    acc[key].push(item);
    return acc;
  }, {});

  return (
    <div className="stack">
      <section className="card">
        <h2>Model Dashboard</h2>
        {dashboard ? (
          <div className="metrics">
            <Metric label="Predictions" value={dashboard.totalPredictions} />
            <Metric label="Accuracy" value={`${(dashboard.accuracy * 100).toFixed(1)}%`} />
            <Metric label="F1-score" value={dashboard.f1Score.toFixed(3)} />
            <Metric label="Avg response" value={`${dashboard.avgResponseTimeMs.toFixed(1)} ms`} />
            <Metric label="Accepted samples" value={dashboard.acceptedSamples} />
            <Metric label="Pending feedback" value={dashboard.pendingFeedback} />
          </div>
        ) : <p>Loading...</p>}
        {dashboard && <p>Active: k={dashboard.activeModel.k}, method={dashboard.activeModel.method}, dataset v{dashboard.datasetVersion}</p>}
      </section>

      <section className="card">
        <h2>Confusion Matrix</h2>
        {dashboard && <ConfusionMatrix matrix={dashboard.confusionMatrix} />}
      </section>

      <section className="card">
        <h2>Tuning</h2>
        <div className="form-row">
          <label>Sample count</label>
          <input type="number" value={sampleCount} onChange={e => setSampleCount(e.target.value)} />
          <label>Method</label>
          <select value={method} onChange={e => setMethod(e.target.value)}>
            <option value="kd_tree">kd-tree</option>
            <option value="lsh">LSH</option>
          </select>
          <label>k values</label>
          <input value={kText} onChange={e => setKText(e.target.value)} />
          <button onClick={runTune}>Run tune</button>
        </div>
        {tuneResult && (
          <table>
            <thead><tr><th>Rank</th><th>k</th><th>Method</th><th>Accuracy</th><th>F1</th><th>Latency</th><th>Train</th><th>Eval</th><th>Action</th></tr></thead>
            <tbody>
              {tuneResult.topResults.map(row => (
                <tr key={`${row.k}-${row.method}`}>
                  <td>{row.rank}</td><td>{row.k}</td><td>{row.method}</td>
                  <td>{(row.accuracy * 100).toFixed(2)}%</td>
                  <td>{row.f1Score.toFixed(3)}</td>
                  <td>{row.avgLatencyMs.toFixed(2)} ms</td>
                  <td>{row.trainingSamples}</td>
                  <td>{row.evaluatedSamples}</td>
                  <td><button onClick={() => activate(row)}>Use this k</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="card">
        <h2>Pending Feedback</h2>
        {Object.keys(grouped).length === 0 && <p>No pending feedback.</p>}
        {Object.entries(grouped).map(([label, items]) => (
          <div key={label} className="feedback-group">
            <h3>True label: {label}</h3>
            <div className="feedback-grid">
              {items.map(item => (
                <div className="feedback-card" key={item.id}>
                  <DigitPreview pixels={item.pixels} size={84} />
                  <p>ID #{item.id}</p>
                  <p>Pred: {item.predictedLabel ?? 'N/A'}</p>
                  <button onClick={() => accept(item.id)}>Accept</button>
                  <button className="danger" onClick={() => reject(item.id)}>Reject</button>
                </div>
              ))}
            </div>
          </div>
        ))}
      </section>
      <p className="status">{status}</p>
    </div>
  );
}

function Metric({ label, value }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

function ConfusionMatrix({ matrix }) {
  return (
    <table className="matrix">
      <thead><tr><th>true\\pred</th>{[0,1,2,3,4,5,6,7,8,9].map(n => <th key={n}>{n}</th>)}</tr></thead>
      <tbody>
        {matrix.map((row, i) => <tr key={i}><th>{i}</th>{row.map((v, j) => <td key={j}>{v}</td>)}</tr>)}
      </tbody>
    </table>
  );
}
