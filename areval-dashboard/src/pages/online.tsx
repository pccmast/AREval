import React, { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock,
  Timer,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { fetchOnlineHealth, fetchOnlineTrend, fetchOnlineAlerts, type HealthResponse, type TrendPoint, type AlertRecord } from "@/lib/apiClient";

const MOCK_TREND: TrendPoint[] = Array.from({ length: 24 }, (_, i) => ({
  time: `${String(i).padStart(2, "0")}:00`,
  pass_rate: 0.85 + Math.random() * 0.1 - 0.05,
  avg_score: 0.70 + Math.random() * 0.15,
  sample_count: 5 + Math.floor(Math.random() * 20),
}));

const MOCK_ALERTS: AlertRecord[] = [
  { type: "pass_rate_drop", severity: "warning", message: "Pass rate 0.62 < 0.70", current_value: 0.62, threshold_value: 0.70, timestamp: new Date().toISOString() },
];

function statusIcon(status: string) {
  switch (status) {
    case "healthy": return <CheckCircle className="w-5 h-5 text-success-500" />;
    case "degraded": return <AlertTriangle className="w-5 h-5 text-warning-500" />;
    case "critical": return <XCircle className="w-5 h-5 text-danger-500" />;
    default: return <Clock className="w-5 h-5 text-slate-400" />;
  }
}

function statusColor(status: string) {
  switch (status) {
    case "healthy": return "bg-success-50 border-success-500 text-success-600";
    case "degraded": return "bg-warning-50 border-warning-500 text-warning-600";
    case "critical": return "bg-danger-50 border-danger-500 text-danger-600";
    default: return "bg-slate-50 border-slate-400 text-slate-500";
  }
}

function alertSeverityColor(severity: string) {
  return severity === "critical"
    ? "border-l-danger-500 bg-danger-50"
    : "border-l-warning-500 bg-warning-50";
}

export default function OnlineMonitor() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [trend, setTrend] = useState<TrendPoint[]>([]);
  const [alerts, setAlerts] = useState<AlertRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const [h, t, a] = await Promise.all([
          fetchOnlineHealth(),
          fetchOnlineTrend(),
          fetchOnlineAlerts(),
        ]);
        if (h) setHealth(h);
        if (t && t.length) setTrend(t);
        if (a && a.length) setAlerts(a);
      } finally {
        setLoading(false);
      }
    }
    load();
    const interval = setInterval(load, 60000); // refresh every 60s
    return () => clearInterval(interval);
  }, []);

  const status = health?.status || "unknown";
  const trendData = trend.length > 0 ? trend : MOCK_TREND;
  const alertList = alerts.length > 0 ? alerts : (health?.active_alerts ? MOCK_ALERTS : []);

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Online Monitor</h1>
          <p className="text-slate-500 mt-1">Real-time quality monitoring & alerting</p>
        </div>
        <div className={`flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium ${statusColor(status)}`}>
          {statusIcon(status)}
          <span className="capitalize">{status}</span>
          {!loading && <span className="text-slate-400">· auto-refresh 60s</span>}
        </div>
      </div>

      {/* Stats cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="card">
          <p className="metric-label">Pass Rate</p>
          <p className="metric-value">
            {health ? `${(health.pass_rate * 100).toFixed(1)}%` : "—"}
          </p>
        </div>
        <div className="card">
          <p className="metric-label">Avg Score</p>
          <p className="metric-value">
            {health ? health.avg_score.toFixed(3) : "—"}
          </p>
        </div>
        <div className="card">
          <p className="metric-label">Samples (30m)</p>
          <p className="metric-value">
            {health ? health.sample_count : "—"}
          </p>
        </div>
        <div className="card">
          <p className="metric-label">Active Alerts</p>
          <p className={`metric-value ${health && health.active_alerts > 0 ? "text-danger-500" : ""}`}>
            {health ? health.active_alerts : "—"}
          </p>
        </div>
      </div>

      {/* Trend chart */}
      <div className="card mb-8">
        <h2 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
          <Activity className="w-4 h-4" />
          Pass Rate Trend (24h)
        </h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={trendData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="time" tick={{ fontSize: 12 }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
            <Tooltip formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
            <Line type="monotone" dataKey="pass_rate" stroke="#0284c7" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Alerts */}
      <div className="card">
        <h2 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4" />
          Recent Alerts
        </h2>
        {alertList.length === 0 ? (
          <p className="text-slate-400 text-sm">No alerts in the last 24 hours.</p>
        ) : (
          <div className="space-y-2">
            {alertList.map((a, i) => (
              <div key={i} className={`border-l-4 p-3 rounded text-sm ${alertSeverityColor(a.severity)}`}>
                <div className="flex items-center justify-between">
                  <span className="font-medium capitalize">{a.type.replace(/_/g, " ")}</span>
                  <span className="text-xs text-slate-500">{new Date(a.timestamp).toLocaleTimeString()}</span>
                </div>
                <p className="text-slate-600 mt-1">{a.message}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
