import React from "react";
import {
  TrendingUp,
  TrendingDown,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Activity,
  DollarSign,
  Clock,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";

// Mock data — in production, fetch from API
const trendData = [
  { date: "Mon", pass_rate: 0.82, avg_score: 0.78, regressions: 2 },
  { date: "Tue", pass_rate: 0.85, avg_score: 0.80, regressions: 1 },
  { date: "Wed", pass_rate: 0.79, avg_score: 0.75, regressions: 4 },
  { date: "Thu", pass_rate: 0.88, avg_score: 0.83, regressions: 0 },
  { date: "Fri", pass_rate: 0.91, avg_score: 0.87, regressions: 0 },
  { date: "Sat", pass_rate: 0.86, avg_score: 0.81, regressions: 1 },
  { date: "Sun", pass_rate: 0.89, avg_score: 0.85, regressions: 0 },
];

const metricBreakdown = [
  { name: "Exact Match", score: 0.92, threshold: 0.80 },
  { name: "Semantic Sim", score: 0.78, threshold: 0.70 },
  { name: "Faithfulness", score: 0.85, threshold: 0.75 },
  { name: "Tool Accuracy", score: 0.68, threshold: 0.80 },
  { name: "Task Complete", score: 0.91, threshold: 0.85 },
];

const recentRuns = [
  { id: "run_001", name: "Nightly Regression", status: "passed", pass_rate: 0.89, cases: 156, time: "2h ago" },
  { id: "run_002", name: "PR #234 - Tool Updates", status: "failed", pass_rate: 0.72, cases: 156, time: "5h ago" },
  { id: "run_003", name: "Model A/B Test", status: "warning", pass_rate: 0.81, cases: 200, time: "8h ago" },
  { id: "run_004", name: "Weekly Benchmark", status: "passed", pass_rate: 0.93, cases: 500, time: "1d ago" },
];

function StatCard({
  title,
  value,
  change,
  changeType,
  icon: Icon,
}: {
  title: string;
  value: string;
  change: string;
  changeType: "positive" | "negative" | "neutral";
  icon: React.ElementType;
}) {
  const changeColors = {
    positive: "text-success-600",
    negative: "text-danger-600",
    neutral: "text-slate-500",
  };

  const ChangeIcon = changeType === "positive" ? TrendingUp : changeType === "negative" ? TrendingDown : Activity;

  return (
    <div className="card">
      <div className="flex items-start justify-between">
        <div>
          <p className="metric-label">{title}</p>
          <p className="metric-value mt-1">{value}</p>
        </div>
        <div className="p-2 bg-slate-50 rounded-lg">
          <Icon className="w-5 h-5 text-slate-600" />
        </div>
      </div>
      <div className={`flex items-center gap-1 mt-4 text-sm ${changeColors[changeType]}`}>
        <ChangeIcon className="w-4 h-4" />
        <span className="font-medium">{change}</span>
        <span className="text-slate-400 ml-1">vs last week</span>
      </div>
    </div>
  );
}

export default function Overview() {
  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-slate-900">Dashboard Overview</h1>
        <p className="text-slate-500 mt-1">
          Monitor agent evaluation metrics and regression trends
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          title="Pass Rate"
          value="89.2%"
          change="+2.4%"
          changeType="positive"
          icon={CheckCircle2}
        />
        <StatCard
          title="Avg Score"
          value="0.842"
          change="+0.05"
          changeType="positive"
          icon={Activity}
        />
        <StatCard
          title="Regressions"
          value="3"
          change="-5"
          changeType="positive"
          icon={AlertTriangle}
        />
        <StatCard
          title="Eval Cost"
          value="$12.40"
          change="+$3.20"
          changeType="negative"
          icon={DollarSign}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Trend Chart */}
        <div className="card lg:col-span-2">
          <h3 className="text-lg font-semibold text-slate-900 mb-4">Pass Rate Trend</h3>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={trendData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" stroke="#64748b" fontSize={12} />
              <YAxis domain={[0.6, 1.0]} stroke="#64748b" fontSize={12} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
              <Tooltip
                formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
                contentStyle={{ borderRadius: "8px", border: "1px solid #e2e8f0" }}
              />
              <Line
                type="monotone"
                dataKey="pass_rate"
                stroke="#0ea5e9"
                strokeWidth={2}
                dot={{ fill: "#0ea5e9", r: 4 }}
                name="Pass Rate"
              />
              <Line
                type="monotone"
                dataKey="avg_score"
                stroke="#22c55e"
                strokeWidth={2}
                dot={{ fill: "#22c55e", r: 4 }}
                name="Avg Score"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Metric Breakdown */}
        <div className="card">
          <h3 className="text-lg font-semibold text-slate-900 mb-4">Metric Scores</h3>
          <div className="space-y-4">
            {metricBreakdown.map((m) => (
              <div key={m.name}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-slate-700">{m.name}</span>
                  <span className={`font-medium ${m.score >= m.threshold ? "text-success-600" : "text-danger-600"}`}>
                    {(m.score * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all ${
                      m.score >= m.threshold ? "bg-success-500" : "bg-danger-500"
                    }`}
                    style={{ width: `${m.score * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent Evaluations */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-slate-900">Recent Evaluations</h3>
          <button className="text-sm text-primary-600 hover:text-primary-700 font-medium">
            View All
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-200">
                <th className="text-left text-xs font-medium text-slate-500 uppercase py-3">Run</th>
                <th className="text-left text-xs font-medium text-slate-500 uppercase py-3">Status</th>
                <th className="text-left text-xs font-medium text-slate-500 uppercase py-3">Pass Rate</th>
                <th className="text-left text-xs font-medium text-slate-500 uppercase py-3">Cases</th>
                <th className="text-left text-xs font-medium text-slate-500 uppercase py-3">Time</th>
              </tr>
            </thead>
            <tbody>
              {recentRuns.map((run) => (
                <tr key={run.id} className="border-b border-slate-100 last:border-0">
                  <td className="py-3">
                    <div>
                      <p className="text-sm font-medium text-slate-900">{run.name}</p>
                      <p className="text-xs text-slate-500">{run.id}</p>
                    </div>
                  </td>
                  <td className="py-3">
                    <span
                      className={`badge ${
                        run.status === "passed"
                          ? "badge-success"
                          : run.status === "failed"
                          ? "badge-danger"
                          : "badge-warning"
                      }`}
                    >
                      {run.status === "passed" && <CheckCircle2 className="w-3 h-3 mr-1" />}
                      {run.status === "failed" && <XCircle className="w-3 h-3 mr-1" />}
                      {run.status === "warning" && <AlertTriangle className="w-3 h-3 mr-1" />}
                      {run.status}
                    </span>
                  </td>
                  <td className="py-3">
                    <div className="flex items-center gap-2">
                      <div className="w-20 bg-slate-100 rounded-full h-1.5">
                        <div
                          className={`h-1.5 rounded-full ${
                            run.pass_rate >= 0.8 ? "bg-success-500" : "bg-danger-500"
                          }`}
                          style={{ width: `${run.pass_rate * 100}%` }}
                        />
                      </div>
                      <span className="text-sm text-slate-700">{(run.pass_rate * 100).toFixed(0)}%</span>
                    </div>
                  </td>
                  <td className="py-3 text-sm text-slate-700">{run.cases}</td>
                  <td className="py-3 text-sm text-slate-500">{run.time}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
