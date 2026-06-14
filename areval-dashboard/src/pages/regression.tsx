import React from "react";
import { GitCompare, TrendingDown, AlertCircle, CheckCircle2 } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

const regressionHistory = [
  { run: "Run #1", baseline: 0.85, current: 0.82, delta: -0.03, regressions: 5 },
  { run: "Run #2", baseline: 0.85, current: 0.88, delta: 0.03, regressions: 0 },
  { run: "Run #3", baseline: 0.88, current: 0.79, delta: -0.09, regressions: 12 },
  { run: "Run #4", baseline: 0.79, current: 0.81, delta: 0.02, regressions: 2 },
  { run: "Run #5", baseline: 0.81, current: 0.91, delta: 0.10, regressions: 0 },
  { run: "Run #6", baseline: 0.91, current: 0.86, delta: -0.05, regressions: 8 },
  { run: "Run #7", baseline: 0.86, current: 0.89, delta: 0.03, regressions: 1 },
];

const affectedTests = [
  { name: "search_agent_basic", baseline: 0.92, current: 0.74, delta: -0.18, severity: "critical" },
  { name: "tool_call_weather", baseline: 0.88, current: 0.65, delta: -0.23, severity: "critical" },
  { name: "rag_faithfulness_q3", baseline: 0.85, current: 0.72, delta: -0.13, severity: "major" },
  { name: "code_gen_python", baseline: 0.79, current: 0.68, delta: -0.11, severity: "major" },
  { name: "multi_step_planning", baseline: 0.90, current: 0.82, delta: -0.08, severity: "minor" },
];

export default function Regression() {
  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-slate-900">Regression Analysis</h1>
        <p className="text-slate-500 mt-1">Detect and analyze performance regressions</p>
      </div>

      {/* Alerts */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="card border-l-4 border-l-danger-500">
          <div className="flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-danger-500" />
            <div>
              <p className="metric-value text-danger-600">5</p>
              <p className="metric-label">Critical Regressions</p>
            </div>
          </div>
        </div>
        <div className="card border-l-4 border-l-warning-500">
          <div className="flex items-center gap-3">
            <TrendingDown className="w-5 h-5 text-warning-500" />
            <div>
              <p className="metric-value text-warning-600">-4.2%</p>
              <p className="metric-label">Avg Score Change</p>
            </div>
          </div>
        </div>
        <div className="card border-l-4 border-l-success-500">
          <div className="flex items-center gap-3">
            <CheckCircle2 className="w-5 h-5 text-success-500" />
            <div>
              <p className="metric-value text-success-600">142/156</p>
              <p className="metric-label">Tests Stable</p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Trend Chart */}
        <div className="card">
          <h3 className="text-lg font-semibold text-slate-900 mb-4">Score Delta Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={regressionHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="run" fontSize={11} stroke="#64748b" />
              <YAxis stroke="#64748b" fontSize={11} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
              <Tooltip formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
              <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="3 3" />
              <Line
                type="monotone"
                dataKey="delta"
                stroke="#ef4444"
                strokeWidth={2}
                dot={{ r: 5 }}
                name="Delta"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Affected Tests */}
        <div className="card">
          <h3 className="text-lg font-semibold text-slate-900 mb-4">Top Regressions</h3>
          <div className="space-y-3">
            {affectedTests.map((test) => (
              <div key={test.name} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg">
                <div className="flex-1">
                  <p className="text-sm font-medium text-slate-900">{test.name}</p>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-xs text-slate-500">{test.baseline.toFixed(2)}</span>
                    <span className="text-xs text-slate-400">→</span>
                    <span className="text-xs text-danger-600 font-medium">{test.current.toFixed(2)}</span>
                  </div>
                </div>
                <div className="text-right">
                  <span className={`text-sm font-bold ${
                    test.severity === "critical" ? "text-danger-600" :
                    test.severity === "major" ? "text-warning-600" :
                    "text-slate-600"
                  }`}>
                    {test.delta:+.2f}
                  </span>
                  <span className={`block text-xs capitalize ${
                    test.severity === "critical" ? "text-danger-500" :
                    test.severity === "major" ? "text-warning-500" :
                    "text-slate-400"
                  }`}>
                    {test.severity}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
