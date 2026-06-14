import React, { useState } from "react";
import { Play, Filter, Download, ChevronDown } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";

const statusColors: Record<string, string> = {
  passed: "#22c55e",
  failed: "#ef4444",
  error: "#f59e0b",
  skipped: "#94a3b8",
};

const evaluationRuns = [
  {
    id: "ev_001",
    name: "Nightly Full Suite",
    dataset: "production-v2",
    metrics: ["exact_match", "semantic_similarity", "faithfulness"],
    judges: ["llm_judge"],
    total: 500,
    passed: 445,
    failed: 42,
    error: 8,
    skipped: 5,
    avg_score: 0.89,
    duration: "12m 34s",
    cost: "$8.42",
    started: "2024-01-15 02:00:00",
  },
  {
    id: "ev_002",
    name: "PR #342 - Prompt Update",
    dataset: "regression-suite",
    metrics: ["exact_match", "contains"],
    judges: [],
    total: 200,
    passed: 156,
    failed: 38,
    error: 4,
    skipped: 2,
    avg_score: 0.78,
    duration: "5m 12s",
    cost: "$3.20",
    started: "2024-01-15 14:23:00",
  },
  {
    id: "ev_003",
    name: "Tool Accuracy Check",
    dataset: "tool-tests",
    metrics: ["tool_call_accuracy", "task_completion"],
    judges: ["agent_judge"],
    total: 150,
    passed: 128,
    failed: 18,
    error: 3,
    skipped: 1,
    avg_score: 0.85,
    duration: "8m 45s",
    cost: "$5.60",
    started: "2024-01-15 10:15:00",
  },
];

const scoreDistribution = [
  { range: "0.0-0.2", count: 12 },
  { range: "0.2-0.4", count: 28 },
  { range: "0.4-0.6", count: 45 },
  { range: "0.6-0.8", count: 89 },
  { range: "0.8-1.0", count: 156 },
];

export default function Evaluations() {
  const [selectedRun, setSelectedRun] = useState(evaluationRuns[0]);

  const pieData = [
    { name: "Passed", value: selectedRun.passed },
    { name: "Failed", value: selectedRun.failed },
    { name: "Error", value: selectedRun.error },
    { name: "Skipped", value: selectedRun.skipped },
  ];

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Evaluations</h1>
          <p className="text-slate-500 mt-1">Run and analyze agent evaluation suites</p>
        </div>
        <button className="flex items-center gap-2 bg-primary-600 text-white px-4 py-2.5 rounded-lg text-sm font-medium hover:bg-primary-700 transition-colors">
          <Play className="w-4 h-4" />
          New Evaluation
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Run List */}
        <div className="card lg:col-span-1">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-slate-900">Recent Runs</h3>
            <button className="text-slate-400 hover:text-slate-600">
              <Filter className="w-4 h-4" />
            </button>
          </div>
          <div className="space-y-2">
            {evaluationRuns.map((run) => (
              <button
                key={run.id}
                onClick={() => setSelectedRun(run)}
                className={`w-full text-left p-3 rounded-lg transition-colors ${
                  selectedRun.id === run.id
                    ? "bg-primary-50 border border-primary-200"
                    : "hover:bg-slate-50 border border-transparent"
                }`}
              >
                <p className="text-sm font-medium text-slate-900">{run.name}</p>
                <p className="text-xs text-slate-500 mt-0.5">{run.dataset}</p>
                <div className="flex items-center gap-3 mt-2">
                  <span className="text-xs text-success-600 font-medium">
                    {(run.passed / run.total * 100).toFixed(0)}% pass
                  </span>
                  <span className="text-xs text-slate-400">{run.duration}</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Run Details */}
        <div className="card lg:col-span-2">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-lg font-semibold text-slate-900">{selectedRun.name}</h3>
              <p className="text-sm text-slate-500">{selectedRun.started}</p>
            </div>
            <button className="flex items-center gap-1 text-sm text-primary-600 hover:text-primary-700">
              <Download className="w-4 h-4" />
              Export
            </button>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-4 gap-4 mb-6">
            <div className="bg-slate-50 rounded-lg p-4">
              <p className="text-xs text-slate-500">Total Cases</p>
              <p className="text-xl font-bold text-slate-900">{selectedRun.total}</p>
            </div>
            <div className="bg-success-50 rounded-lg p-4">
              <p className="text-xs text-success-600">Passed</p>
              <p className="text-xl font-bold text-success-700">{selectedRun.passed}</p>
            </div>
            <div className="bg-danger-50 rounded-lg p-4">
              <p className="text-xs text-danger-600">Failed</p>
              <p className="text-xl font-bold text-danger-700">{selectedRun.failed}</p>
            </div>
            <div className="bg-primary-50 rounded-lg p-4">
              <p className="text-xs text-primary-600">Avg Score</p>
              <p className="text-xl font-bold text-primary-700">{selectedRun.avg_score}</p>
            </div>
          </div>

          {/* Charts */}
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-medium text-slate-700 mb-3">Result Distribution</h4>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={statusColors[entry.name.toLowerCase()] || "#94a3b8"} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div>
              <h4 className="text-sm font-medium text-slate-700 mb-3">Score Distribution</h4>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={scoreDistribution}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="range" fontSize={10} stroke="#64748b" />
                  <YAxis fontSize={10} stroke="#64748b" />
                  <Tooltip />
                  <Bar dataKey="count" fill="#0ea5e9" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
