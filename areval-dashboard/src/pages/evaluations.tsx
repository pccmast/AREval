import React, { useEffect, useState } from "react";
import { Play, Filter, Download } from "lucide-react";
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
import { fetchEvalRuns, fetchEvalRun, type EvalRun, type EvalRunDetail } from "@/lib/apiClient";

const statusColors: Record<string, string> = {
  passed: "#22c55e",
  failed: "#ef4444",
  error: "#f59e0b",
  skipped: "#94a3b8",
};

// Mock fallback
const MOCK_RUNS = [
  { id: "ev_001", name: "Nightly Full Suite", dataset: "production-v2", metrics: ["exact_match", "semantic_similarity"], judges: ["llm_judge"], total: 500, passed: 445, failed: 42, error: 8, skipped: 5, avg_score: 0.89, duration: "12m 34s", cost: "$8.42", started: "2024-01-15 02:00:00" },
  { id: "ev_002", name: "PR #342 - Prompt Update", dataset: "regression-suite", metrics: ["exact_match", "contains"], judges: [], total: 200, passed: 156, failed: 38, error: 4, skipped: 2, avg_score: 0.78, duration: "5m 12s", cost: "$3.20", started: "2024-01-15 14:23:00" },
  { id: "ev_003", name: "Tool Accuracy Check", dataset: "tool-tests", metrics: ["tool_call_accuracy", "task_completion"], judges: ["agent_judge"], total: 150, passed: 128, failed: 18, error: 3, skipped: 1, avg_score: 0.85, duration: "8m 45s", cost: "$5.60", started: "2024-01-15 10:15:00" },
];

const MOCK_SCORE_DIST = [
  { range: "0.0-0.2", count: 12 },
  { range: "0.2-0.4", count: 28 },
  { range: "0.4-0.6", count: 45 },
  { range: "0.6-0.8", count: 89 },
  { range: "0.8-1.0", count: 156 },
];

interface LocalRun {
  id: string;
  name: string;
  dataset: string;
  metrics: string[];
  judges: string[];
  total: number;
  passed: number;
  failed: number;
  error: number;
  skipped: number;
  avg_score: number;
  duration: string;
  cost: string;
  started: string;
}

function mapApiRun(r: EvalRun): LocalRun {
  return {
    id: r.id,
    name: r.name,
    dataset: "—",
    metrics: [],
    judges: [],
    total: r.total_cases,
    passed: Math.round(r.total_cases * r.pass_rate),
    failed: r.total_cases - Math.round(r.total_cases * r.pass_rate),
    error: 0,
    skipped: 0,
    avg_score: r.avg_score,
    duration: "—",
    cost: `$${(r.total_cases * 0.02).toFixed(2)}`,
    started: r.started_at,
  };
}

export default function Evaluations() {
  const [allRuns, setAllRuns] = useState<LocalRun[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [selectedDetail, setSelectedDetail] = useState<EvalRunDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      const runs = await fetchEvalRuns();
      if (runs && runs.length > 0) {
        const mapped = runs.map(mapApiRun);
        setAllRuns(mapped);
        setSelectedId(mapped[0].id);
        // Load detail for first run
        const detail = await fetchEvalRun(mapped[0].id);
        setSelectedDetail(detail);
      } else {
        setAllRuns(MOCK_RUNS);
        setSelectedId(MOCK_RUNS[0].id);
      }
      setLoading(false);
    }
    load();
  }, []);

  const selected = allRuns.find((r) => r.id === selectedId) || allRuns[0];

  const pieData = [
    { name: "Passed", value: selected?.passed ?? 0 },
    { name: "Failed", value: selected?.failed ?? 0 },
    { name: "Error", value: selected?.error ?? 0 },
    { name: "Skipped", value: selected?.skipped ?? 0 },
  ];

  // Score distribution from real detail or mock
  const scoreDistribution = selectedDetail?.test_results?.length
    ? (() => {
        const scores = selectedDetail.test_results.map((t) => t.overall_score);
        const bins = [0.2, 0.4, 0.6, 0.8, 1.0];
        const labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"];
        return labels.map((range, i) => ({
          range,
          count: scores.filter((s) => s <= bins[i] && (i === 0 || s > bins[i - 1])).length,
        }));
      })()
    : MOCK_SCORE_DIST;

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Evaluations</h1>
          <p className="text-slate-500 mt-1">
            {loading ? "Loading..." : allRuns.length > 0 && allRuns[0].id !== "ev_001"
              ? `${allRuns.length} runs from API`
              : "Mock data — start API to see real runs"}
          </p>
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
          <div className="space-y-2 max-h-[500px] overflow-y-auto">
            {allRuns.map((run) => (
              <button
                key={run.id}
                onClick={async () => {
                  setSelectedId(run.id);
                  const d = await fetchEvalRun(run.id);
                  setSelectedDetail(d);
                }}
                className={`w-full text-left p-3 rounded-lg transition-colors ${
                  selectedId === run.id
                    ? "bg-primary-50 border border-primary-200"
                    : "hover:bg-slate-50 border border-transparent"
                }`}
              >
                <p className="text-sm font-medium text-slate-900 truncate">{run.name}</p>
                <p className="text-xs text-slate-500 mt-0.5">{run.dataset}</p>
                <div className="flex items-center gap-3 mt-2">
                  <span className="text-xs text-success-600 font-medium">
                    {run.total > 0 ? (run.passed / run.total * 100).toFixed(0) : 0}% pass
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
              <h3 className="text-lg font-semibold text-slate-900">{selected?.name}</h3>
              <p className="text-sm text-slate-500">{selected?.started}</p>
            </div>
            <button className="flex items-center gap-1 text-sm text-primary-600 hover:text-primary-700">
              <Download className="w-4 h-4" />
              Export
            </button>
          </div>

          {/* Detail metrics (from API or mock) */}
          {selectedDetail ? (
            <div className="grid grid-cols-4 gap-4 mb-6">
              <div className="bg-slate-50 rounded-lg p-4">
                <p className="text-xs text-slate-500">Total</p>
                <p className="text-xl font-bold text-slate-900">{selectedDetail.total_cases}</p>
              </div>
              <div className="bg-success-50 rounded-lg p-4">
                <p className="text-xs text-success-600">Passed</p>
                <p className="text-xl font-bold text-success-700">{selectedDetail.passed_cases}</p>
              </div>
              <div className="bg-danger-50 rounded-lg p-4">
                <p className="text-xs text-danger-600">Failed</p>
                <p className="text-xl font-bold text-danger-700">{selectedDetail.failed_cases}</p>
              </div>
              <div className="bg-primary-50 rounded-lg p-4">
                <p className="text-xs text-primary-600">Avg Score</p>
                <p className="text-xl font-bold text-primary-700">{selectedDetail.avg_score.toFixed(3)}</p>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-4 gap-4 mb-6">
              <div className="bg-slate-50 rounded-lg p-4"><p className="text-xs text-slate-500">Total</p><p className="text-xl font-bold">{selected?.total}</p></div>
              <div className="bg-success-50 rounded-lg p-4"><p className="text-xs text-success-600">Passed</p><p className="text-xl font-bold">{selected?.passed}</p></div>
              <div className="bg-danger-50 rounded-lg p-4"><p className="text-xs text-danger-600">Failed</p><p className="text-xl font-bold">{selected?.failed}</p></div>
              <div className="bg-primary-50 rounded-lg p-4"><p className="text-xs text-primary-600">Avg Score</p><p className="text-xl font-bold">{selected?.avg_score}</p></div>
            </div>
          )}

          {/* Charts */}
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h4 className="text-sm font-medium text-slate-700 mb-3">Result Distribution</h4>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={pieData} cx="50%" cy="50%" innerRadius={50} outerRadius={80} dataKey="value">
                    {pieData.map((_, i) => (
                      <Cell key={i} fill={statusColors[pieData[i].name.toLowerCase()] || "#94a3b8"} />
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
