import React, { useEffect, useState } from "react";
import { Check, X, CheckCheck, Clock, AlertTriangle } from "lucide-react";
import { fetchDatasets, type DatasetEntry } from "@/lib/apiClient";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8700";

interface ReviewCase {
  id: string;
  name: string;
  input: string;
  expected_output: string | null;
  tags: string[];
  metadata: Record<string, unknown>;
}

interface ReviewStats {
  total: number;
  approved: number;
  pending_review: number;
}

export default function CurationReview() {
  const [datasets, setDatasets] = useState<DatasetEntry[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [cases, setCases] = useState<ReviewCase[]>([]);
  const [stats, setStats] = useState<ReviewStats | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    async function load() {
      const ds = await fetchDatasets();
      if (ds) {
        const pending = ds.filter((d) => d.tags?.includes("pending_review"));
        setDatasets(pending);
        if (pending.length > 0) setSelectedId(pending[0].id);
      }
    }
    load();
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    async function loadCases() {
      setLoading(true);
      try {
        const res = await fetch(`${API_BASE}/api/v1/datasets/${selectedId}`);
        if (res.ok) {
          const data = await res.json();
          const pending = (data.test_cases || []).filter(
            (c: ReviewCase) => c.tags?.includes("pending_review")
          );
          setCases(pending);
          setStats({
            total: data.test_cases?.length || 0,
            approved: (data.test_cases || []).filter(
              (c: ReviewCase) => c.tags?.includes("approved")
            ).length,
            pending_review: pending.length,
          });
        }
      } finally {
        setLoading(false);
      }
    }
    loadCases();
  }, [selectedId]);

  async function handleApprove(caseId: string) {
    const res = await fetch(
      `${API_BASE}/api/v1/datasets/${selectedId}/cases/${caseId}/approve`,
      { method: "PUT" }
    );
    if (res.ok) {
      const data = await res.json();
      setCases((prev) => prev.filter((c) => c.id !== caseId));
      setStats(data.review_stats);
    }
  }

  async function handleReject(caseId: string) {
    const res = await fetch(
      `${API_BASE}/api/v1/datasets/${selectedId}/cases/${caseId}/reject`,
      { method: "PUT" }
    );
    if (res.ok) {
      const data = await res.json();
      setCases((prev) => prev.filter((c) => c.id !== caseId));
      setStats(data.review_stats);
    }
  }

  async function handleApproveAll() {
    const res = await fetch(
      `${API_BASE}/api/v1/datasets/${selectedId}/approve-all`,
      { method: "POST" }
    );
    if (res.ok) {
      const data = await res.json();
      setCases([]);
      setStats(data.review_stats);
    }
  }

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Curation Review</h1>
          <p className="text-slate-500 mt-1">Approve or reject auto-curated test cases</p>
        </div>
      </div>

      {datasets.length === 0 ? (
        <div className="card text-center py-16">
          <Clock className="w-12 h-12 text-slate-300 mx-auto mb-4" />
          <p className="text-slate-500">No datasets awaiting review.</p>
          <p className="text-slate-400 text-sm mt-1">
            Run <code className="bg-slate-100 px-1 rounded">areval curate</code> to auto-generate test cases.
          </p>
        </div>
      ) : (
        <>
          {/* Dataset selector */}
          <div className="flex gap-2 mb-6">
            {datasets.map((d) => (
              <button
                key={d.id}
                onClick={() => setSelectedId(d.id)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
                  selectedId === d.id
                    ? "bg-primary-500 text-white"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                }`}
              >
                {d.name} ({d.size})
              </button>
            ))}
          </div>

          {/* Stats bar */}
          {stats && (
            <div className="flex gap-4 mb-6 text-sm">
              <span className="bg-yellow-50 text-yellow-700 px-3 py-1 rounded-full flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" />
                {stats.pending_review} pending
              </span>
              <span className="bg-green-50 text-green-700 px-3 py-1 rounded-full">
                {stats.approved} approved
              </span>
              <span className="bg-slate-50 text-slate-500 px-3 py-1 rounded-full">
                {stats.total} total
              </span>
              {cases.length > 0 && (
                <button
                  onClick={handleApproveAll}
                  className="bg-primary-500 text-white px-4 py-1 rounded-full flex items-center gap-1 hover:bg-primary-600 ml-auto"
                >
                  <CheckCheck className="w-3 h-3" />
                  Approve All ({cases.length})
                </button>
              )}
            </div>
          )}

          {/* Case cards */}
          {loading ? (
            <p className="text-slate-400">Loading...</p>
          ) : cases.length === 0 ? (
            <p className="text-slate-400 text-sm">All cases reviewed ✓</p>
          ) : (
            <div className="space-y-4">
              {cases.map((c) => (
                <div key={c.id} className="card border border-slate-200">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1 min-w-0">
                      <p className="text-xs text-slate-400 mb-1">
                        {c.metadata?.curation_category as string || "auto-curated"}
                        {" · "}score {(c.metadata?.value_score as number || 0).toFixed(2)}
                        {" · "}{c.id}
                      </p>
                      <p className="text-slate-900 text-sm font-medium line-clamp-3">
                        {c.input}
                      </p>
                      {c.expected_output && (
                        <p className="text-slate-500 text-xs mt-2 bg-slate-50 p-2 rounded line-clamp-2">
                          <span className="text-slate-400">Reference: </span>
                          {c.expected_output}
                        </p>
                      )}
                    </div>
                    <div className="flex gap-2 ml-4 shrink-0">
                      <button
                        onClick={() => handleApprove(c.id)}
                        className="p-2 rounded-lg bg-green-50 text-green-600 hover:bg-green-100 transition"
                        title="Approve"
                      >
                        <Check className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleReject(c.id)}
                        className="p-2 rounded-lg bg-red-50 text-red-500 hover:bg-red-100 transition"
                        title="Reject"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
