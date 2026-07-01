/**
 * AREval Dashboard API client.
 *
 * Unified fetch wrapper with base URL, error handling, and typed responses.
 * When the API is unreachable, returns null so pages can fall back to mock data.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8700";

// ---------------------------------------------------------------------------
// Types matching areval-api response shapes
// ---------------------------------------------------------------------------

export interface StatEntry {
  name: string;
  value: string;
  change: string;
  changeType: "positive" | "negative" | "neutral";
}

export interface EvalRun {
  id: string;
  name: string;
  total_cases: number;
  pass_rate: number;
  avg_score: number;
  regression_count: number;
  started_at: string;
}

export interface EvalRunDetail extends EvalRun {
  description: string;
  config: Record<string, unknown>;
  completed_at: string | null;
  passed_cases: number;
  failed_cases: number;
  error_cases: number;
  total_cost_usd: number;
  total_tokens: number;
  duration_seconds: number;
  test_results: TestResultEntry[];
}

export interface TestResultEntry {
  test_case: { id: string; name: string; input: string; expected_output: string | null; tags: string[] };
  agent_output: { output: string; latency_ms: number; token_usage: Record<string, number>; cost_usd: number };
  status: string;
  scores: Record<string, number>;
  overall_score: number;
  threshold: number;
  passed: boolean;
  error_message: string | null;
  judge_reasoning: string | null;
  execution_time_ms: number;
  is_regression: boolean;
  regression_delta: number | null;
}

export interface DatasetEntry {
  id: string;
  name: string;
  description: string;
  size: number;
  tags: string[];
  created_at: string;
}

export interface BaselineEntry {
  id: string;
  name: string;
  run_id: string;
  created_at: string;
  tags: string[];
}

export interface StatsResponse {
  total_evaluations: number;
  total_test_cases: number;
  average_pass_rate: number;
  total_regressions: number;
  datasets: number;
  baselines: number;
}

export interface ComparisonEntry {
  test_id: string;
  test_name: string;
  baseline_score: number;
  current_score: number;
  delta: number;
  regressed: boolean;
}

export interface HealthResponse {
  status: string;
  pass_rate: number;
  avg_score: number;
  sample_count: number;
  active_alerts: number;
}

export interface TrendPoint {
  time: string;
  pass_rate: number;
  avg_score: number;
  sample_count: number;
}

export interface AlertRecord {
  type: string;
  severity: string;
  message: string;
  current_value: number;
  threshold_value: number;
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Core fetch helper
// ---------------------------------------------------------------------------

async function apiGet<T>(path: string): Promise<T | null> {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
    });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Exported API functions
// ---------------------------------------------------------------------------

export async function fetchStats(): Promise<StatsResponse | null> {
  return apiGet<StatsResponse>("/api/v1/stats");
}

export async function fetchEvalRuns(): Promise<EvalRun[] | null> {
  return apiGet<EvalRun[]>("/api/v1/evaluations");
}

export async function fetchEvalRun(runId: string): Promise<EvalRunDetail | null> {
  return apiGet<EvalRunDetail>(`/api/v1/evaluations/${runId}`);
}

export async function fetchDatasets(): Promise<DatasetEntry[] | null> {
  return apiGet<DatasetEntry[]>("/api/v1/datasets");
}

export async function fetchBaselines(): Promise<BaselineEntry[] | null> {
  return apiGet<BaselineEntry[]>("/api/v1/baselines");
}

export async function fetchOnlineHealth(): Promise<HealthResponse | null> {
  return apiGet<HealthResponse>("/api/v1/online/health");
}

export async function fetchOnlineTrend(windowMinutes = 1440): Promise<TrendPoint[] | null> {
  return apiGet<TrendPoint[]>(`/api/v1/online/trend?window_minutes=${windowMinutes}`);
}

export async function fetchOnlineAlerts(limit = 50): Promise<AlertRecord[] | null> {
  return apiGet<AlertRecord[]>(`/api/v1/online/alerts?limit=${limit}`);
}
