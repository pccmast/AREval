import React, { useEffect, useState } from "react";
import { Upload, Database, Tag, FileJson, ChevronRight } from "lucide-react";
import { fetchDatasets, type DatasetEntry } from "@/lib/apiClient";

// Mock fallback
const MOCK_DATASETS = [
  { id: "ds_001", name: "Production Regression Suite", description: "500 test cases covering core agent functionality", size: 500, tags: ["regression", "production", "critical"], lastUpdated: "2024-01-14", version: 3 },
  { id: "ds_002", name: "Tool Call Validation", description: "150 cases for testing tool selection and parameter accuracy", size: 150, tags: ["tools", "accuracy"], lastUpdated: "2024-01-13", version: 2 },
  { id: "ds_003", name: "RAG Quality Assessment", description: "200 RAG-focused cases with context and faithfulness checks", size: 200, tags: ["rag", "faithfulness", "relevance"], lastUpdated: "2024-01-12", version: 1 },
  { id: "ds_004", name: "SWE-bench Subset", description: "Real GitHub issues for code agent evaluation", size: 100, tags: ["swe-bench", "code", "github"], lastUpdated: "2024-01-10", version: 1 },
];

function mapApiDataset(d: DatasetEntry) {
  return {
    id: d.id,
    name: d.name,
    description: d.description || `${d.size} test cases`,
    size: d.size,
    tags: d.tags,
    lastUpdated: d.created_at,
    version: 1,
  };
}

export default function Datasets() {
  const [datasets, setDatasets] = useState(MOCK_DATASETS);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      const ds = await fetchDatasets();
      if (ds && ds.length > 0) {
        setDatasets(ds.map(mapApiDataset));
      }
      setLoading(false);
    }
    load();
  }, []);

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Datasets</h1>
          <p className="text-slate-500 mt-1">
            {loading
              ? "Loading..."
              : datasets[0].id.startsWith("ds_")
              ? "Mock data — start API to load real datasets"
              : `${datasets.length} datasets from API`}
          </p>
        </div>
        <button className="flex items-center gap-2 bg-primary-600 text-white px-4 py-2.5 rounded-lg text-sm font-medium hover:bg-primary-700 transition-colors">
          <Upload className="w-4 h-4" />
          Upload Dataset
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {datasets.map((ds) => (
          <div key={ds.id} className="card hover:shadow-md transition-shadow cursor-pointer">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-primary-50 rounded-lg">
                  <FileJson className="w-5 h-5 text-primary-600" />
                </div>
                <div>
                  <h3 className="font-semibold text-slate-900">{ds.name}</h3>
                  <p className="text-sm text-slate-500 mt-0.5">{ds.description}</p>
                </div>
              </div>
              <ChevronRight className="w-5 h-5 text-slate-400" />
            </div>
            <div className="flex items-center gap-6 mt-4 text-sm">
              <div className="flex items-center gap-1.5 text-slate-600">
                <Database className="w-4 h-4" />
                <span className="font-medium">{ds.size}</span>
                <span className="text-slate-400">cases</span>
              </div>
              <div className="flex items-center gap-1.5 text-slate-600">
                <Tag className="w-4 h-4" />
                <span>v{ds.version}</span>
              </div>
            </div>
            <div className="flex flex-wrap gap-2 mt-3">
              {ds.tags.map((tag) => (
                <span key={tag} className="badge bg-slate-100 text-slate-600">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
