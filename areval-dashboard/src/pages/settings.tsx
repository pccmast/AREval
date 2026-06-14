import React from "react";
import { Key, Database, Bell, Shield } from "lucide-react";

export default function Settings() {
  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-slate-900">Settings</h1>
        <p className="text-slate-500 mt-1">Configure AREval evaluation environment</p>
      </div>

      <div className="max-w-2xl space-y-6">
        {/* API Keys */}
        <div className="card">
          <div className="flex items-center gap-3 mb-4">
            <Key className="w-5 h-5 text-primary-600" />
            <h3 className="text-lg font-semibold text-slate-900">API Keys</h3>
          </div>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">OpenAI API Key</label>
              <input
                type="password"
                placeholder="sk-..."
                className="w-full px-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Anthropic API Key</label>
              <input
                type="password"
                placeholder="sk-ant-..."
                className="w-full px-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>
          </div>
        </div>

        {/* Evaluation Defaults */}
        <div className="card">
          <div className="flex items-center gap-3 mb-4">
            <Shield className="w-5 h-5 text-primary-600" />
            <h3 className="text-lg font-semibold text-slate-900">Evaluation Defaults</h3>
          </div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-700">Default Pass Threshold</p>
                <p className="text-xs text-slate-500">Minimum score to pass evaluation</p>
              </div>
              <input
                type="number"
                defaultValue={0.7}
                min={0}
                max={1}
                step={0.05}
                className="w-20 px-3 py-2 border border-slate-200 rounded-lg text-sm text-right"
              />
            </div>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-700">Timeout (seconds)</p>
                <p className="text-xs text-slate-500">Max time per test case</p>
              </div>
              <input
                type="number"
                defaultValue={120}
                min={10}
                max={600}
                className="w-20 px-3 py-2 border border-slate-200 rounded-lg text-sm text-right"
              />
            </div>
          </div>
        </div>

        {/* Notifications */}
        <div className="card">
          <div className="flex items-center gap-3 mb-4">
            <Bell className="w-5 h-5 text-primary-600" />
            <h3 className="text-lg font-semibold text-slate-900">Notifications</h3>
          </div>
          <div className="space-y-3">
            {[
              "Notify on regression detection",
              "Notify on evaluation completion",
              "Send daily summary",
            ].map((label) => (
              <label key={label} className="flex items-center gap-3 cursor-pointer">
                <input type="checkbox" className="w-4 h-4 text-primary-600 rounded" defaultChecked />
                <span className="text-sm text-slate-700">{label}</span>
              </label>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
