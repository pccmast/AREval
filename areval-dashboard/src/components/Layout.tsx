import React from "react";
import Link from "next/link";
import { useRouter } from "next/router";
import {
  Activity,
  Database,
  BarChart3,
  Settings,
  Shield,
  GitCompare,
} from "lucide-react";

const navItems = [
  { href: "/", label: "Overview", icon: Activity },
  { href: "/evaluations", label: "Evaluations", icon: BarChart3 },
  { href: "/datasets", label: "Datasets", icon: Database },
  { href: "/regression", label: "Regression", icon: GitCompare },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Layout({ children }: { children: React.ReactNode }) {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-slate-50 flex">
      {/* Sidebar */}
      <aside className="w-64 bg-white border-r border-slate-200 flex-shrink-0">
        <div className="p-6">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
              <Shield className="w-5 h-5 text-white" />
            </div>
            <span className="text-lg font-bold text-slate-900">AREval</span>
          </div>
          <p className="text-xs text-slate-500 mt-1">Agent Regression Harness</p>
        </div>

        <nav className="px-3 py-4">
          {navItems.map((item) => {
            const isActive = router.pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors mb-1 ${
                  isActive
                    ? "bg-primary-50 text-primary-700"
                    : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                }`}
              >
                <item.icon className="w-4 h-4" />
                {item.label}
              </Link>
            );
          })}
        </nav>

        {/* Status indicator */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-slate-200">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <div className="w-2 h-2 bg-success-500 rounded-full animate-pulse" />
            System Operational
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        {children}
      </main>
    </div>
  );
}
