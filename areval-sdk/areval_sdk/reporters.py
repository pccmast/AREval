"""Result reporters for CI/CD integration.

Formats evaluation results for GitHub Actions, GitLab CI,
and other CI/CD platforms.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from areval.test_case import EvaluationRun


class CIReporter:
    """Generate CI-friendly evaluation reports.

    Produces GitHub Actions workflow commands, JUnit XML,
    and markdown summaries.
    """

    def __init__(self, run: EvaluationRun):
        self.run = run

    def github_summary(self) -> str:
        """Generate GitHub Actions step summary markdown."""
        status = "✅ PASSED" if self.run.pass_rate >= 0.7 else "❌ FAILED"

        lines = [
            "## AREval Results",
            "",
            f"**Status:** {status}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Cases | {self.run.total_cases} |",
            f"| Passed | {self.run.passed_cases} ({self.run.pass_rate:.1%}) |",
            f"| Failed | {self.run.failed_cases} |",
            f"| Skipped | {self.run.skipped_cases} |",
            f"| Timed Out | {self.run.timed_out_cases} |",
            f"| Errors | {self.run.error_cases} |",
            f"| Avg Score | {self.run.avg_score:.3f} |",
            f"| Regressions | {self.run.regression_count} |",
            f"| Cost | ${self.run.total_cost_usd:.4f} |",
            f"| Duration | {self.run.duration_seconds:.1f}s |",
            "",
        ]

        # Regression details
        if self.run.regression_count > 0:
            lines.extend([
                "### ⚠ Regressions Detected",
                "",
            ])
            for result in self.run.test_results:
                if result.is_regression:
                    lines.append(
                        f"- **{result.test_case.name}**: "
                        f"{result.baseline_score:.3f} → {result.overall_score:.3f} "
                        f"(Δ{result.regression_delta:+.3f})"
                    )
            lines.append("")

        # Failed tests
        failed = [r for r in self.run.test_results if not r.passed]
        if failed:
            lines.extend([
                "### ❌ Failed Tests",
                "",
            ])
            for result in failed[:10]:  # Limit to first 10
                lines.append(
                    f"- **{result.test_case.name}**: "
                    f"score={result.overall_score:.3f} "
                    f"(threshold={result.threshold})"
                )
            if len(failed) > 10:
                lines.append(f"- ... and {len(failed) - 10} more")
            lines.append("")

        return "\n".join(lines)

    def github_commands(self) -> List[str]:
        """Generate GitHub Actions workflow commands.

        Uses the GITHUB_OUTPUT environment file (the modern replacement
        for the deprecated `::set-output` command).
        """
        commands = []

        # Write output variables to GITHUB_OUTPUT
        output_file = os.environ.get("GITHUB_OUTPUT")
        if output_file:
            try:
                with open(output_file, "a") as f:
                    f.write(f"pass_rate={self.run.pass_rate:.4f}\n")
                    f.write(f"regression_count={self.run.regression_count}\n")
            except OSError:
                pass
        else:
            # Fallback for non-GitHub environments
            commands.append(f"pass_rate={self.run.pass_rate:.4f}")
            commands.append(f"regression_count={self.run.regression_count}")

        # Annotations for failed tests
        for result in self.run.test_results:
            if not result.passed:
                msg = f"{result.test_case.name}: score={result.overall_score:.3f}"
                commands.append(
                    f"::error title=Test Failed::{msg}"
                )

        # Group annotations for regressions
        if self.run.regression_count > 0:
            commands.append("::warning::Regressions detected in evaluation run")

        return commands

    def write_github_summary(self) -> None:
        """Write summary to GitHub Actions step summary file."""
        summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_path:
            with open(summary_path, "a") as f:
                f.write(self.github_summary())


class JSONReporter:
    """Export evaluation results to JSON format."""

    def __init__(self, run: EvaluationRun):
        self.run = run

    def export(self, path: str) -> None:
        """Export run results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.run.to_dict(), f, indent=2, default=str)

    def export_summary(self, path: str) -> None:
        """Export summary statistics to JSON."""
        summary = {
            "run_id": self.run.id,
            "name": self.run.name,
            "pass_rate": self.run.pass_rate,
            "avg_score": self.run.avg_score,
            "total_cases": self.run.total_cases,
            "passed_cases": self.run.passed_cases,
            "failed_cases": self.run.failed_cases,
            "skipped_cases": self.run.skipped_cases,
            "timed_out_cases": self.run.timed_out_cases,
            "error_cases": self.run.error_cases,
            "regression_count": self.run.regression_count,
            "total_cost_usd": self.run.total_cost_usd,
            "duration_seconds": self.run.duration_seconds,
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
