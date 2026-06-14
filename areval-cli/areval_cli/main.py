"""AREval CLI - Command line interface for agent evaluation.

Usage:
    areval run --config eval.yaml --dataset tests.jsonl
    areval dashboard
    areval baseline create --run-id <id>
    areval compare --current <file> --baseline <id>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from areval.evaluator import Evaluator
from areval.metrics import get_metric
from areval.judges import get_judge
from areval.datasets import DatasetManager
from areval.regression.baseline import BaselineManager
from areval.test_case import TestCase, AgentOutput, EvaluationRun
from areval.utils.serialization import reconstruct_run

app = typer.Typer(
    name="areval",
    help="Agent Regression Evaluation Harness",
    no_args_is_help=True,
)
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_results_json(path: str) -> Dict[str, Any]:
    """Load a results JSON file previously saved by JSONReporter."""
    with open(path) as f:
        return json.load(f)


def _reconstruct_run(data: Dict[str, Any]) -> EvaluationRun:
    """Reconstruct an EvaluationRun from its serialized dict.

    Delegates to the shared :func:`areval.utils.serialization.reconstruct_run`.
    """
    return reconstruct_run(data)


def _parse_yaml_config(config_path: str) -> Dict[str, Any]:
    """Parse an eval_config.yaml and return a normalized config dict."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return raw.get("evaluation", raw)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command()
def run(
    config: str = typer.Option(
        "eval.yaml", "--config", "-c", help="Evaluation config file (YAML)"
    ),
    dataset: str = typer.Option(
        None, "--dataset", "-d", help="Dataset file (JSONL) — overrides config"
    ),
    output: str = typer.Option(
        "results.json", "--output", "-o", help="Output results file"
    ),
    threshold: float = typer.Option(
        None, "--threshold", "-t", help="Pass threshold — overrides config"
    ),
) -> None:
    """Run a full evaluation pipeline.

    Reads eval_config.yaml (or --config) to determine datasets, metrics,
    judges, and regression settings.  Outputs results.json and prints
    a summary table.  Exits with code 1 if pass_rate < threshold.
    """
    console.print(Panel.fit("[bold blue]AREval[/bold blue] — Agent Regression Evaluation Harness"))

    # ---- Resolve config ----
    cfg: Dict[str, Any] = {}
    if Path(config).exists():
        cfg = _parse_yaml_config(config)
        console.print(f"[dim]Config loaded from {config}[/dim]")

    # ---- Dataset ----
    ds_path = dataset or (cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), str) else cfg.get("dataset", {}).get("path", ""))  # type: ignore[union-attr]
    ds_format = (
        cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
    ).get("format", "jsonl")
    if not ds_path:
        console.print(
            "[red]Error: No dataset specified (--dataset or config.evaluation.dataset.path)[/red]"
        )
        raise typer.Exit(code=2)

    dm = DatasetManager()
    ds = dm.create_from_file(ds_path, name=Path(ds_path).stem, format=ds_format)
    console.print(f"[dim]Dataset: {ds.name} ({ds.size} cases)[/dim]")

    # ---- Threshold ----
    thresh = threshold if threshold is not None else cfg.get("threshold", 0.7)

    # ---- Build evaluator ----
    evaluator = Evaluator(threshold=thresh)

    # Metrics (from config or fallback CLI defaults)
    metric_entries: list[dict[str, Any]] = cfg.get("metrics", [])
    if not metric_entries:
        # Fallback: exact_match + semantic_similarity
        metric_entries = [{"name": "exact_match"}, {"name": "semantic_similarity"}]

    for entry in metric_entries:
        name = entry["name"]
        m_threshold = entry.get("threshold", thresh)
        m_config = entry.get("config", {})
        try:
            metric = get_metric(name, threshold=m_threshold, **m_config)
            evaluator.add_metric(metric)
            console.print(f"  [green]+[/green] metric: {name}")
        except KeyError:
            console.print(f"  [yellow]?[/yellow] unknown metric '{name}', skipped")

    # Judges (optional, from config)
    judge_entries: list[dict[str, Any]] = cfg.get("judges", [])
    for entry in judge_entries:
        name = entry["name"]
        j_threshold = entry.get("threshold", thresh)
        j_config = entry.get("config", {})
        try:
            judge = get_judge(name, threshold=j_threshold, **j_config)
            evaluator.add_judge(judge)
            console.print(f"  [green]+[/green] judge: {name}")
        except KeyError:
            console.print(f"  [yellow]?[/yellow] unknown judge '{name}', skipped")

    # ---- Regression config ----
    reg_cfg = cfg.get("regression", {})
    compare_baseline = reg_cfg.get("compare_baseline", True)

    console.print()

    # ---- Agent function ----
    # When no real agent is wired, use an echo agent (repeat input)
    def _echo_agent(tc: TestCase) -> AgentOutput:
        return AgentOutput(
            output=tc.input[:100],
            latency_ms=50.0,
            token_usage={"input": len(tc.input.split()), "output": 10},
        )

    # ---- Run evaluation ----
    eval_run = evaluator.evaluate(
        test_cases=ds.test_cases,
        agent_fn=_echo_agent,
        run_name=cfg.get("name", "cli-evaluation"),
        run_description=cfg.get("description", ""),
        config=cfg,
        compare_baseline=compare_baseline,
    )

    # ---- Display summary ----
    console.print(evaluator.summary(eval_run))

    # ---- Export ----
    from areval_sdk.reporters import JSONReporter

    JSONReporter(eval_run).export(output)
    console.print(f"\n[green]Results saved to {output}[/green]")

    # ---- Auto-baseline ----
    if reg_cfg.get("auto_baseline", False):
        bid = evaluator.create_baseline(eval_run, name=cfg.get("name", "auto-baseline"))
        console.print(f"[green]Auto-baseline created: {bid}[/green]")

    # ---- Exit code ----
    if eval_run.pass_rate < thresh:
        console.print(f"\n[red]FAILED: pass_rate {eval_run.pass_rate:.1%} < threshold {thresh:.1%}[/red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------


@app.command()
def dashboard(
    port: int = typer.Option(3000, "--port", "-p", help="Dashboard HTTP port"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Dashboard host"),
) -> None:
    """Launch the evaluation dashboard (Next.js dev server).

    Requires Node.js and npm to be available.  The dashboard source lives in
    the ``areval-dashboard/`` directory alongside the project root.

    If the dashboard directory or npm is not found, prints clear setup
    instructions instead of failing silently.
    """
    dash_dir = Path(__file__).resolve().parents[2] / "areval-dashboard"

    if not dash_dir.is_dir():
        console.print("[yellow]Dashboard directory not found.[/yellow]")
        console.print(f"Expected: {dash_dir}")
        console.print("Dashboard is a Next.js app.  See areval-dashboard/README.md for setup.")
        return

    pkg_json = dash_dir / "package.json"
    if not pkg_json.exists():
        console.print("[yellow]areval-dashboard/package.json not found.[/yellow]")
        console.print("Run:  cd areval-dashboard && npm install && npm run dev")
        return

    console.print(f"[bold blue]Starting dashboard on http://{host}:{port}[/bold blue]")

    try:
        subprocess.run(
            ["npm", "run", "dev", "--", "--port", str(port), "--hostname", host],
            cwd=str(dash_dir),
            check=False,
        )
    except FileNotFoundError:
        console.print("[yellow]npm not found.  Please install Node.js and npm.[/yellow]")
        console.print(f"Then run:  cd {dash_dir} && npm install && npm run dev")


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------


@app.command()
def baseline(
    action: str = typer.Argument(..., help="create, list, or delete"),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Run ID (for create)"
    ),
    results_file: Optional[str] = typer.Option(
        None, "--results", "-r", help="Results JSON file (for create, alternative to --run-id)"
    ),
    baseline_id: Optional[str] = typer.Option(
        None, "--baseline-id", help="Baseline ID (for delete)"
    ),
    name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Baseline display name (for create)"
    ),
    tag: Optional[List[str]] = typer.Option(
        None, "--tag", help="Tags for the baseline (for create)"
    ),
) -> None:
    """Manage evaluation baselines.

    Create a baseline from an evaluation run or saved results file,
    list all baselines, or delete one by ID.
    """
    bm = BaselineManager()

    if action == "list":
        baselines = bm.list_baselines()
        if not baselines:
            console.print("[dim]No baselines found.[/dim]")
            return

        table = Table(title="Baselines", expand=False)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="green")
        table.add_column("Run ID", style="dim")
        table.add_column("Created", style="yellow")
        table.add_column("Tags", style="magenta")

        for b in baselines:
            table.add_row(
                b.id,
                b.name,
                b.run_id or "—",
                b.created_at.strftime("%Y-%m-%d %H:%M"),
                ", ".join(b.tags) if b.tags else "—",
            )
        console.print(table)
        return

    if action == "create":
        if results_file:
            if not Path(results_file).exists():
                console.print(f"[red]Results file not found: {results_file}[/red]")
                raise typer.Exit(code=1)
            data = _load_results_json(results_file)
            run = _reconstruct_run(data)
            baseline_obj = bm.create_baseline(
                run=run,
                name=name or f"Baseline from {Path(results_file).stem}",
                tags=list(tag) if tag else ["cli"],
            )
            console.print(f"[green]Baseline created: {baseline_obj.id}[/green]")
            console.print(f"  Name: {baseline_obj.name}")
            console.print(f"  Cases: {len(run.test_results)}")
            console.print(f"  Avg score: {run.avg_score:.3f}")
            return

        if run_id:
            # Try to find a saved run JSON with this ID
            results_dir = Path(".areval/results")
            if results_dir.exists():
                for f in results_dir.glob("*.json"):
                    try:
                        data = _load_results_json(str(f))
                        if data.get("id") == run_id:
                            run = _reconstruct_run(data)
                            baseline_obj = bm.create_baseline(
                                run=run,
                                name=name or f"Baseline {run_id[:8]}",
                                tags=list(tag) if tag else ["cli"],
                            )
                            console.print(f"[green]Baseline created from run {run_id}: {baseline_obj.id}[/green]")
                            return
                    except (json.JSONDecodeError, KeyError):
                        continue

            console.print(f"[red]Run {run_id} not found.  Use --results to load from a JSON file.[/red]")
            raise typer.Exit(code=1)

        console.print(
            "[red]Provide --run-id or --results to create a baseline.[/red]"
        )
        raise typer.Exit(code=2)

    if action == "delete" and baseline_id:
        if bm.delete_baseline(baseline_id):
            console.print(f"[green]Deleted baseline {baseline_id}[/green]")
        else:
            console.print(f"[red]Baseline {baseline_id} not found[/red]")
            raise typer.Exit(code=1)
        return

    if action == "delete":
        console.print("[red]Use --baseline-id to specify which baseline to delete.[/red]")
        raise typer.Exit(code=2)

    console.print(f"[red]Unknown action: {action}[/red]")
    console.print("Available actions: create, list, delete")
    raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@app.command()
def compare(
    current: str = typer.Option(
        ..., "--current", "-c", help="Current results JSON file"
    ),
    baseline_id: str = typer.Option(
        ..., "--baseline", "-b", help="Baseline ID to compare against"
    ),
) -> None:
    """Compare current evaluation results against a saved baseline.

    Loads results from a JSON file (saved by ``areval run`` or the API)
    and compares each test case against the specified baseline.

    Displays a Rich table with per-case deltas and highlights regressions.
    """
    # Load current results
    if not Path(current).exists():
        console.print(f"[red]Results file not found: {current}[/red]")
        raise typer.Exit(code=1)

    current_data = _load_results_json(current)
    current_run = _reconstruct_run(current_data)

    # Load baseline
    bm = BaselineManager()
    comp = bm.compare_to_baseline(current_run.test_results, baseline_id)

    if "error" in comp:
        console.print(f"[red]{comp['error']}[/red]")
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"[bold]Current:[/bold] {current_run.name}  |  [bold]Baseline:[/bold] {comp['baseline_name']}",
            title="Comparison",
        )
    )

    comparisons: list[dict[str, Any]] = comp["comparisons"]
    if not comparisons:
        console.print("[dim]No common test cases found between current and baseline.[/dim]")
        return

    # --- Per-case table ---
    table = Table(title="Test Case Deltas", expand=False)
    table.add_column("Test", style="cyan")
    table.add_column("Baseline", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Status", justify="center")

    regressed = 0
    for c in comparisons:
        delta = c["delta"]
        is_reg = c["regressed"]
        if is_reg:
            regressed += 1

        delta_style = f"[red]{delta:+.3f}[/red]" if is_reg else f"[dim]{delta:+.3f}[/dim]"
        status = "[red]FAIL[/red]" if is_reg else "[green]PASS[/green]"

        table.add_row(
            c.get("test_name", c.get("test_id", "?")),
            f"{c['baseline_score']:.3f}",
            f"{c['current_score']:.3f}",
            delta_style,
            status,
        )

    console.print(table)

    # --- Summary ---
    console.print()
    summary_text = Text()
    summary_text.append(f"Average delta: {comp['avg_delta']:+.4f}  ", style="bold")
    if regressed:
        summary_text.append(
            f"Regressed cases: {regressed}/{len(comparisons)}", style="red"
        )
    else:
        summary_text.append("No regressions detected", style="green")
    console.print(summary_text)

    if regressed:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@app.command()
def config(
    api_key: Optional[str] = typer.Option(None, "--api-key", help="OpenAI API key"),
    anthropic_key: Optional[str] = typer.Option(None, "--anthropic-key", help="Anthropic API key"),
) -> None:
    """Configure AREval settings (API keys, etc.)."""
    config_dir = Path.home() / ".areval"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "config.yaml"

    # Load existing config
    existing: Dict[str, str] = {}
    if config_file.exists():
        with open(config_file) as f:
            existing = yaml.safe_load(f) or {}

    if api_key:
        existing["openai_api_key"] = api_key
    if anthropic_key:
        existing["anthropic_api_key"] = anthropic_key

    if api_key or anthropic_key:
        with open(config_file, "w") as f:
            yaml.dump(existing, f)
        console.print("[green]Configuration saved to ~/.areval/config.yaml[/green]")
    else:
        console.print("[yellow]No changes. Use --api-key or --anthropic-key.[/yellow]")


# ---------------------------------------------------------------------------
# curate
# ---------------------------------------------------------------------------


@app.command()
def curate(
    traces_file: str = typer.Option(
        ".areval/traces.json", "--traces", help="Trace file path (JSON array of spans)"
    ),
    output_name: str = typer.Option(
        "auto-curated", "--name", "-n", help="Dataset name"
    ),
    max_cases: int = typer.Option(100, "--max", help="Maximum test cases to curate"),
    min_score: float = typer.Option(0.3, "--min-score", help="Minimum value score"),
) -> None:
    """Curate a test dataset from production trace data.

    Loads a JSON trace file, analyses each trace for evaluation value,
    de-duplicates similar inputs, strips PII, and saves the resulting
    test cases as a new dataset.
    """
    import json
    from areval.datasets.curator import TraceCurator, CurationConfig
    from areval.tracing.tracer import TraceSpan

    console.print("[bold blue]AREval[/bold blue] — Trace Curation\n")

    # Load trace data from JSON file
    if not Path(traces_file).exists():
        console.print(f"[red]Trace file not found: {traces_file}[/red]")
        raise typer.Exit(code=1)

    with open(traces_file) as f:
        raw = json.load(f)

    # Parse into TraceSpan objects
    # Accept two formats: {"trace_id": [span_dicts]} or [{span_dict}]
    traces: Dict[str, List[TraceSpan]] = {}
    if isinstance(raw, dict):
        for tid, spans_list in raw.items():
            traces[tid] = []
            for sd in spans_list:
                span = TraceSpan(name=sd.get("name", ""), span_id=sd.get("span_id", ""))
                span.trace_id = sd.get("trace_id")
                span.start_time = sd.get("start_time", 0)
                span.end_time = sd.get("end_time", 0)
                span.status = sd.get("status", "ok")
                span.attributes = sd.get("attributes", {})
                traces[tid].append(span)
    elif isinstance(raw, list):
        # Flat list of spans — group by trace_id
        for sd in raw:
            span = TraceSpan(name=sd.get("name", ""), span_id=sd.get("span_id", ""))
            span.trace_id = sd.get("trace_id", "unknown")
            span.start_time = sd.get("start_time", 0)
            span.end_time = sd.get("end_time", 0)
            span.status = sd.get("status", "ok")
            span.attributes = sd.get("attributes", {})
            traces.setdefault(span.trace_id or "unknown", []).append(span)

    console.print(f"Loaded {sum(len(v) for v in traces.values())} spans across {len(traces)} traces")

    # Configure & curate
    config = CurationConfig(min_value_score=min_score, max_cases=max_cases)
    curator = TraceCurator(config=config)
    dataset = curator.curate_from_traces(traces)

    dm = DatasetManager()
    dm.save_dataset(dataset)

    console.print(f"\n[green]Dataset '{output_name}' curated: {dataset.size} test cases[/green]")
    for tc in dataset.test_cases[:10]:
        cat = tc.metadata.get("curation_category", "?")
        score = tc.metadata.get("value_score", 0)
        console.print(f"  [{cat}] (score={score:.2f}) {tc.input[:60]}...")
    if dataset.size > 10:
        console.print(f"  ... and {dataset.size - 10} more")


if __name__ == "__main__":
    app()
