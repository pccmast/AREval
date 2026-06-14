"""AREval CLI - Command line interface for agent evaluation.

Usage:
    areval run --config eval.yaml --dataset tests.jsonl
    areval dashboard
    areval baseline create --run-id <id>
    areval compare --current <id> --baseline <id>
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from areval.evaluator import Evaluator
from areval.metrics import (
    ExactMatchMetric,
    ContainsMetric,
    SemanticSimilarityMetric,
    FaithfulnessMetric,
    AnswerRelevanceMetric,
    ToolCallAccuracyMetric,
)
from areval.judges import LLMJudge
from areval.datasets import DatasetManager
from areval.regression.baseline import BaselineManager

app = typer.Typer(
    name="areval",
    help="Agent Regression Evaluation Harness",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    config: str = typer.Option("eval.yaml", "--config", "-c", help="Evaluation config file"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset file (JSONL)"),
    output: str = typer.Option("results.json", "--output", "-o", help="Output file"),
    metrics: list[str] = typer.Option(
        ["exact_match", "semantic_similarity"],
        "--metric",
        "-m",
        help="Metrics to use",
    ),
    threshold: float = typer.Option(0.7, "--threshold", "-t", help="Pass threshold"),
    compare_baseline: bool = typer.Option(True, "--compare/--no-compare", help="Compare to baseline"),
) -> None:
    """Run evaluation on a dataset."""
    console.print("[bold blue]AREval[/bold blue] - Running Evaluation\n")

    # Load dataset
    dm = DatasetManager()
    ds = dm.create_from_file(dataset, name="evaluation-run")

    # Build evaluator
    evaluator = Evaluator(threshold=threshold)

    metric_map = {
        "exact_match": ExactMatchMetric(),
        "contains": ContainsMetric(),
        "semantic_similarity": SemanticSimilarityMetric(),
        "faithfulness": FaithfulnessMetric(),
        "answer_relevance": AnswerRelevanceMetric(),
        "tool_call_accuracy": ToolCallAccuracyMetric(),
    }

    for m in metrics:
        if m in metric_map:
            evaluator.add_metric(metric_map[m])
        else:
            console.print(f"[yellow]Warning: Unknown metric '{m}'[/yellow]")

    # Run evaluation
    eval_run = evaluator.evaluate(
        test_cases=ds.test_cases,
        agent_fn=None,  # Would be provided in real usage
        run_name="cli-evaluation",
        compare_baseline=compare_baseline,
    )

    # Display results
    console.print(evaluator.summary(eval_run))

    # Save results
    from areval_sdk.reporters import JSONReporter
    reporter = JSONReporter(eval_run)
    reporter.export(output)
    console.print(f"\n[green]Results saved to {output}[/green]")

    # Exit with error code if failed
    if eval_run.pass_rate < threshold:
        raise typer.Exit(code=1)


@app.command()
def dashboard(
    port: int = typer.Option(3000, "--port", "-p", help="Dashboard port"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Dashboard host"),
) -> None:
    """Launch the evaluation dashboard."""
    console.print(f"[bold blue]Starting dashboard on http://{host}:{port}[/bold blue]")
    console.print("[yellow]Dashboard is served from areval-dashboard/[/yellow]")
    # In production: launch Next.js dev server or serve built files


@app.command()
def baseline(
    action: str = typer.Argument(..., help="create, list, or delete"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Run ID to baseline"),
    baseline_id: Optional[str] = typer.Option(None, "--baseline-id", help="Baseline ID"),
) -> None:
    """Manage evaluation baselines."""
    bm = BaselineManager()

    if action == "list":
        baselines = bm.list_baselines()
        table = Table(title="Baselines")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Created", style="yellow")
        table.add_column("Tags", style="magenta")

        for b in baselines:
            table.add_row(
                b.id,
                b.name,
                b.created_at.strftime("%Y-%m-%d %H:%M"),
                ", ".join(b.tags),
            )
        console.print(table)

    elif action == "delete" and baseline_id:
        if bm.delete_baseline(baseline_id):
            console.print(f"[green]Deleted baseline {baseline_id}[/green]")
        else:
            console.print(f"[red]Baseline {baseline_id} not found[/red]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")


@app.command()
def compare(
    current: str = typer.Option(..., "--current", "-c", help="Current run ID"),
    baseline_id: str = typer.Option(..., "--baseline", "-b", help="Baseline ID"),
) -> None:
    """Compare current results against a baseline."""
    bm = BaselineManager()
    # Load and compare
    console.print(f"[blue]Comparing {current} against {baseline_id}...[/blue]")


@app.command()
def config(
    api_key: Optional[str] = typer.Option(None, "--api-key", help="OpenAI API key"),
) -> None:
    """Configure AREval settings."""
    config_dir = Path.home() / ".areval"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "config.yaml"

    if api_key:
        import yaml
        with open(config_file, "w") as f:
            yaml.dump({"openai_api_key": api_key}, f)
        console.print("[green]Configuration saved[/green]")
    else:
        console.print("[yellow]No changes made. Use --api-key to set API key.[/yellow]")


if __name__ == "__main__":
    app()
