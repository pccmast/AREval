"""Tests for trace analyser and curator (Phase 2)."""

from areval.tracing.analyzers import TraceAnalyzer, TraceAnalysis
from areval.tracing.tracer import TraceSpan
from areval.datasets.curator import TraceCurator, CurationConfig
from areval.datasets.manager import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_span(name: str, status: str = "ok", **attrs: object) -> TraceSpan:
    s = TraceSpan(name=name)
    s.status = status
    s.start_time = 0.0
    s.end_time = 100.0 if status == "ok" else 200.0
    for k, v in attrs.items():
        s.attributes[k] = v
    return s


# ---------------------------------------------------------------------------
# TraceAnalyzer
# ---------------------------------------------------------------------------


class TestTraceAnalyzer:
    def test_analyzer_low_score_trace(self) -> None:
        a = TraceAnalyzer(score_threshold=0.7)
        spans = [_make_span("eval", score=0.3, **{"input.args": "hello", "output": "bad"})]
        result = a.analyze_trace("t1", spans)
        assert result.value_score > 0.3
        assert result.category in ("low_score", "error", "edge_case")

    def test_analyzer_error_trace(self) -> None:
        a = TraceAnalyzer()
        spans = [
            _make_span("agent_call", status="error", **{"input.args": "do stuff"}),
            _make_span("tool_call", status="error"),
        ]
        result = a.analyze_trace("t2", spans)
        assert result.error_count == 2
        assert result.value_score > 0.2

    def test_analyzer_normal_trace(self) -> None:
        a = TraceAnalyzer()
        spans = [_make_span("agent_call", score=0.95, **{"input.args": "hi", "output": "ok"})]
        result = a.analyze_trace("t3", spans)
        # Normal + high score should have low curation value
        assert result.value_score < 0.5
        assert result.category in ("normal", "edge_case")

    def test_extract_input_output(self) -> None:
        a = TraceAnalyzer()
        spans = [
            _make_span("root", **{"input.args": "What is Python?", "output": "A language"})
        ]
        inp, out = a._extract_input_output(spans)
        assert inp == "What is Python?"
        assert out == "A language"

    def test_extract_tool_calls(self) -> None:
        a = TraceAnalyzer()
        spans = [
            _make_span("tool_search", **{"tool.name": "search", "query": "test"}),
            _make_span("tool_calc", **{"tool.name": "calculator"}),
        ]
        tools = a._extract_tool_calls(spans)
        assert len(tools) == 2

    def test_analyze_all_sorts_by_value(self) -> None:
        a = TraceAnalyzer()
        traces = {
            "a": [_make_span("ok", score=0.9)],
            "b": [_make_span("bad", score=0.2)],
        }
        results = a.analyze_all(traces)
        assert results[0].value_score >= results[-1].value_score


# ---------------------------------------------------------------------------
# TraceCurator
# ---------------------------------------------------------------------------


class TestTraceCurator:
    def test_curator_basic(self) -> None:
        # Use raw spans (not tracer.start_span which auto-finishes with ok)
        traces = {
            "t1": [_make_span("agent", status="error", score=0.1, **{"input.args": "What is 2+2?"})],
            "t2": [_make_span("agent", status="error", score=0.0, **{"input.args": "Capital of France?"})],
        }
        curator = TraceCurator(config=CurationConfig(
            min_value_score=0.1, max_ratio=1.0, require_review=False,
        ))
        ds = curator.curate_from_traces(traces)
        assert ds.size >= 1
        assert "auto-curated" in ds.tags
        for tc in ds.test_cases:
            assert tc.name.startswith("curated-")

    def test_curator_dynamic_limit(self) -> None:
        """max_ratio + min_keep enforce a dynamic ceiling."""
        traces = {
            f"t{i}": [_make_span("agent", status="error", score=0.05, **{"input.args": f"unique query {i}"})]
            for i in range(100)
        }
        curator = TraceCurator(config=CurationConfig(
            min_value_score=0.0,
            max_ratio=0.1,       # 100 × 0.1 = 10
            min_keep=5,
            absolute_max=8,      # max(5, min(8, 10)) = 8
            require_review=False,
        ))
        ds = curator.curate_from_traces(traces)
        assert ds.size <= 8

    def test_dedup_similar_inputs(self) -> None:
        traces = {
            f"t{i}": [_make_span("dup", status="error", score=0.05, **{"input.args": "What is the weather today?"})]
            for i in range(3)
        }
        curator = TraceCurator(config=CurationConfig(min_value_score=0.0, dedup_similarity=0.8))
        ds = curator.curate_from_traces(traces)
        # Very similar inputs → dedup should leave ~1
        assert ds.size <= 2

    def test_pii_stripping(self) -> None:
        curator = TraceCurator()
        text = "Email me at alice@example.com or call 13800138000"
        cleaned = curator._strip_pii(text)
        assert "alice@example.com" not in cleaned
        assert "13800138000" not in cleaned
        assert "[EMAIL]" in cleaned
        assert "[PHONE]" in cleaned

    def test_analysis_to_test_case(self) -> None:
        curator = TraceCurator()
        analysis = TraceAnalysis(
            trace_id="abc12345def",
            input_text="test input",
            value_score=0.8,
            category="error",
        )
        tc = curator._analysis_to_test_case(analysis)
        assert tc.name == "curated-abc12345"
        assert tc.input == "test input"
        assert "auto-curated" in tc.tags
        assert tc.metadata["source_trace_id"] == "abc12345def"


# ---------------------------------------------------------------------------
# DatasetManager.create_from_traces
# ---------------------------------------------------------------------------


class TestManagerCreateFromTraces:
    def test_create_from_traces_persists(self, tmp_path: object) -> None:
        import tempfile, shutil
        from areval.datasets.manager import DatasetManager
        from areval.tracing.tracer import TraceSpan as TS
        from areval.tracing.tracer import EvalTracer

        td = tempfile.mkdtemp()
        try:
            storage = __import__("pathlib").Path(td) / ".areval" / "datasets"
            dm = DatasetManager(storage_path=storage)

            tracer = EvalTracer()
            # Manually create spans with error status and low score
            for i in range(5):
                span = TS(name=f"call_{i}")
                span.status = "error"
                span.start_time = 0.0
                span.end_time = 100.0
                span.attributes = {"input.args": f"query_{i}", "score": 0.1}
                tracer._spans.append(span)

            ds = dm.create_from_traces(tracer, name="test-trace-ds", description="curation test")
            assert ds.name == "test-trace-ds"
            assert ds.size >= 1

            # Reload
            dm2 = DatasetManager(storage_path=storage)
            reloaded = dm2.get_dataset(ds.id)
            assert reloaded is not None
            assert reloaded.size == ds.size
        finally:
            shutil.rmtree(td, ignore_errors=True)
