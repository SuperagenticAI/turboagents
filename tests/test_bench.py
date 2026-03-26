from turboagents.bench.kv import build_kv_report
from turboagents.bench.paper import build_paper_report
from turboagents.bench.rag import build_rag_report


def test_kv_report_contains_multi_bit_metrics() -> None:
    report = build_kv_report()
    assert report.payload["dataset"] == "tiny-kv"
    assert report.payload["vectors"] == 64
    assert "b3.5_mse" in report.payload
    assert "b3.5_mean_payload_bytes" in report.payload


def test_rag_report_contains_recall_metrics() -> None:
    report = build_rag_report()
    assert report.payload["queries"] == 16
    assert "b3.5_recall_at_10" in report.payload


def test_paper_report_contains_mse_and_cosine() -> None:
    report = build_paper_report()
    assert "mse_b3.5" in report.payload
    assert "cosine_b3.5" in report.payload
