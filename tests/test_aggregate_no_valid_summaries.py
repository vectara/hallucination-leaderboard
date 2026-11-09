"""A model+date group with no valid summaries must not produce a stats row.

compute_hallucination_rate returns the sentinel -1.0 for such a group. Written
as a stat it becomes hallucination_rate=-100.0, which update_readme.py then
ranks first with a 200% factual consistency rate.

Run from the repo root:
    python -m pytest tests/test_aggregate_no_valid_summaries.py -v
"""
import json
from types import SimpleNamespace

import src.pipeline.aggregate as aggregate
from src.data_model import BasicLLMConfig, BasicSummary

COMPANY = "testco"
MODEL = "test-model"


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _run(tmp_path, monkeypatch, groups):
    """Run the real aggregation over {summary_date: [is_valid, ...]}."""
    monkeypatch.setitem(
        aggregate.MODEL_REGISTRY, COMPANY, {"summary_class": BasicSummary}
    )
    model_dir = tmp_path / COMPANY / MODEL
    model_dir.mkdir(parents=True)

    summaries, judgments = [], []
    for summary_date, validity in groups.items():
        for index, is_valid in enumerate(validity):
            uid = f"{summary_date}-{index}"
            summaries.append({
                "article_id": index,
                "summary_uid": uid,
                "summary": "text",
                "company": COMPANY,
                "model_name": MODEL,
                "date_code": "",
                "eval_name": "test-eval",
                "summary_date": summary_date,
            })
            judgments.append({
                "eval_name": "test-eval",
                "judgment_date": "2026-03-01",
                "summary_uid": uid,
                "hhem_version": "2.3",
                "hhem_score": 0.9 if is_valid else -1.0,
                "is_valid": is_valid,
                "word_count": 50 if is_valid else 0,
            })
    _write_jsonl(model_dir / "summaries.jsonl", summaries)
    _write_jsonl(model_dir / "judgments.jsonl", judgments)

    eval_config = SimpleNamespace(
        eval_name="test-eval",
        hhem_version="2.3",
        output_dir=str(tmp_path),
        summary_file="summaries.jsonl",
        judgment_file="judgments.jsonl",
        stats_file="stats.jsonl",
    )
    llm_config = BasicLLMConfig(company=COMPANY, model_name=MODEL)
    aggregate.generate_and_save_results(eval_config, llm_config)
    return model_dir / "stats.jsonl"


def test_group_without_valid_summaries_is_skipped(tmp_path, monkeypatch):
    stats_path = _run(tmp_path, monkeypatch, {
        "2026-01-01": [True, True],
        "2026-02-01": [False, False],
    })

    rows = [json.loads(line) for line in stats_path.read_text().splitlines()]
    assert [row["summary_date"] for row in rows] == ["2026-01-01"]
    assert rows[0]["hallucination_rate"] == 0.0
    assert rows[0]["answer_rate"] == 100.0


def test_only_invalid_summaries_writes_no_stats(tmp_path, monkeypatch):
    stats_path = _run(tmp_path, monkeypatch, {"2026-02-01": [False, False]})

    assert not stats_path.exists()
