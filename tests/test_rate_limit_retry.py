"""Rate limits (HTTP 429) are retried with backoff instead of becoming failed rows.

Before the fix, try_to_summarize_one_article caught every exception (429s
included) and returned MODEL_FAILED_TO_RETURN_OUTPUT, so the pipeline's retry
wrapper never saw an exception; the wrapper also used retry_if_exception_type
with a predicate function, which would not have worked either, and the
single-thread path had no retry at all. Seen in the 2026-10-16 gemini-4-argon
run: one 429 became a permanent "MODEL FAILED" row.

Fixed behaviour asserted below (offline, backoff patched to zero):
  * a transient 429 is retried and the article gets a real summary;
  * a 429 that never clears ends as MODEL_FAILED_TO_RETURN_OUTPUT after 10 attempts,
    without crashing the run or dropping the row;
  * other errors still fail immediately (no retry), as before;
  * both the threaded and the single-thread paths behave the same.

Run from the repo root:
    ../lb_env/bin/python -m pytest tests/test_rate_limit_retry.py -v
"""
import json

import pandas as pd
import pytest
from tenacity import wait_none

from src.data_model import SummaryError
from src.LLMs.openai import OpenAIConfig, OpenAILLM, OpenAISummary
from src.pipeline import summarize as summarize_mod

from test_threaded_silent_drop import OFFLINE_CFG, _articles, _eval_config, _rows

MODEL_FAILED = SummaryError.MODEL_FAILED_TO_RETURN_OUTPUT.value
GOOD = "A perfectly ordinary summary with more than five words in it."


class RateLimited(Exception):
    status_code = 429


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(summarize_mod, "_summarize_retrying_rate_limits",
                        summarize_mod._summarize_retrying_rate_limits.retry_with(wait=wait_none()))


def _patch_llm(monkeypatch, fail_first, exc=RateLimited("Error code: 429 - rate limited")):
    """Each article raises `exc` on its first `fail_first` calls, then succeeds."""
    calls = {}

    def summarize(self, prepared_text):
        calls[prepared_text] = calls.get(prepared_text, 0) + 1
        if calls[prepared_text] <= fail_first:
            raise exc
        return GOOD

    monkeypatch.setattr(OpenAILLM, "setup", lambda self: setattr(self, "client", object()))
    monkeypatch.setattr(OpenAILLM, "teardown", lambda self: None)
    monkeypatch.setattr(OpenAILLM, "summarize", summarize)
    return calls


def _run(tmp_path, threaded, n=5):
    out = tmp_path / "summaries.jsonl"
    out.touch()
    cfg = OpenAIConfig(**OFFLINE_CFG)
    if threaded:
        summarize_mod.generate_summaries_for_one_llm_multithreaded(
            llm_factory=lambda ec, lc: OpenAILLM(lc), article_df=_articles_distinct(n),
            eval_config=_eval_config(), llm_config=cfg, summaries_jsonl_path=str(out),
            LLM_SUMMARY_CLASS=OpenAISummary, max_workers=3)
    else:
        summarize_mod.generate_summaries_for_one_llm(
            llm=OpenAILLM(cfg), article_df=_articles_distinct(n), eval_name="retry-test",
            eval_date="2026-11-17", summaries_jsonl_path=str(out), LLM_SUMMARY_CLASS=OpenAISummary)
    return _rows(out)


def _articles_distinct(n):
    df = _articles(n)
    df["text"] = ["Article number %d with enough words to summarise." % i for i in df["article_id"]]
    return df


@pytest.mark.parametrize("threaded", [True, False])
def test_transient_429_is_retried(tmp_path, monkeypatch, threaded):
    calls = _patch_llm(monkeypatch, fail_first=2)
    rows = _run(tmp_path, threaded)
    assert len(rows) == 5
    assert all(r["summary"] == GOOD for r in rows), [r["summary"] for r in rows]
    assert all(v == 3 for v in calls.values())  # 2 rate-limited attempts + 1 success


@pytest.mark.parametrize("threaded", [True, False])
def test_persistent_429_becomes_model_failed_without_crash(tmp_path, monkeypatch, threaded):
    calls = _patch_llm(monkeypatch, fail_first=10**6)
    rows = _run(tmp_path, threaded)
    assert len(rows) == 5
    assert all(r["summary"] == MODEL_FAILED for r in rows)
    assert all(v == 10 for v in calls.values())  # stop_after_attempt(10)


@pytest.mark.parametrize("threaded", [True, False])
def test_other_errors_are_not_retried(tmp_path, monkeypatch, threaded):
    calls = _patch_llm(monkeypatch, fail_first=1, exc=ValueError("Error code: 400 - bad request"))
    rows = _run(tmp_path, threaded)
    assert len(rows) == 5
    assert all(r["summary"] == MODEL_FAILED for r in rows)
    assert all(v == 1 for v in calls.values())
