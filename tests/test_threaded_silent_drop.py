"""Regression tests for the silent row-drop seen in the 2026-09-22 gpt-5.6-sol live run.

Observed: 7658 of 7731 rows written, progress bar reached 7731/7731, nothing in
the log. Root cause (reproduced before the fix, now guarded here):

  1. In generate_summaries_for_one_llm_multithreaded, the worker's `except`
     handler referenced `m`, which is only bound after `with llm as m:` enters.
     If setup() raised, the handler itself raised NameError, no THREAD ERROR row
     was queued, and the main loop (which never called future.result()) swallowed it.
  2. setup() raised because each article constructs a brand-new OpenAI client
     that was never closed (teardown was a no-op). Under fd pressure (ulimit -n
     1024 on cpu3) the client constructor failed with EMFILE while loading CA certs.

Fixed behaviour asserted below:
  * every article yields exactly one row, failures as SummaryError.THREAD_ERROR;
  * every failure is logged with "Worker failed";
  * OpenAILLM.teardown() closes the client so fds do not accumulate;
  * the real threaded function under a low fd limit loses nothing.

Run from the repo root:
    ../lb_env/bin/python -m pytest tests/test_threaded_silent_drop.py -s -v

Tests marked `network` make a handful of tiny real gpt-4.1-nano calls (< $0.01).
"""
import gc
import json
import logging
import os
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from src.data_model import BasicLLMConfig, EvalConfig, SummaryError
from src.LLMs.openai import OpenAIConfig, OpenAILLM, OpenAISummary
from src.pipeline.summarize import generate_summaries_for_one_llm_multithreaded

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAS_ENV = os.path.exists(os.path.join(REPO, ".env"))
THREAD_ERROR = SummaryError.THREAD_ERROR.value


def _eval_config(name="silent-drop-test"):
    return EvalConfig(
        eval_name=name,
        eval_date="2026-09-22",
        hhem_version="2.3-API",
        pipeline=["summarize"],
        common_LLM_config=BasicLLMConfig(),
        per_LLM_configs=[],
    )


def _articles(n, text="Some article text that is long enough to matter."):
    return pd.DataFrame({"article_id": list(range(1, n + 1)), "text": [text] * n})


def _rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _run_threaded(out, cfg, n, factory=None, workers=4):
    generate_summaries_for_one_llm_multithreaded(
        llm_factory=factory or (lambda ec, lc: OpenAILLM(lc)),
        article_df=_articles(n),
        eval_config=_eval_config(),
        llm_config=cfg,
        summaries_jsonl_path=str(out),
        LLM_SUMMARY_CLASS=OpenAISummary,
        max_workers=workers,
    )


OFFLINE_CFG = dict(model_name="gpt-6-sol", threads=4, temperature=-1.0,
                   min_throttle_time=0.0, api_type="default")


def _offline_summarize(self, prepared_text):
    return "A perfectly ordinary summary with more than five words in it."


# --------------------------------------------------------------------------
# 1. setup() failing before `m` is bound -> THREAD ERROR row + log line, no drop
# --------------------------------------------------------------------------
def test_setup_failure_is_recorded_not_dropped(tmp_path, monkeypatch, caplog):
    n_articles, fail_every = 20, 5
    counter = {"n": 0}

    def flaky_setup(self):
        counter["n"] += 1
        if counter["n"] % fail_every == 0:
            raise OSError(24, "Too many open files")  # what EMFILE looks like
        self.client = object()

    monkeypatch.setattr(OpenAILLM, "setup", flaky_setup)
    monkeypatch.setattr(OpenAILLM, "summarize", _offline_summarize)

    out = tmp_path / "summaries.jsonl"; out.touch()
    caplog.set_level(logging.DEBUG)
    _run_threaded(out, OpenAIConfig(**OFFLINE_CFG), n_articles)

    rows = _rows(out)
    errors = [r for r in rows if r["summary"] == THREAD_ERROR]
    expected_failures = n_articles // fail_every
    print(f"\nwritten={len(rows)}/{n_articles}; THREAD ERROR rows={len(errors)}; "
          f"'Worker failed' log lines={caplog.text.count('Worker failed')}")

    assert len(rows) == n_articles, "rows were dropped"
    assert sorted(r["article_id"] for r in rows) == list(range(1, n_articles + 1))
    assert len(errors) == expected_failures
    assert caplog.text.count("Worker failed") == expected_failures
    assert "Too many open files" in caplog.text, "traceback of the real cause should be logged"
    # provenance on the error row matches the config, so downstream judge/aggregate accept it
    assert errors[0]["model_name"] == "gpt-6-sol" and errors[0]["company"] == "openai"


# --------------------------------------------------------------------------
# 2. llm_factory() itself failing (llm never constructed) -> still recorded
# --------------------------------------------------------------------------
def test_factory_failure_is_recorded_not_dropped(tmp_path, monkeypatch, caplog):
    n_articles = 6
    calls = {"n": 0}

    def flaky_factory(ec, lc):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("factory exploded")
        return OpenAILLM(lc)

    monkeypatch.setattr(OpenAILLM, "setup", lambda self: setattr(self, "client", object()))
    monkeypatch.setattr(OpenAILLM, "summarize", _offline_summarize)

    out = tmp_path / "summaries.jsonl"; out.touch()
    caplog.set_level(logging.DEBUG)
    _run_threaded(out, OpenAIConfig(**OFFLINE_CFG), n_articles, factory=flaky_factory, workers=1)

    rows = _rows(out)
    assert len(rows) == n_articles
    assert sum(r["summary"] == THREAD_ERROR for r in rows) == 1
    assert caplog.text.count("Worker failed") == 1 and "factory exploded" in caplog.text


# --------------------------------------------------------------------------
# 3. Real clients: teardown() now closes the client, so fds do not accumulate
# --------------------------------------------------------------------------
def _open_fds():
    return len(os.listdir("/proc/self/fd"))


@pytest.mark.network
@pytest.mark.skipif(not HAS_ENV, reason="needs .env with OPENAI_API_KEY")
def test_openai_client_does_not_leak_fds():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO, ".env"))
    cfg = OpenAIConfig(model_name="gpt-4.1-nano", temperature=0.0, max_tokens=8,
                       min_throttle_time=0.0, api_type="default",
                       prompt="Reply with the single word OK. {article}")
    n = 25
    gc.disable()  # the collector must NOT be what saves us
    try:
        before = _open_fds()
        for _ in range(n):
            llm = OpenAILLM(cfg)
            with llm as m:
                m.try_to_summarize_one_article("Say OK.")
            del llm, m
        after = _open_fds()
    finally:
        gc.enable()
    print(f"\nopen fds: before={before} after {n} clients (gc disabled)={after}")
    assert after - before <= 2, f"fds leaked: {after - before} extra after {n} clients"


# --------------------------------------------------------------------------
# 4. End to end: real threaded function under a low fd limit loses nothing
# --------------------------------------------------------------------------
_SUBPROC = textwrap.dedent("""
    import gc, json, logging, os, resource, sys
    sys.path.insert(0, {repo!r}); os.chdir({repo!r})
    from dotenv import load_dotenv; load_dotenv(".env")
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    import pandas as pd
    from src.data_model import BasicLLMConfig, EvalConfig
    from src.LLMs.openai import OpenAIConfig, OpenAILLM, OpenAISummary
    from src.pipeline.summarize import generate_summaries_for_one_llm_multithreaded

    resource.setrlimit(resource.RLIMIT_NOFILE, ({limit}, {limit}))
    if {gc_off}: gc.disable()

    cfg = OpenAIConfig(model_name="gpt-4.1-nano", threads=8, temperature=0.0, max_tokens=8,
                       min_throttle_time=0.0, api_type="default",
                       prompt="Reply with the single word OK. {{article}}")
    ec = EvalConfig(eval_name="fd-test", eval_date="2026-09-22", hhem_version="2.3-API",
                    pipeline=["summarize"], common_LLM_config=BasicLLMConfig(), per_LLM_configs=[cfg])
    n = {n}
    df = pd.DataFrame({{"article_id": list(range(1, n + 1)), "text": ["Say OK."] * n}})
    out = {out!r}; open(out, "w").close()
    generate_summaries_for_one_llm_multithreaded(
        llm_factory=lambda e, c: OpenAILLM(c), article_df=df, eval_config=ec, llm_config=cfg,
        summaries_jsonl_path=out, LLM_SUMMARY_CLASS=OpenAISummary, max_workers=8)
    rows = [json.loads(l) for l in open(out) if l.strip()]
    print(json.dumps({{"n": n, "written": len(rows), "unique": len({{r["article_id"] for r in rows}}),
                      "thread_error": sum(r["summary"] == "THREAD ERROR" for r in rows),
                      "model_failed": sum(r["summary"].startswith("MODEL FAILED") for r in rows)}}))
""")


@pytest.mark.network
@pytest.mark.skipif(not HAS_ENV, reason="needs .env with OPENAI_API_KEY")
@pytest.mark.parametrize("gc_off", [False, True], ids=["gc-natural", "gc-lagging"])
def test_low_fd_limit_loses_nothing_end_to_end(tmp_path, gc_off):
    out = str(tmp_path / "summaries.jsonl")
    script = _SUBPROC.format(repo=REPO, limit=48, gc_off=gc_off, n=60, out=out)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
    summary_line = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    assert summary_line, f"subprocess produced no summary line\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-3000:]}"
    res = json.loads(summary_line[-1])
    print(f"\n[{'gc-lagging' if gc_off else 'gc-natural'}] {res} | log: 'Worker failed'="
          f"{proc.stderr.count('Worker failed')}, 'Unhandled exception'={proc.stderr.count('Unhandled exception')}")

    assert res["written"] == res["n"] == res["unique"], "rows missing or duplicated"
    # With the client closed per article, no fd pressure should build at all:
    assert res["thread_error"] == 0 and res["model_failed"] == 0, "failures occurred under fd limit"
    assert "Unhandled exception" not in proc.stderr
