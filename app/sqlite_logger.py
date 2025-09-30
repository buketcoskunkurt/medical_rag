"""
Lightweight SQLite logger for QA requests.
Logs timestamp, query, answers, timings, and references (URLs) to a local DB.

Enable/disable via env LOG_SQLITE_PATH (empty/"0" disables). Default: data/qa_logs.sqlite
"""
from __future__ import annotations
import os
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List


APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
DEFAULT_DB = os.path.join(BASE_DIR, "data", "qa_logs.sqlite")


def _resolve_db_path() -> str | None:
    p = os.environ.get("LOG_SQLITE_PATH", DEFAULT_DB)
    if not p or str(p).strip().lower() in {"0", "none", "disabled"}:
        return None
    return p


def _init_db(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS qa_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_utc TEXT NOT NULL,
                query_en TEXT,
                query_tr TEXT,
                answer_en TEXT,
                answer_tr TEXT,
                k INTEGER,
                retrieval_time_seconds REAL,
                generation_time_seconds REAL,
                total_time_seconds REAL,
                used_snippets_json TEXT,
                used_urls TEXT
            )
            """
        )


def log_qa(response: Dict[str, Any], k: int | None = None) -> None:
    """Insert a QA log row; safe no-op if disabled or on error."""
    path = _resolve_db_path()
    if not path:
        return
    try:
        _init_db(path)
        ts = datetime.now(timezone.utc).isoformat()
        q_en = response.get("query_en") or ""
        q_tr = response.get("query_tr") or ""
        ans_en = ((response.get("english") or {}).get("text") or "")
        ans_tr = ((response.get("turkish") or {}).get("text") or "")
        rts = float(response.get("retrieval_time_seconds", 0.0) or 0.0)
        gts = float(response.get("generation_time_seconds", 0.0) or 0.0)
        tts = float(response.get("total_time_seconds", 0.0) or 0.0)
        used = response.get("used_snippets") or []
        used_urls = ";".join([str(u.get("url") or "") for u in used if u.get("url")])[:1024]
        used_json = json.dumps(used[:10], ensure_ascii=False)
        with sqlite3.connect(path) as con:
            con.execute(
                """
                INSERT INTO qa_logs (
                    ts_utc, query_en, query_tr, answer_en, answer_tr, k,
                    retrieval_time_seconds, generation_time_seconds, total_time_seconds,
                    used_snippets_json, used_urls
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts, q_en, q_tr, ans_en, ans_tr, int(k or 0),
                    rts, gts, tts,
                    used_json, used_urls,
                ),
            )
    except Exception:
        # Never let logging break the API
        return
