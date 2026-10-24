"""
OpenPlag HTTP API.   Run:  uvicorn api.main:app --port 8000

POST   /v1/jobs               submit text or a file  -> 202 {job_id}
GET    /v1/jobs/{id}          status / progress / result
GET    /v1/jobs/{id}/report.html   rendered report
DELETE /v1/jobs/{id}          delete a job and its report immediately
GET    /v1/config             limits, defaults, available sources
GET    /health

Design notes
- Jobs run on a small thread pool and are persisted in SQLite (no Redis needed for one server).
- Uploaded files are written to a temp path, parsed, then deleted. Only the report is kept,
  and it is purged after OPENPLAG_RETENTION_DAYS (default 7).
- Auth: set OPENPLAG_API_KEY and clients must send it in the X-API-Key header.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import sqlite3
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Path as PathParam, Request, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ValidationError

import openplag as op

log = logging.getLogger("openplag.api")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


DATA_DIR = Path(os.environ.get("OPENPLAG_DATA_DIR", "./data"))
API_KEY = os.environ.get("OPENPLAG_API_KEY", "")
SEARXNG_URL = os.environ.get("OPENPLAG_SEARXNG", "")
CONTACT = os.environ.get("OPENPLAG_CONTACT_EMAIL", "")
USE_NEURAL = os.environ.get("OPENPLAG_NEURAL", "0") == "1"
MAX_UPLOAD_MB = _env_int("OPENPLAG_MAX_UPLOAD_MB", 15)
MAX_TEXT_CHARS = _env_int("OPENPLAG_MAX_TEXT_CHARS", 300_000)
RATE_PER_HOUR = _env_int("OPENPLAG_RATE_LIMIT_PER_HOUR", 30)
MAX_PENDING = _env_int("OPENPLAG_MAX_PENDING", 25)
WORKERS = _env_int("OPENPLAG_WORKERS", 2)
RETENTION_DAYS = _env_int("OPENPLAG_RETENTION_DAYS", 7)
ALLOWED_EXT = {".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm"}
SELECTABLE_BACKENDS = ["wikipedia", "openalex", "europepmc", "crossref", "semanticscholar", "core",
                       "marginalia", "arxiv"]
DEFAULT_SOURCES = ["wikipedia", "openalex", "europepmc", "crossref"]


# ----------------------------------------------------------------------------- models
class Options(BaseModel):
    exclude_quotes: bool = True
    exclude_references: bool = True
    exclude_citations: bool = True
    min_match_words: int = Field(8, ge=5, le=40)         # ignore matches shorter than this
    detect_paraphrase: bool = True
    check_ai: bool = True
    check_web: bool = True
    backends: list[str] = Field(default_factory=lambda: list(DEFAULT_SOURCES))
    exclude_domains: list[str] = Field(default_factory=list, max_length=50)


# ----------------------------------------------------------------------------- job store
class JobStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.lock = threading.Lock()
        self.db.execute("""CREATE TABLE IF NOT EXISTS jobs(
            id TEXT PRIMARY KEY, status TEXT, stage TEXT, progress REAL, created REAL, updated REAL,
            options TEXT, result TEXT, error TEXT)""")

    def create(self, job_id: str, options: Options):
        now = time.time()
        with self.lock:
            self.db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)",
                            (job_id, "queued", "Waiting in queue", 0.0, now, now, options.model_dump_json(), None, None))

    def update(self, job_id: str, **f):
        f["updated"] = time.time()
        cols = ", ".join(f"{k}=?" for k in f)
        with self.lock:
            self.db.execute(f"UPDATE jobs SET {cols} WHERE id=?", [*f.values(), job_id])

    def get(self, job_id: str) -> Optional[dict]:
        with self.lock:
            r = self.db.execute("SELECT id,status,stage,progress,created,updated,result,error FROM jobs WHERE id=?",
                                (job_id,)).fetchone()
        if not r:
            return None
        return {"id": r[0], "status": r[1], "stage": r[2], "progress": r[3], "created": r[4], "updated": r[5],
                "result": json.loads(r[6]) if r[6] else None, "error": r[7]}

    def delete(self, job_id: str) -> bool:
        with self.lock:
            return self.db.execute("DELETE FROM jobs WHERE id=?", (job_id,)).rowcount > 0

    def pending(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]

    def purge(self, older_than_days: int) -> int:
        with self.lock:
            return self.db.execute("DELETE FROM jobs WHERE created < ?", (time.time() - older_than_days * 86400,)).rowcount

    def fail_interrupted(self):
        with self.lock:
            self.db.execute("UPDATE jobs SET status='failed', error='Server restarted while the job was running. "
                            "Please submit again.' WHERE status IN ('queued','running')")


class RateLimiter:
    def __init__(self, per_hour: int):
        self.per_hour, self.hits, self.lock = per_hour, defaultdict(deque), threading.Lock()

    def allow(self, key: str) -> bool:
        if self.per_hour <= 0:
            return True
        now = time.time()
        with self.lock:
            q = self.hits[key]
            while q and q[0] < now - 3600:
                q.popleft()
            if len(q) >= self.per_hour:
                return False
            q.append(now)
            return True


# ----------------------------------------------------------------------------- shared services
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "uploads").mkdir(exist_ok=True)
BASE_CFG = op.Config(cache_dir=DATA_DIR / "cache", contact_email=CONTACT)
HTTP = op.Http(BASE_CFG)                       # shared so per-host politeness holds across jobs
CACHE = op.DiskCache(BASE_CFG.cache_dir, BASE_CFG.cache_ttl_days, BASE_CFG.cache_max_mb)
STORE = op.FingerprintStore(str(DATA_DIR / "fingerprints.db"), BASE_CFG) if os.environ.get("OPENPLAG_STORE") == "1" else None
JOBS = JobStore(DATA_DIR / "jobs.db")
LIMITER = RateLimiter(RATE_PER_HOUR)
EXECUTOR = ThreadPoolExecutor(max_workers=max(1, WORKERS), thread_name_prefix="openplag")
_neural = {"obj": None, "tried": False}
_neural_lock = threading.Lock()


def get_neural():
    if not USE_NEURAL:
        return None
    with _neural_lock:
        if not _neural["tried"]:
            _neural["tried"] = True
            try:
                _neural["obj"] = op.NeuralAIDetector()
            except Exception as e:                       # torch/transformers missing, offline, etc.
                log.warning("Neural AI detector unavailable (%s); using stylometry only", e)
        return _neural["obj"]


def make_detector(opts: Options) -> op.PlagiarismDetector:
    cfg = op.Config(cache_dir=BASE_CFG.cache_dir, contact_email=CONTACT,
                    exclude_quotes=opts.exclude_quotes, exclude_references=opts.exclude_references,
                    exclude_citations=opts.exclude_citations, min_span_tokens=opts.min_match_words,
                    detect_paraphrase=opts.detect_paraphrase, exclude_domains=tuple(opts.exclude_domains))
    det = op.PlagiarismDetector(cfg, backends=[], neural_ai=get_neural(), store=STORE)
    det.http, det.cache = HTTP, CACHE
    names = [b for b in opts.backends if b in SELECTABLE_BACKENDS]
    det.backends = op.build_backends(HTTP, cfg, names, SEARXNG_URL, public_searx=False, cache=CACHE)
    return det


def run_job(job_id: str, text: Optional[str], upload: Optional[Path], opts: Options):
    try:
        JOBS.update(job_id, status="running", stage="Reading document", progress=0.02)
        if upload is not None:
            text = op.extract_text(str(upload), ocr=True)
        if not text or not text.strip():
            raise ValueError("No readable text found. If this is a scanned PDF, OCR may be unavailable on the server.")
        det = make_detector(opts)
        rep = det.analyze(text, web=opts.check_web, check_ai=opts.check_ai,
                          progress=lambda stage, frac: JOBS.update(job_id, stage=stage, progress=round(frac, 3)))
        JOBS.update(job_id, status="done", stage="Done", progress=1.0, result=json.dumps(asdict(rep)))
    except Exception as e:
        log.exception("job %s failed", job_id)
        msg = str(e) if isinstance(e, (ValueError, RuntimeError)) else "Analysis failed unexpectedly."
        JOBS.update(job_id, status="failed", stage="Failed", error=msg[:500])
    finally:
        if upload is not None:
            upload.unlink(missing_ok=True)               # never keep the original document


def _purge_loop():
    while True:
        try:
            n = JOBS.purge(RETENTION_DAYS)
            if n:
                log.info("purged %d expired jobs", n)
        except Exception:
            log.exception("purge failed")
        time.sleep(3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if API_KEY in ("", "change-me"):
        log.warning("OPENPLAG_API_KEY is empty or still 'change-me' - set a real secret before exposing this server.")
    JOBS.fail_interrupted()
    for f in (DATA_DIR / "uploads").glob("*"):          # leftovers from a crash
        f.unlink(missing_ok=True)
    threading.Thread(target=_purge_loop, daemon=True).start()
    yield


app = FastAPI(title="OpenPlag API", version="0.2.0", lifespan=lifespan)


def auth(x_api_key: Optional[str] = Header(None)):
    if API_KEY and not secrets.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(401, "Invalid or missing API key")


JOB_ID = PathParam(..., pattern=r"^[0-9a-f]{32}$")


# ----------------------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    return {"status": "ok", "searxng": bool(SEARXNG_URL), "neural_ai": USE_NEURAL, "auth": bool(API_KEY)}


@app.get("/v1/config", dependencies=[Depends(auth)])
def config():
    return {"max_upload_mb": MAX_UPLOAD_MB, "max_text_chars": MAX_TEXT_CHARS, "allowed_extensions": sorted(ALLOWED_EXT),
            "sources": SELECTABLE_BACKENDS, "default_sources": DEFAULT_SOURCES, "retention_days": RETENTION_DAYS,
            "defaults": Options().model_dump()}


@app.post("/v1/jobs", status_code=202, dependencies=[Depends(auth)])
async def create_job(request: Request, text: Optional[str] = Form(None), options: Optional[str] = Form(None),
                     file: Optional[UploadFile] = File(None)):
    try:
        opts = Options(**(json.loads(options) if options else {}))
    except (ValueError, ValidationError) as e:
        raise HTTPException(422, f"Invalid options: {e}")
    has_text, has_file = bool(text and text.strip()), file is not None and bool(file.filename)
    if has_text == has_file:
        raise HTTPException(400, "Send either pasted text or one file (not both, not neither).")
    if has_text and len(text) > MAX_TEXT_CHARS:
        raise HTTPException(413, f"Text too long (max {MAX_TEXT_CHARS:,} characters).")
    if not LIMITER.allow(request.client.host if request.client else "?"):
        raise HTTPException(429, "Too many submissions this hour. Please try again later.")
    if JOBS.pending() >= MAX_PENDING:
        raise HTTPException(503, "The server is busy. Please try again in a few minutes.")

    job_id, upload = uuid.uuid4().hex, None
    if has_file:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(415, f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXT))}")
        upload = DATA_DIR / "uploads" / f"{job_id}{ext}"
        size, limit = 0, MAX_UPLOAD_MB * 1_000_000
        try:
            with open(upload, "wb") as out:
                while chunk := await file.read(1 << 20):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_MB} MB).")
                    out.write(chunk)
        except HTTPException:
            upload.unlink(missing_ok=True)
            raise
    JOBS.create(job_id, opts)
    EXECUTOR.submit(run_job, job_id, text if has_text else None, upload, opts)
    return {"job_id": job_id}


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(auth)])
def get_job(job_id: str = JOB_ID):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found (it may have expired or been deleted).")
    return job


@app.get("/v1/jobs/{job_id}/report.html", response_class=HTMLResponse, dependencies=[Depends(auth)])
def get_report_html(job_id: str = JOB_ID):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job["status"] != "done":
        raise HTTPException(409, f"Job is {job['status']}.")
    html = op.render_html(op.report_from_dict(job["result"]))
    return HTMLResponse(html, headers={
        "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'",
        "X-Content-Type-Options": "nosniff"})


@app.delete("/v1/jobs/{job_id}", dependencies=[Depends(auth)])
def delete_job(job_id: str = JOB_ID):
    if not JOBS.delete(job_id):
        raise HTTPException(404, "Job not found.")
    return {"deleted": True}
