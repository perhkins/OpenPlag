"""API tests (no network). Needs: pip install -r requirements-dev.txt"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
_TMP = tempfile.mkdtemp()
os.environ.update({"OPENPLAG_DATA_DIR": _TMP, "OPENPLAG_API_KEY": "test-key", "OPENPLAG_RATE_LIMIT_PER_HOUR": "1000"})
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

import openplag as op  # noqa: E402
from api import main as api  # noqa: E402

SRC = ("Photosynthesis is the process by which green plants and some other organisms use sunlight to "
       "synthesize nutrients from carbon dioxide and water. Photosynthesis in plants generally involves "
       "the green pigment chlorophyll and generates oxygen as a byproduct.")
H = {"X-API-Key": "test-key"}


@pytest.fixture(autouse=True)
def offline_detector(monkeypatch):
    def make(opts):
        det = op.PlagiarismDetector(op.Config(cache_dir=Path(_TMP) / "c", min_span_tokens=opts.min_match_words), backends=[])
        det.local_sources.append(op.Source("https://example.org/bio", "Bio page", "web", SRC))
        return det
    monkeypatch.setattr(api, "make_detector", make)


@pytest.fixture
def client():
    with TestClient(api.app) as c:
        yield c


def wait(client, job_id, timeout=15):
    t = time.time()
    while time.time() - t < timeout:
        j = client.get(f"/v1/jobs/{job_id}", headers=H).json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def submit(client, **kw):
    return client.post("/v1/jobs", headers=H, **kw)


def test_health_is_public(client):
    assert client.get("/health").json()["status"] == "ok"


def test_auth_required(client):
    assert client.post("/v1/jobs", data={"text": "x"}).status_code == 401
    assert client.get("/v1/config", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/v1/config", headers=H).status_code == 200


def test_text_job_end_to_end(client):
    r = submit(client, data={"text": "My own opening about gardens. " + SRC + " And a closing thought of mine.",
                             "options": json.dumps({"check_ai": False, "check_web": False})})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    j = wait(client, jid)
    assert j["status"] == "done" and j["result"]["similarity_percent"] > 40
    assert j["result"]["sources"][0]["url"] == "https://example.org/bio"
    html = client.get(f"/v1/jobs/{jid}/report.html", headers=H)
    assert html.status_code == 200 and 'class="m ' in html.text
    assert "default-src 'none'" in html.headers["content-security-policy"]
    assert client.delete(f"/v1/jobs/{jid}", headers=H).json() == {"deleted": True}
    assert client.get(f"/v1/jobs/{jid}", headers=H).status_code == 404


def test_file_upload_and_cleanup(client):
    r = submit(client, files={"file": ("essay.txt", SRC.encode(), "text/plain")},
               data={"options": json.dumps({"check_ai": False})})
    assert r.status_code == 202
    j = wait(client, r.json()["job_id"])
    assert j["status"] == "done" and j["result"]["similarity_percent"] > 80
    assert list((Path(_TMP) / "uploads").glob("*")) == []                # original never kept


def test_empty_file_fails_gracefully(client):
    j = wait(client, submit(client, files={"file": ("e.txt", b"   ", "text/plain")}).json()["job_id"])
    assert j["status"] == "failed" and "No readable text" in j["error"]


@pytest.mark.parametrize("kw,code", [
    ({"data": {}}, 400),                                                                   # neither
    ({"data": {"text": "hi"}, "files": {"file": ("a.txt", b"x", "text/plain")}}, 400),    # both
    ({"files": {"file": ("evil.exe", b"MZ", "application/octet-stream")}}, 415),
    ({"data": {"text": "hello", "options": "{not json"}}, 422),
    ({"data": {"text": "hello", "options": json.dumps({"min_match_words": 2})}}, 422),
])
def test_validation_errors(client, kw, code):
    assert submit(client, **kw).status_code == code


def test_size_limits(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_MB", 0)
    assert submit(client, files={"file": ("a.txt", b"some bytes", "text/plain")}).status_code == 413
    assert list((Path(_TMP) / "uploads").glob("*")) == []
    monkeypatch.setattr(api, "MAX_TEXT_CHARS", 10)
    assert submit(client, data={"text": "x" * 11}).status_code == 413


def test_rate_limit_and_busy(client, monkeypatch):
    monkeypatch.setattr(api, "LIMITER", api.RateLimiter(2))
    codes = [submit(client, data={"text": SRC, "options": json.dumps({"check_web": False})}).status_code for _ in range(3)]
    assert codes == [202, 202, 429]
    monkeypatch.setattr(api, "LIMITER", api.RateLimiter(0))
    monkeypatch.setattr(api.JOBS, "pending", lambda: 999)
    assert submit(client, data={"text": SRC}).status_code == 503


def test_job_id_must_be_hex(client):
    assert client.get("/v1/jobs/../../etc/passwd", headers=H).status_code in (404, 422)
    assert client.get("/v1/jobs/not-a-valid-id", headers=H).status_code == 422
    assert client.get("/v1/jobs/" + "0" * 32, headers=H).status_code == 404


def test_report_before_done_is_conflict(client):
    api.JOBS.create("a" * 32, api.Options())
    assert client.get("/v1/jobs/" + "a" * 32 + "/report.html", headers=H).status_code == 409


def test_purge_and_restart_recovery():
    api.JOBS.create("b" * 32, api.Options())
    api.JOBS.fail_interrupted()
    assert api.JOBS.get("b" * 32)["status"] == "failed"
    api.JOBS.update("b" * 32, created=time.time() - 10 * 86400)
    assert api.JOBS.purge(7) >= 1 and api.JOBS.get("b" * 32) is None
