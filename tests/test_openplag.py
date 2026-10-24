"""Offline unit tests for OpenPlag (no network). Run:  pytest -q"""
import json
import random
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openplag as op  # noqa: E402

SRC = ("Photosynthesis is the process by which green plants and some other organisms use sunlight to "
       "synthesize nutrients from carbon dioxide and water. Photosynthesis in plants generally involves "
       "the green pigment chlorophyll and generates oxygen as a byproduct. The process takes place "
       "primarily in the chloroplasts of leaf cells, where light energy is converted into chemical "
       "energy stored in glucose molecules.")
OWN = ("My garden has tomatoes, basil and a stubborn lemon tree that refuses to fruit despite constant "
       "attention from me every single weekend morning.")


def detector(tmp_path, **cfg):
    return op.PlagiarismDetector(op.Config(cache_dir=tmp_path / "cache", **cfg), backends=[])


def local(d, text=SRC, title="bio.txt"):
    d.local_sources.append(op.Source(f"https://example.org/{title}", title, "local", text))
    return d


# ---------------------------------------------------------------- text utils
def test_normalize_hidden_and_homoglyphs():
    txt, st = op.normalize_text("Photosynth\u0435sis\u200b is gr\u0435at")   # Cyrillic e + zero-width space
    assert txt == "Photosynthesis is great"
    assert st["hidden_chars"] == 1 and st["homoglyphs"] == 2


def test_normalize_keeps_real_cyrillic():
    txt, st = op.normalize_text("Привет мир")
    assert txt == "Привет мир" and st["homoglyphs"] == 0


def test_normalize_dehyphenates_pdf_linebreaks():
    assert op.normalize_text("photo-\nsynthesis")[0] == "photosynthesis"


def test_sentence_split_respects_abbreviations():
    t = "Smith et al. showed this. Dr. Jones disagreed, e.g. in 2020. Final one."
    spans = op.split_sentences(t)
    assert [t[a:b] for a, b in spans] == ["Smith et al. showed this.", "Dr. Jones disagreed, e.g. in 2020.", "Final one."]


def test_winnowing_guarantee():
    """Any shared run of >= w+k-1 tokens must share at least one fingerprint."""
    k, w = 5, 4
    rng = random.Random(7)
    vocab = [f"w{i}" for i in range(5000)]
    for _ in range(60):
        shared = [rng.choice(vocab) for _ in range(w + k - 1)]
        a = [rng.choice(vocab) for _ in range(rng.randint(5, 80))] + shared + [rng.choice(vocab) for _ in range(40)]
        b = [rng.choice(vocab) for _ in range(rng.randint(5, 80))] + shared + [rng.choice(vocab) for _ in range(40)]
        fa = set(op.winnow(op.kgram_hashes(a, k), w))
        fb = set(op.winnow(op.kgram_hashes(b, k), w))
        assert fa & fb


# ------------------------------------------------------------------ matching
def test_verbatim_copy(tmp_path):
    rep = local(detector(tmp_path)).analyze(SRC, web=False, check_ai=False)
    assert rep.similarity_percent > 90
    assert rep.sources[0].url == "https://example.org/bio.txt"


def test_unrelated_text_has_no_matches(tmp_path):
    rep = local(detector(tmp_path)).analyze(OWN * 2, web=False, check_ai=False)
    assert rep.similarity_percent == 0 and not rep.matches


def test_edited_copy_is_near_match(tmp_path):
    edited = SRC.replace("generates oxygen", "produces oxygen").replace("primarily", "mainly")
    rep = local(detector(tmp_path)).analyze(edited, web=False, check_ai=False)
    assert rep.similarity_percent > 70
    assert {m.kind for m in rep.matches} <= {"exact", "near"}


def test_paraphrase_detected(tmp_path):
    src = ("Plants convert sunlight, water and carbon dioxide into nutrients through photosynthesis, "
           "relying on the green pigment chlorophyll.")
    para = ("Through photosynthesis, plants turn sunlight, water and carbon dioxide into nutrients, "
            "depending upon the green pigment chlorophyll. Pasta should boil for ten minutes in salted water.")
    rep = local(detector(tmp_path), src).analyze(para, web=False, check_ai=False)
    assert any(m.kind == "paraphrase" for m in rep.matches)


def test_quotes_are_excluded_from_score(tmp_path):
    q = ("As the textbook puts it, \u201cThe process takes place primarily in the chloroplasts of leaf cells, "
         "where light energy is converted into chemical energy\u201d and I agree. " + OWN)
    rep = local(detector(tmp_path)).analyze(q, web=False, check_ai=False)
    assert rep.similarity_percent == 0
    assert any(m.quoted for m in rep.matches)


def test_references_section_ignored(tmp_path):
    body = OWN * 3 + "\n\nReferences\n" + SRC
    rep = local(detector(tmp_path)).analyze(body, web=False, check_ai=False)
    assert rep.similarity_percent == 0


def test_citation_markers_not_counted(tmp_path):
    d = detector(tmp_path)
    doc = op.Prepared("Plants grow well (Smith & Jones, 2020, p. 4) in light [1, 2] and also [3-5] in shade.", 5)
    mask = d._excluded_mask(doc)
    words = doc.tok.words
    assert all(mask[i] for i, w in enumerate(words) if w in {"smith", "jones", "2020", "4"})
    assert not mask[words.index("plants")] and not mask[words.index("shade")]


def test_common_phrase_filter(tmp_path):
    phrase = "important to note that the results of the study were significant"
    docs = [f"Unrelated filler {i} about gardens and weather. It is {phrase} overall in {i} cases." for i in range(4)]
    text = f"My own opening about tomatoes. It is {phrase} here, said nobody."
    d = detector(tmp_path)
    d.local_sources = [op.Source(f"https://e.org/{i}", f"s{i}", "local", t) for i, t in enumerate(docs)]
    assert d.analyze(text, web=False, check_ai=False).similarity_percent == 0       # 4 sources -> boilerplate
    d2 = detector(tmp_path)
    d2.local_sources = d.local_sources[:1]
    assert d2.analyze(text, web=False, check_ai=False).similarity_percent > 0       # 1 source -> real match


def test_highlight_url_for_web_sources(tmp_path):
    d = detector(tmp_path)
    d.local_sources.append(op.Source("https://example.org/page", "page", "web", SRC))
    rep = d.analyze(SRC, web=False, check_ai=False)
    assert rep.matches[0].highlight_url.startswith("https://example.org/page#:~:text=")


# ----------------------------------------------------------- fingerprint store
def test_store_roundtrip_and_no_text(tmp_path):
    cfg = op.Config(cache_dir=tmp_path / "c")
    st = op.FingerprintStore(str(tmp_path / "s.db"), cfg)
    st.add(SRC, "Stored bio", "https://example.org/bio")
    d = op.PlagiarismDetector(cfg, backends=[], store=st)
    rep = d.analyze(OWN + " " + SRC, web=False, check_ai=False)
    assert rep.matches and rep.matches[0].kind == "corpus" and rep.matches[0].source_title == "Stored bio"
    cols = {r[1] for t in ("docs", "fps") for r in sqlite3.connect(str(tmp_path / "s.db")).execute(f"PRAGMA table_info({t})")}
    assert cols == {"id", "title", "url", "kind", "h", "doc"}                         # hashes only, no text column


def test_store_df_filter_drops_boilerplate(tmp_path):
    cfg = op.Config(cache_dir=tmp_path / "c", store_max_df=2)
    st = op.FingerprintStore(str(tmp_path / "s.db"), cfg)
    for i in range(4):
        st.add(f"Different opening number {i} about gardens. {SRC}", f"doc{i}")
    fps = op.winnow(op.kgram_hashes(op.tokenize(op.normalize_text(SRC)[0]).words, cfg.k), cfg.window)
    shared = st.query(fps)
    assert not shared                                                                  # in 4 docs > max_df=2


# -------------------------------------------------------------------- ingestion
def test_extract_txt_and_docx(tmp_path):
    t = tmp_path / "a.txt"
    t.write_text("hello text")
    assert op.extract_text(str(t)) == "hello text"
    dx = tmp_path / "b.docx"
    with zipfile.ZipFile(dx, "w") as z:                                                # malformed docx -> XML fallback
        z.writestr("word/document.xml", "<w:document><w:p><w:r><w:t>Hello docx</w:t></w:r></w:p></w:document>")
    assert "Hello docx" in op.extract_text(str(dx))


def test_pdf_ocr_fallback(monkeypatch):
    monkeypatch.setattr(op, "_pdf_native", lambda data, max_pages=300: "")
    called = []
    monkeypatch.setattr(op, "_ocr_pdf", lambda data, max_pages=60: called.append(1) or "scanned words")
    assert op.pdf_to_text(b"%PDF", ocr=True) == "scanned words" and called
    called.clear()
    assert op.pdf_to_text(b"%PDF", ocr=False) == "" and not called
    monkeypatch.setattr(op, "_pdf_native", lambda data, max_pages=300: "x" * 500)       # has a text layer
    assert op.pdf_to_text(b"%PDF", ocr=True) == "x" * 500 and not called


def test_html_to_text_strips_scripts_and_nav():
    t = op.html_to_text("<html><nav>menu</nav><p>Hello <b>world</b></p><script>x=1</script></html>")
    assert "Hello world" in t and "menu" not in t and "x=1" not in t


# ------------------------------------------------------------------------- http
class FakeResponse:
    def __init__(self, status, body=b"ok", headers=None):
        self.status_code, self._b, self.headers, self.encoding = status, body, headers or {}, "utf-8"

    def iter_content(self, n):
        yield self._b

    def close(self):
        pass


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls, self.headers = list(responses), [], {}

    def get(self, url, **kw):
        self.calls.append(url)
        return self.responses.pop(0)


def test_http_retries_with_backoff(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    h = op.Http(op.Config())
    h.s = FakeSession([FakeResponse(503), FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200, b"data")])
    r = h.request("https://api.example.org/x", retries=2)
    assert r.status == 200 and r.data == b"data" and len(h.s.calls) == 3
    assert any(6.9 <= s <= 7.5 for s in sleeps)                                        # Retry-After honoured


def test_http_does_not_retry_403_or_404(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    h = op.Http(op.Config())
    h.s = FakeSession([FakeResponse(404)])
    assert h.request("https://x.org/", retries=3).status == 404 and len(h.s.calls) == 1


def test_disk_cache_negative_ttl_and_prune(tmp_path):
    c = op.DiskCache(tmp_path, ttl_days=1, max_mb=1)
    c.put("k", "")
    assert c.get("k") == ""
    p = c._p("k")
    o = json.loads(p.read_text())
    o["ts"] -= 4000                                                                     # older than 1h negative TTL
    p.write_text(json.dumps(o))
    assert c.get("k") is None
    for i in range(30):
        (tmp_path / f"{i}.json").write_text("x" * 100_000)
    op.DiskCache(tmp_path, ttl_days=1, max_mb=1)                                        # prune on init
    assert sum(f.stat().st_size for f in tmp_path.glob("*.json")) <= 1_000_000


# ------------------------------------------------------------ SearXNG failover
class FakeHttp:
    """Scripted by host: value is a Resp or a callable returning one."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def request(self, url, params=None, headers=None, retries=0, timeout=0, gap=None, **kw):
        host = url.split("/")[2]
        self.calls.append(host)
        v = self.script.get(host, op.Resp(0))
        return v() if callable(v) else v

    def get(self, url, params=None, headers=None, **kw):
        host = url.split("/")[2]
        self.calls.append(host)
        v = self.script.get(host)
        return None if v is None else (json.dumps(v).encode(), "application/json", "utf-8")


def ok_json(*urls):
    body = json.dumps({"results": [{"url": u, "title": "t"} for u in urls]}).encode()
    return op.Resp(200, body, "application/json")


def test_searx_pool_fails_over_and_remembers():
    http = FakeHttp({"a.org": op.Resp(403), "b.org": op.Resp(429), "c.org": ok_json("https://hit.example/1")})
    pool = op.SearXNGPool(http, op.Config(), ["https://a.org", "https://b.org", "https://c.org"])
    assert [r.url for r in pool.search("some phrase", 5)] == ["https://hit.example/1"]
    assert http.calls == ["a.org", "b.org", "c.org"]
    http.calls.clear()
    pool.search("another phrase", 5)
    assert http.calls == ["c.org"]                                                      # a & b are cooling down


def test_searx_pool_detects_html_and_blocked_engines():
    html_resp = op.Resp(200, b"<html>no json</html>", "text/html")
    blocked = op.Resp(200, json.dumps({"results": [], "unresponsive_engines": [["g", "CAPTCHA"], ["b", "x"], ["d", "y"]]}).encode())
    http = FakeHttp({"a.org": html_resp, "b.org": blocked, "c.org": ok_json("https://x.example/")})
    pool = op.SearXNGPool(http, op.Config(), ["https://a.org", "https://b.org", "https://c.org"])
    assert pool.search("q", 3)[0].url == "https://x.example/"
    assert http.calls == ["a.org", "b.org", "c.org"]


def test_searx_health_persists_across_runs(tmp_path):
    cache = op.DiskCache(tmp_path, 1)
    http = FakeHttp({"a.org": op.Resp(429), "b.org": ok_json("https://z.example/")})
    op.SearXNGPool(http, op.Config(), ["https://a.org", "https://b.org"], cache=cache).search("q", 3)
    assert http.calls == ["a.org", "b.org"]
    http.calls.clear()
    op.SearXNGPool(http, op.Config(), ["https://a.org", "https://b.org"], cache=cache).search("q", 3)   # new process
    assert http.calls == ["b.org"]                                                      # a.org remembered as rate-limited


def test_searx_pool_all_down_returns_empty():
    pool = op.SearXNGPool(FakeHttp({}), op.Config(), ["https://a.org", "https://b.org"])
    assert pool.search("q", 3) == []


def test_searx_pool_empty_but_healthy_does_not_failover():
    http = FakeHttp({"a.org": op.Resp(200, b'{"results": []}', "application/json"), "b.org": ok_json("https://y/")})
    pool = op.SearXNGPool(http, op.Config(), ["https://a.org", "https://b.org"])
    assert pool.search("q", 3) == [] and http.calls == ["a.org"]


def test_searx_discovery_used_when_seeds_fail():
    listing = {"instances": {
        "https://good.org/": {"http": {"status_code": 200}, "timing": {"search": {"success_percentage": 100, "all": {"median": 0.4}}}},
        "https://slow.org/": {"http": {"status_code": 200}, "timing": {"search": {"success_percentage": 60, "all": {"median": 2}}}},
        "http://hidden.onion/": {"http": {"status_code": 200}},
        "https://down.org/": {"http": {"status_code": 503}}}}
    http = FakeHttp({"seed.org": op.Resp(403), "searx.space": op.Resp(200, json.dumps(listing).encode()),
                     "good.org": ok_json("https://found.example/")})
    pool = op.SearXNGPool(http, op.Config(), ["https://seed.org"], discover=True)
    assert pool.search("q", 3)[0].url == "https://found.example/"
    assert pool.instances[:3] == ["https://seed.org", "https://good.org", "https://slow.org"]
    assert not any(".onion" in i or "down.org" in i for i in pool.instances)


# ------------------------------------------------------------- backend parsers
def test_tier0b_backend_parsers():
    cfg = op.Config()
    http = FakeHttp({
        "api.crossref.org": {"message": {"items": [{"DOI": "10.1/x", "title": ["T"], "URL": "https://doi.org/10.1/x",
                                                     "abstract": "<jats:p>Hello <i>abstract</i></jats:p>"}]}},
        "api.semanticscholar.org": {"data": [{"title": "S", "abstract": "abs", "url": "https://s2/1",
                                              "externalIds": {"DOI": "10.2/y"}, "openAccessPdf": {"url": "https://p/pdf"}}]},
        "api2.marginalia-search.com": {"results": [{"url": "https://small.example/a", "title": "A"}]},
        "archive.org": {"response": {"docs": [{"identifier": "bk1", "title": "Book"}]}},
    })
    c = op.Crossref(http, cfg).search("q", 3)[0]
    assert c.url == "https://doi.org/10.1/x" and "abstract" in c.text and "<" not in c.text
    s = op.SemanticScholar(http, cfg).search("q", 3)[0]
    assert s.url == "https://doi.org/10.2/y" and s.fetch_url == "https://p/pdf"
    assert op.Marginalia(http, cfg).search("q", 3)[0].url == "https://small.example/a"
    a = op.InternetArchive(http, cfg).search("photosynthesis chlorophyll pigment", 3)[0]
    assert a.url.endswith("/details/bk1") and a.fetch_url.endswith("bk1_djvu.txt")


def test_core_requires_key(monkeypatch):
    http = FakeHttp({"api.core.ac.uk": {"results": [{"doi": "10.3/z", "title": "C", "fullText": "full text here"}]}})
    monkeypatch.delenv("CORE_API_KEY", raising=False)
    assert op.CORE(http, op.Config()).search("q", 3) == []
    monkeypatch.setenv("CORE_API_KEY", "k")
    r = op.CORE(http, op.Config()).search("q", 3)[0]
    assert r.url == "https://doi.org/10.3/z" and r.text == "full text here"


def test_build_backends_composition(monkeypatch):
    monkeypatch.delenv("CORE_API_KEY", raising=False)
    monkeypatch.delenv("S2_API_KEY", raising=False)
    cfg = op.Config()
    assert [b.name for b in op.build_backends(None, cfg)] == ["wikipedia", "openalex", "europepmc", "crossref"]
    monkeypatch.setenv("S2_API_KEY", "k")
    assert "semanticscholar" in [b.name for b in op.build_backends(None, cfg)]
    pool = op.build_backends(None, cfg, searxng_url="https://mine.org")[-1]
    assert pool.name == "searxng" and pool.instances == ["https://mine.org"] and not pool.discover
    pub = op.build_backends(None, cfg, public_searx=True)[-1]                           # opt-in only
    assert pub.name == "searxng" and pub.discover


# ------------------------------------------------- discovery cache & fetching
class CountingBackend(op.Backend):
    name = "counting"

    def __init__(self):
        self.n = 0

    def search(self, phrase, n):
        self.n += 1
        return [op.Source("https://example.org/a", "A", "web", text="x")]


def test_search_results_cached(tmp_path):
    d = detector(tmp_path)
    d.backends = [CountingBackend()]
    d._discover(["alpha beta gamma"])
    d._discover(["alpha beta gamma"])
    assert d.backends[0].n == 1


def test_fetch_failure_keeps_abstract_and_is_negative_cached(tmp_path):
    d = detector(tmp_path)

    class Down:
        calls = 0

        def get(self, *a, **k):
            Down.calls += 1

    d.http = Down()
    s = op.Source("https://doi.org/1", "t", "academic", text="abstract text", fetch_url="https://x.org/p.pdf")
    d._fetch_one(s)
    d._fetch_one(s)
    assert s.text == "abstract text" and Down.calls == 1


# --------------------------------------------------------------------- AI + HTML
def test_ai_stylometry_and_short_text():
    ai = " ".join(["Furthermore, it is important to note that this multifaceted topic plays a crucial role in "
                   "the ever-evolving landscape of society and underscores the pivotal need to delve into nuances."] * 10)
    r = op.detect_ai(ai, op.split_sentences(ai), None)
    assert r.label == "likely_ai" and r.confidence == "low"
    short = op.detect_ai("Too short.", op.split_sentences("Too short."), None)
    assert short.label == "insufficient_text" and short.score is None


def test_html_report_escapes_and_links(tmp_path):
    d = local(detector(tmp_path), SRC + " <script>alert(1)</script>")
    d.local_sources[0].url = "https://example.org/bio"
    d.local_sources[0].kind = "web"
    doc = SRC + " <script>alert(1)</script> " + OWN
    rep = d.analyze(doc, web=False, check_ai=False)
    html = op.render_html(rep)
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html
    assert 'class="m ' in html and "https://example.org/bio" in html
    # non-http(s) URLs are never linkified
    rep.sources[0].url = "javascript:alert(1)"
    rep.matches[0].source_url, rep.matches[0].highlight_url = "javascript:alert(1)", None
    assert 'href="javascript:' not in op.render_html(rep)


def test_progress_callback_and_report_roundtrip(tmp_path):
    seen = []
    d = local(detector(tmp_path))
    rep = d.analyze(SRC + " " + OWN, web=False, check_ai=True, progress=lambda stage, f: seen.append((stage, f)))
    fracs = [f for _, f in seen]
    assert fracs == sorted(fracs) and fracs[0] < 0.1 and fracs[-1] > 0.9
    back = op.report_from_dict(json.loads(json.dumps(__import__("dataclasses").asdict(rep))))
    assert back.similarity_percent == rep.similarity_percent and back.matches[0].text == rep.matches[0].text
    assert op.render_html(back) == op.render_html(rep)
