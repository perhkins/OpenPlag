# OpenPlag: Build Plan

*An open-source, zero-per-query-cost plagiarism and AI-content detector*

## 1. Goal and honest scope

**Goal:** a plagiarism checker that accepts PDF, DOC, DOCX, TXT and pasted text, finds the matching sources on the open web and in academic repositories, links directly to each source, and flags likely AI-generated passages, all without paid APIs.

**Reality check on "as good as Turnitin":**

- Turnitin's real moat is *data*, not algorithms: a large proprietary repository of previously submitted student papers, publisher agreements, and a huge web crawl. The core algorithms (fingerprinting, alignment, paraphrase detection) are well known and reproducible.
- So the plan is: match Turnitin on **algorithmic quality** from day one, then close the **data gap** over time with open dumps, a self-hosted crawler and an opt-in community fingerprint index.
- AI detection is a *probabilistic signal*, not proof. No detector (including Turnitin's) is reliable enough to be the sole basis for an accusation, and all are known to over-flag non-native English writers. OpenPlag will show per-passage confidence, calibrate for a low false-positive rate, and say so prominently in every report.

## 2. Architecture

Pipeline, left to right:

1. **Ingest** (PDF / DOCX / DOC / TXT / pasted text)
2. **Normalize** (strip hidden characters, undo homoglyph tricks, fix PDF hyphenation)
3. **Segment** (sentences, word tokens, quotes, bibliography)
4. **Query planner** (choose distinctive passages across the document)
5. **Source discovery** (free backends, run in parallel)
6. **Fetch and extract** (HTML, PDF, XML; cached; robots.txt aware)
7. **Matching engine** (exact/near via k-gram index, paraphrase via lexical + semantic)
8. **Aggregation** (non-overlapping attribution per source, similarity %)
9. **AI detector** (stylometry + paired-LLM Binoculars score, per segment)
10. **Report** (JSON now; HTML with highlights and links later)

## 3. Components

**Ingestion.** PyMuPDF (fallback pypdf) for PDF; python-docx (fallback: raw XML from the zip) for DOCX; antiword or headless LibreOffice for legacy DOC; trafilatura or a built-in parser for HTML. Scanned PDFs are detected (almost no extractable text) and routed to OCR (ocrmypdf / Tesseract) in Phase 1.

**Normalization and anti-evasion.** Unicode NFKC, removal of zero-width characters, and replacement of Cyrillic/Greek look-alike letters inside Latin words. These are common tricks to fool checkers, so the report warns when they are found.

**Query planner.** Split the document into buckets, pick the most distinctive 10-word window (rare, long, non-stopword terms) from each bucket, and search for it as an exact phrase. This spreads queries over the whole document instead of only its opening.

**Matching engine.**

- 5-word shingles hashed with BLAKE2b; every shingle of the submission is looked up in a hash index of each candidate source.
- Shared shingles are merged into spans with a small gap tolerance, so one-word substitutions, insertions and deletions still count as a single "near" match.
- False-positive controls: minimum span of 8 tokens, minimum content-word count, a density threshold, and later a corpus document-frequency filter for boilerplate phrases.
- **Winnowing** (the MOSS algorithm, window 4) is used for the persistent fingerprint store. It guarantees detection of any match of at least 8 tokens while storing only about 40% of the hashes, and it lets us index documents without keeping their text (privacy-friendly).
- **Paraphrase pass** on sentences not already covered: stemmed content-word Jaccard overlap first, then optional MiniLM sentence embeddings (local, free). Embedding hits must also share some vocabulary, to avoid flagging merely topical similarity.

**Scoring.** Each word is attributed to at most one source (largest source first), so percentages never double-count. Quoted text and the bibliography are excluded from both numerator and denominator, as Turnitin allows, but quoted matches are still shown.

**Direct source links.** Every match carries the source URL. For web pages we also generate a browser *text-fragment* link (`#:~:text=...`) that scrolls to and highlights the matched passage in Chrome, Edge and Safari.

## 4. Free source discovery (no API cost)

**Tier 0: live free endpoints (in the prototype)**

- Wikipedia search API
- OpenAlex (free, no key; metadata plus abstracts plus OA PDF links)
- Europe PMC (free full-text phrase search over open-access biomedical literature)
- DuckDuckGo HTML results (free but scraped, so fragile and rate-limited)
- Self-hosted **SearXNG** (recommended replacement for DuckDuckGo scraping: free, JSON output, aggregates many engines)
- arXiv API (metadata search only)

**Tier 0b: to add:** Crossref, Semantic Scholar (free tier), CORE, Internet Archive, Marginalia.

**Tier 1: local corpus.** Folders of reference documents and, opt-in, previous submissions via the SQLite fingerprint store.

**Tier 2: your own offline index from open dumps** (zero per-query cost, one-time compute and storage): Wikipedia dumps, OpenAlex snapshot, arXiv bulk metadata/full text, PubMed Central OA subset, S2ORC, Project Gutenberg, and Common Crawl segments. Index them as winnowed fingerprints in Postgres or a search engine such as Tantivy / Elasticsearch.

**Tier 3: community index.** Opt-in institutions contribute only hashed fingerprints (never text), growing a shared repository over time. This is the open-source answer to Turnitin's student-paper database.

## 5. AI-content detection

- **Layer 1, stylometry (no dependencies):** sentence-length burstiness and density of phrases that LLMs overuse. Cheap but weak; used as a minor signal and as a fallback.
- **Layer 2, Binoculars (zero-shot, local):** the ratio of an observer model's perplexity to the cross-perplexity between an observer/performer pair of small models (default Qwen2.5-0.5B and its Instruct variant). Scored over rolling windows to localize flagged passages. Needs **calibration** on your own human/AI samples; the code ships a `calibrate()` helper that picks a threshold at a target false-positive rate.
- **Layer 3 (later):** a fine-tuned DeBERTa-class classifier trained on open datasets, ensembled with Layer 2.
- **Safeguards:** minimum text length, report at ≤1% target false-positive rate, a mandatory fairness test on non-native English essays, and wording that never says "proves".
- **Known limits:** paraphrasing tools and "humanizers" reduce detection; short texts are unreliable; heavily edited AI text is hard to catch. Say this in the UI.

## 6. Roadmap

**Phase 0, weeks 1-2: prototype.** The single-file `openplag.py` delivered with this plan: ingestion, normalization, discovery, matching, scoring, AI signals, CLI.

**Phase 1, weeks 3-6: hardening.** Unit tests, OCR for scanned PDFs, citation-aware exclusion, boilerplate filter, retry and backoff, disk cache tuning, standalone HTML report with colour-coded highlights.

**Phase 2, weeks 7-12: service.** FastAPI backend, job queue (RQ or Celery + Redis), web UI with a paste box and file upload, docker-compose bundle including SearXNG, user-configurable settings (exclude quotes, bibliography, small matches).

**Phase 3, weeks 13-20: data.** Build offline indexes from the open dumps above; fingerprint DB in Postgres; background crawler for domains that matter to your users; deduplication.

**Phase 4, weeks 21-28: AI detector.** Calibration, ensemble, per-segment output, fairness audits, adversarial testing against paraphrasers.

**Phase 5: community.** Federated fingerprint sharing, Moodle/Canvas LTI plugins, docs, contributor guide, public benchmark leaderboard.

## 7. Evaluation plan

- **Plagiarism:** PAN text-alignment corpora (precision, recall, granularity, plagdet), plus synthetic tests: take Wikipedia and open-access passages and apply obfuscation (verbatim, word swaps, reordering, back-translation, LLM paraphrase).
- **AI detection:** RAID, HC3, M4 and similar benchmarks; report AUROC and TPR at 1% FPR; include a non-native-English essay set to measure bias.
- **CI regression suite:** fixed sample documents with expected matches and scores, run on every commit.

## 8. Tech stack and layout

- Python 3.10+, `requests`, optional `trafilatura`, `PyMuPDF`, `python-docx`, `sentence-transformers`, `transformers` + `torch`.
- Suggested layout: `openplag/ingest`, `normalize`, `discovery/backends`, `fetch`, `match`, `paraphrase`, `store`, `ai_detect`, `report`, `api`, `cli`, plus `tests/` and `benchmarks/`.

## 9. Risks, legal and privacy

- **Scraping etiquette and terms:** respect robots.txt, rate-limit per host, cache aggressively, identify the bot with a contact address. Prefer APIs and SearXNG over scraping search engines.
- **Student data:** documents are processed in memory by default; stored only as fingerprints and only with explicit opt-in (FERPA / GDPR considerations).
- **Copyright:** cache fetched text with a short TTL; long-term store only hashes.
- **AI false accusations:** hard product rule that AI scores are advisory, always shown with caveats and confidence.
- **Search-engine fragility:** DuckDuckGo scraping can break or block; keep SearXNG and offline indexes as the dependable path.

## 10. Zero-cost checklist

- No paid search or LLM APIs: only free public APIs, scraping with polite limits, self-hosted SearXNG, and local models.
- One-time costs only: disk for dumps and indexes, CPU/GPU time for embeddings and indexing.
- Everything runs on one modest server; scale out by sharding the fingerprint index.
