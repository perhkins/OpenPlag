# OpenPlag

Open-source plagiarism + AI-content checker. No paid APIs.

* `openplag.py` - the detection engine (also usable from the command line)
* `api/` - FastAPI service (job queue, file upload, reports)
* `ui/` - Streamlit web page (paste box + file upload)
* `deploy/` - Dockerfiles, SearXNG config, HTTPS proxy config
* `tests/` - `pytest -q`

## Quick start (your own computer)

1. Install **Docker Desktop** (Windows/macOS) or Docker Engine (Linux).
2. In this folder:

```bash
cp .env.example .env                                   # Windows: copy .env.example .env
# put a random secret into SearXNG's config (Mac/Linux):
sed -i "s|ultrasecretkey|$(openssl rand -hex 32)|" deploy/searxng/settings.yml
# macOS needs:  sed -i '' "s|ultrasecretkey|$(openssl rand -hex 32)|" deploy/searxng/settings.yml
docker compose up -d --build
```

3. Open http://localhost:8501

Edit `.env` first (API key, contact e-mail). The full beginner walkthrough, including putting it on a
server with HTTPS, is in the deployment guide that came with this project.

## Without Docker (development)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt -r requirements-ui.txt
export OPENPLAG_API_KEY=dev-key
uvicorn api.main:app --port 8000 &
OPENPLAG_API_KEY=dev-key streamlit run ui/app.py
pytest -q
```

## Command line

```bash
python openplag.py essay.pdf --html report.html --searxng http://localhost:8080
python openplag.py --text "pasted text ..." --no-web
```

## Security and privacy notes

* Set `OPENPLAG_UI_PASSWORD` for any public deployment (the production compose file refuses to start without it).
* The API is never published to the internet by the compose files; only the HTTPS proxy is.
* Uploaded files are deleted right after text extraction; reports are purged after `OPENPLAG_RETENTION_DAYS`.
* The HTML report escapes all dynamic content, but Streamlit embeds it as-is: never feed it untrusted HTML.
* AI-content scores are probabilistic signals, not proof. Do not use them as sole evidence against anyone.
