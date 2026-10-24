# Test documents

Place documents submitted to OpenPlag in this folder when manually testing
file ingestion. The application accepts `.txt`, `.pdf`, `.docx`, and legacy
`.doc` files.

Keep downloaded or private documents out of version control; the directory is
ignored by default except for this README.

## Local corpus test

From the repository root, run:

```powershell
.venv\Scripts\python.exe openplag.py tests\docs\sample_submission.txt --no-web --no-ai --local-corpus tests\ref
```

## Online test

Online runs are opt-in because providers can rate-limit requests. Load the
ignored `.env` values into the current PowerShell session, then run:

```powershell
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#=]+)\s*=\s*(.*)\s*$') {
        [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), 'Process')
    }
}
.venv\Scripts\python.exe openplag.py tests\docs\sample_submission.txt --no-ai
```

Set `OPENPLAG_SEARXNG` or an API key in `.env` before using providers that
require configuration. Routine unit tests remain offline.
