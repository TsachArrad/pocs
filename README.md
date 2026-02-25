# Azure OpenAI Python Server

Minimal FastAPI server with Azure OpenAI chat integration.

## 1) Setup

```powershell
cd c:\Users\User\pocs
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set your Azure values:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_DEPLOYMENT`
- Optional: `AZURE_OPENAI_API_VERSION`, `OPENAI_MAX_TOKENS`, `OPENAI_TEMPERATURE`

## 2) Run

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 3) Test

Health check:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/health -Method Get
```

Chat request:

```powershell
$body = @{
  messages = @(
    @{ role = "system"; content = "You are a concise assistant." },
    @{ role = "user"; content = "Give me 3 ideas for a weekend project." }
  )
  temperature = 0.7
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri http://localhost:8000/chat -Method Post -ContentType "application/json" -Body $body
```

## API

- `GET /health`
- `POST /chat`
  - Body:

```json
{
  "messages": [
    { "role": "user", "content": "Hello" }
  ],
  "model": "gpt-5",
  "temperature": 0.7,
  "max_tokens": 2000
}
```

- `POST /execute-connector`
  - Runs a connector definition inside Docker and returns parsed output.
  - Body (minimal):

```json
{
  "connector": {
    "runtime": {
      "language": "python",
      "entryPoint": "runner_entrypoint.py",
      "timeoutMs": 30000,
      "memoryMb": 256,
      "dependencies": [
        { "name": "requests", "version": "2.32.3" }
      ]
    },
    "configuration": {
      "spreadsheetId": "SPREADSHEET_ID",
      "sheetName": "Sheet1",
      "columnIndex": 3,
      "valueToInsert": "Hello"
    },
    "files": {
      "runner_entrypoint.py": "import json\nprint(json.dumps({'ok': True, 'result': {'hello': 'world'}}))"
    }
  },
  "env": {
    "CLIENT_ID": "...",
    "CLIENT_SECRET": "..."
  },
  "configuration_overrides": {
    "valueToInsert": "Value set at execution time"
  },
  "network_mode": "bridge"
}
```

Notes:
- Docker must be installed and reachable from the API server process.
- Currently supports `runtime.language = python`.
- `runtime.dependencies` is optional and can be either:
  - array of objects: `{ "name": "package", "version": "1.0.0" }`
  - array of strings: `"package"`
  - object map: `{ "package": "1.0.0", "other-package": "2.0.0" }`
- When dependencies are provided, they are installed with `pip` inside the container before entrypoint execution.
- For external APIs (Google, etc.) use `network_mode = bridge`; for isolated jobs use `none`.
- The engine scans code in `files` to detect referenced connector JSON paths (for example `connector.auth.clientId`, `context.connector.configuration.sheetName`).
- The engine enforces required top-level keys used by code (excluding: `schemaVersion`, `connectorType`, `displayName`, `description`, `version`, `notes`, `files`).
- If code references a connector path that is missing in JSON, execution is blocked with a 400 error listing missing paths.
- Metadata keys are ignored for matching/validation: `schemaVersion`, `connectorType`, `displayName`, `description`, `version`, `notes`, `files`.
- Required env vars are auto-detected from `{{env:KEY}}` placeholders and `process.env.KEY` usage in files; missing values fail fast with a 400 error.
