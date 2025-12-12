# Trustworthy Model Registry (Phase 2)

A web-based registry for uploading, rating, searching, and downloading ML artifacts (models, datasets, and code). It runs as a FastAPI service on AWS Lambda + API Gateway with S3-backed persistence, and includes a React/TypeScript frontend.

**Team:** Aakash Bathini (@aakash-bathini), Neal Singh (@NSingh1227), Vishal Madhudi (@vishalm3416), Rishi Mantri (@rishimantri795)

## Production URLs

- **Frontend**: `https://main.d1vmhndnokays2.amplifyapp.com/dashboard`
- **Backend API**: `https://3vfheectz4.execute-api.us-east-1.amazonaws.com/prod`
- **Swagger / OpenAPI docs** (local): `http://localhost:8000/docs`

## What you can do

- **Upload** artifacts (ZIP upload for models, or create artifacts by URL)
- **Ingest** Hugging Face models
- **Rate** artifacts using Phase 1 + Phase 2 metrics
- **Search** by exact name or safe regex (README-aware)
- **Download** with sub-aspect filtering (full / weights / datasets / code)
- **Lineage**, **cost**, and **license compatibility** checks
- **Security track** features: sensitive model download monitoring, audit trail, package-confusion analytics

## Quick start (local)

### Backend

```bash
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Local dev uses SQLite (creates `registry.db`). In production, the backend uses S3.

### Frontend

```bash
cd frontend
npm install
npm start
```

The dev server runs on `http://localhost:3000` and talks to the backend at `http://localhost:8000`.

## Authentication

Default admin credentials (required by the assignment):
- **Username**: `ece30861defaultadminuser`
- **Password**: `correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;`

Get a JWT:

```bash
curl -X PUT http://localhost:8000/authenticate \
  -H "Content-Type: application/json" \
  -d '{"user": {"name": "ece30861defaultadminuser", "is_admin": true}, "secret": {"password": "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;"}}'
```

Then set it in the frontend (browser console):

```javascript
localStorage.setItem('token', 'bearer <jwt_token>');
location.reload();
```

## API overview

The backend serves Swagger UI at `/docs`.

Common endpoints:
- `DELETE /reset`
- `POST /models/ingest?model_name=<hf_repo_id>`
- `POST /models/upload` (ZIP upload)
- `GET /models/{id}/rate`
- `GET /models/{id}/download?aspect=full|weights|datasets|code`
- `POST /artifact/{artifact_type}` (create by URL)
- `GET /artifacts/{artifact_type}/{id}`
- `POST /artifact/byRegEx`
- `GET /artifact/byName/{name}`
- `GET /artifact/model/{id}/lineage`
- `GET /artifact/{artifact_type}/{id}/cost`
- `POST /artifact/model/{id}/license-check`

## Deployment

### Backend (Lambda)

Automated deployment is defined in `.github/workflows/cd.yml`.

It packages `app.py` + `src/` + `requirements-prod.txt` into a Lambda zip, uploads it to an S3 deployment bucket, and updates the Lambda function.

Required GitHub Secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `S3_BUCKET_NAME` (artifact storage bucket)
- `GH_TOKEN` (used for GitHub API calls where applicable)

### Frontend (Amplify)

The frontend is deployed via AWS Amplify. Set:
- `REACT_APP_API_URL` to your API Gateway base URL

## Testing

Backend:

```bash
pytest -q
pytest --cov=app --cov=src --cov-report=term-missing
```

Frontend:

```bash
cd frontend
npm test
npm run type-check
npm run lint
```

End-to-end UI tests:
- Selenium tests live in `tests/test_frontend_ui.py`.

## Accessibility / ADA

- The React UI uses accessible components and explicit ARIA/autocomplete attributes.
- We run Lighthouse checks and keep the UI WCAG-minded (keyboard navigation, labels, focus).

Example run:

```bash
lhci autorun --collect.url=https://main.d1vmhndnokays2.amplifyapp.com/dashboard
```

## Optional LLM helpers (no managed endpoints)

LLMs are **optional** helpers for README/relationship analysis. If no keys are configured, the system stays deterministic via heuristics.

Supported provider keys:
- `GEMINI_API_KEY` (optional `GEMINI_MODEL_ID`)
- `GEN_AI_STUDIO_API_KEY` (Purdue GenAI)

No SageMaker (or other managed endpoint) integration is used.

## Notes on course evaluation

- Latest autograder run reported: **300/322** (Dec 12, 2025)
- Code coverage: **61%** locally via `pytest --cov`
