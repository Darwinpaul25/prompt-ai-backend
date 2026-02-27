# Cloudflare Deploy (Container)

This FastAPI backend uses SQLAlchemy + Gemini and should be deployed on Cloudflare as a **containerized service**.

## 1. Build Command / Runtime
- The service runs from `Dockerfile` in repo root.
- App start command inside container:
  - `uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}`

## 2. Required Environment Variables
Set these in Cloudflare:
- `GOOGLE_API_KEY` = your Gemini API key
- `JWT_SECRET_KEY` = long random secret
- `DATABASE_URL` = Postgres connection string
- `JWT_EXPIRE_MINUTES` = optional (default `1440`)
- `PORT` = optional (defaults to `8080`)

## 3. Database
Use managed Postgres (recommended). Do not rely on local files in production.

## 4. Health Check
Use `/docs` or any API endpoint such as `GET /sessions` (with auth token) for smoke testing.

## 5. Auth Flow for Frontend
1. `POST /auth/token` with `{ "user_id": "<id>" }`
2. Use returned bearer token on all protected endpoints:
   - `POST /chat`
   - `GET /sessions`
   - `GET /history/{session_id}`
   - `POST /reset`
   - `GET /summary/{session_id}`
