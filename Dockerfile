# Stage 1: Build ASAAS UI Frontend (React + Vite + TypeScript)
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Production Python FastAPI + Discovery Agents Backend
FROM python:3.12-slim AS runtime
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    PYTHONPATH=/app:/app/src

COPY requirements-deploy.txt ./
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy backend modules and agents
COPY api/ ./api/
COPY core/ ./core/
COPY utils/ ./utils/
COPY prompts/ ./prompts/
COPY seed/ ./seed/
COPY integrations/ ./integrations/
COPY ai/ ./ai/
COPY src/ ./src/
COPY ingestion_sub_agent.py external_search_sub_agent.py monitor_service.py ./

# Copy built frontend static assets
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
