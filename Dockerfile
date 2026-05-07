FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# ── Layer 1: dependencies only (cached until pyproject.toml/uv.lock change) ──
COPY pyproject.toml uv.lock ./
# --no-install-project: install all third-party deps but skip building the
# project itself (setuptools needs src/ which doesn't exist yet at this step)
RUN uv sync --no-dev --frozen --no-install-project

# ── Layer 2: project source ──
COPY . .

# Install the project now that src/ is present
RUN uv sync --no-dev --frozen

# Both src/ (agent, ml, common) and repo root (service, async_queue, approvals, dashboard)
ENV PYTHONPATH=/app/src:/app

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
