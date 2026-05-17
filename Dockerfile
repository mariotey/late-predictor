FROM python:3.11

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Dependency layer (cached)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy app
COPY . .

# Make "src" importable
ENV PYTHONPATH=/app/src

# Expose port for FastAPI
EXPOSE 10000

# Run FastAPI
CMD ["/app/.venv/bin/uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "10000"]