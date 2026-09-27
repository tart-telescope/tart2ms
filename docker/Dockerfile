FROM python:3.12-slim

# git: uv fetches dask-ms from the tmolteno fork, a git source in
# pyproject.toml / uv.lock (the fork carries the casacure backend).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Dependencies first (a cached layer): exactly the versions in uv.lock,
# including dask-ms from the fork's git commit and casacure (no casacore
# C++ libraries or python-casacore needed).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Then tart2ms itself, from this source tree
COPY tart2ms ./tart2ms
RUN uv sync --frozen --no-dev --no-editable

ENV PATH=/opt/venv/bin:$PATH

# Fail the build unless dask-ms runs on the casacure backend
RUN python -c "import sys, tart2ms, daskms, casacure, casacure.tables; \
assert sys.modules['casacore.tables'] is sys.modules['casacure.tables'], sys.modules['casacore.tables']; \
print('tart2ms on dask-ms', daskms.__file__, 'with casacure', casacure.__version__)"

# Set metadata labels
LABEL org.opencontainers.image.source="https://github.com/tart-telescope/tart2ms"
LABEL org.opencontainers.image.description="Convert TART observation data to Measurement Sets"
LABEL org.opencontainers.image.licenses="GPL-3.0"

# Default command
CMD ["tart2ms", "--help"]
