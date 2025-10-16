# docker build --no-cache -t ghcr.io/pyprophet/pyprophet:3.0.2 .
# docker push ghcr.io/pyprophet/pyprophet:3.0.2

# ---------- builder: create venv and install ----------
FROM ubuntu:24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

# System Python + toolchain for building wheels (builder only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3 python3-pip python3-dev python3-venv \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Virtualenv (avoid PEP 668 issues)
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Upgrade pip and install toolchain/runtime deps
# Pins chosen to have manylinux wheels for Python 3.12
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir --prefer-binary \
        "numpy==1.26.4" "cython==0.29.36" "scipy==1.12.*" \
        duckdb seaborn psutil

# Bring in the PyProphet source tree
WORKDIR /src
COPY pyproject.toml setup.py requirements.txt README* LICENSE* ./
COPY pyprophet ./pyprophet

# Ensure setuptools/wheel are present (and avoid latest setuptools breaking flags)
RUN python -m pip install --no-cache-dir "setuptools<75" wheel

# --- force-build the Cython extension before installing ---
# This compiles pyprophet/scoring/_optimized.pyx into a .so in-place.
RUN python setup.py build_ext --inplace

# Install PyProphet into the venv (reuse prebuilt wheels, no isolation)
RUN pip install --no-cache-dir --no-build-isolation .

# Verify the optimized module is importable (fail-fast if not)
RUN python - <<'PY'
import importlib, sys
m = importlib.import_module('pyprophet.scoring._optimized')
print("Found optimized extension:", m.__file__)
PY

# Pre-install DuckDB sqlite_scanner extension (as requested)
RUN python - <<'PY'
import duckdb
con = duckdb.connect()
con.execute("INSTALL 'sqlite_scanner'")
con.execute("LOAD 'sqlite_scanner'")
print("duckdb sqlite_scanner installed & loadable")
PY

# Optional: trim pyc/caches
RUN python -m compileall -q "${VIRTUAL_ENV}" || true && \
    find "${VIRTUAL_ENV}" -type d -name "__pycache__" -prune -exec rm -rf {} + && \
    find "${VIRTUAL_ENV}" -type f -name "*.pyc" -delete

# ---------- runtime: only the venv + runtime libs ----------
FROM ubuntu:24.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

# Minimal native libs needed by pyopenms/pyprophet (safe superset)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3 \
    libglib2.0-0 libgomp1 \
    libxerces-c3.2 zlib1g libbz2-1.0 libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

# Bring the venv
COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Default workdir
WORKDIR /data/
