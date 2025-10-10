# docker build -t ghcr.io/pyprophet/pyprophet:3.0.2 .
# docker push ghcr.io/pyprophet/pyprophet:3.0.2

# PyProphet Dockerfile (Ubuntu 24.04 + venv)
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3 python3-pip python3-dev python3-venv \
    build-essential git \
    # --- runtime libs needed by pyopenms ---
    libglib2.0-0 \
    libgomp1 \
    # these are rarely needed, but harmless and useful if pyopenms/OpenMS is present
    libxerces-c3.2 \
    zlib1g \
    libbz2-1.0 \
    libsqlite3-0 \
 && rm -rf /var/lib/apt/lists/*

# venv
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# toolchain (pins optional)
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir "numpy==1.26.4" "cython==0.29.36" duckdb seaborn psutil

# install PyProphet
ADD . /pyprophet
WORKDIR /pyprophet
RUN pip install .

# warm up duckdb sqlite_scanner (optional)
RUN python - <<'PY'
import duckdb
con = duckdb.connect()
con.execute("INSTALL 'sqlite_scanner'")
con.execute("LOAD 'sqlite_scanner'")
PY

WORKDIR /data/
