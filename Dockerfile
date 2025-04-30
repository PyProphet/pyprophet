# PyProphet Dockerfile
FROM python:3.9.1

# install numpy & cython
RUN pip install numpy cython

# install duckdb and its extensions before pyprophet
RUN pip install duckdb


# install PyProphet and dependencies
ADD . /pyprophet
WORKDIR /pyprophet
# RUN python setup.py install
RUN pip install .
RUN python -c "import duckdb; conn = duckdb.connect(); conn.execute(\"INSTALL 'sqlite_scanner'\"); conn.execute(\"LOAD 'sqlite_scanner'\");"
WORKDIR /
RUN rm -rf /pyprophet

# Set final working directory, useful for when binding to a local mount
WORKDIR /data/
