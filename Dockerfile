# PyProphet Dockerfile
FROM python:3.7.3

# install numpy
RUN pip install numpy

# install PyProphet and dependencies
ADD . /pyprophet
WORKDIR /pyprophet
RUN python setup.py install
WORKDIR /
RUN rm -rf /pyprophet
