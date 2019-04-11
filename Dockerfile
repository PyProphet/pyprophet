# PyProphet Dockerfile
FROM ubuntu:18.04

# install base dependencies
RUN apt-get -y update
RUN apt-get install -y python3 python3-pip python3-numpy

# patch Python
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# install PyProphet and dependencies
ADD . /pyprophet
WORKDIR /pyprophet
RUN python3 setup.py install
WORKDIR /
RUN rm -rf /pyprophet
