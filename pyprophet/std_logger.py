# encoding: utf-8
from __future__ import print_function

import logging


def create(loggers=[]):
    if loggers:
        return loggers[0]
    logger = logging.getLogger("pyprophet")
    logger.setLevel("INFO")
    fmt = "%(levelname)-8s -- [pid=%(process)5d] : %(asctime)s: %(message)s"
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(fmt))
    logger.addHandler(h)
    loggers.append(logger)
    return logger

logging = create()
