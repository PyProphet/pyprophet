import pyprophet


import pyprophet.config
import pyprophet.pyprophet
import os.path
import warnings

import logging
format_ = "%(levelname)s -- [pid=%(process)s] : %(asctime)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=format_)

logging.info("started")

try:
    profile
except:
    profile = lambda x: x

@profile
def run():
    CONFIG, __ = pyprophet.config.standard_config()
    CONFIG["num_processes"] = 1
    CONFIG["xeval.num_iter"] = 1
    logging.info("config settings:")
    for k, v in sorted(CONFIG.items()):
        logging.info("    %s: %s" % (k, v))
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "tests", "test_data.txt")
    path = "data3.csv"
    path = "/scratch/georger/pyprophet/napedro_L120420_005_SW.csv"
    path = "tests/test_data.txt"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 
        a, b, c = pyprophet.pyprophet.PyProphet().process_csv(path, "\t")
        print
        print a


run()

