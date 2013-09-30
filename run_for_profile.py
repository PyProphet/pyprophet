import pyprophet


import pyprophet.config
import pyprophet.pyprophet
import os.path

config, __ = pyprophet.config.standard_config()
config["num_processes"] = 2
path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "tests", "test_data.txt")
a, b, c = pyprophet.pyprophet.PyProphet().process_csv(path, "\t")
print a


