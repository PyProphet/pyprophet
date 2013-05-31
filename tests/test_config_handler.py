from nose.tools import *

import pyprophet.config_handler
import os.path

def data_path(fname):
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "data", fname)

def test_io():
    default = pyprophet.config_handler.default_config
    pyprophet.config_handler.write_config(default, data_path("test.json"))
    default2 = pyprophet.config_handler.read_config(data_path("test.json"))
    eq_(default, default2)

    pyprophet.config_handler.update(default,
                                     "init_run.select_positives_with_fdr", 0.2)
    eq_(default.init_run.select_positives_with_fdr, 0.2)
    pyprophet.config_handler.update(default, "num_cross_eval", 6)
    eq_(default.num_cross_eval, 6)

    key_pathes = pyprophet.config_handler.get_key_pathes(default)
    eq_(len(key_pathes), 5)



