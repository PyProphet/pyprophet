
import os
# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
os.putenv("OPENBLAS_NUM_THREADS", "1")


try:
    profile
except:
    profile = lambda x: x


from pyprophet import PyProphet
from config    import standard_config
import sys


def print_help():
    print
    script = os.path.basename(sys.argv[0])
    print "usage:"
    print "       %s [options] input_file" % script
    print "   or "
    print "       %s --help" % script
    print
    print "options:"
    print
    print "    --delim=[t|,|;]    default is ','"

    config = standard_config()
    dump_config_info(config)

def dump_config_info(config):
    print
    print "parameters:"
    print
    for k, v in sorted(config.items()):
        print "    --%-40s   default: %s" % (k,v)
    print

def dump_config(config):
    print
    print "used parameters:"
    print
    for k, v in sorted(config.items()):
        print "    %-40s   : %s" % (k,v)
    print

def main():

    options = [ p for p in sys.argv[1:] if p.startswith("--")]
    no_options = [ p for p in sys.argv[1:] if p not in options]

    options = dict()
    path = None

    if "--help" in sys.argv[1:]:
        print_help()
        return

    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            pre, __, post = arg.partition("=")
            options[pre[2:]]=post
        else:
            if path is not None:
                print_help()
                raise Exception("duplicate input file argument")
            path = arg

    delim = options.get("delim", ",")
    if delim == "tab":
        delim = "\t"

    if path is None:
        print_help()
        raise Exception("no input file given")

    config = standard_config()
    config.update(options)
    dump_config(config)

    config = standard_config()
    summary_table, final_table = PyProphet().process_csv(path, delim, config)
    print
    print "="*78
    print
    print summary_table
    print
    print "="*78

