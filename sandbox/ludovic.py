import sys
import pylab
import pyprophet.stats as stats
import random
import numpy as np

path = sys.argv[1]

lines = open(path).readlines()
header = [h.lower() for h in lines[0].split()]

data_lines = [line.split() for line in lines[1:]]

if header == ["decoy", "score"]:
    data_lines = [(np.log(float(s.strip())), l.strip()) for (l, s) in data_lines]
    targets = [s for (s, l) in data_lines if l.upper() == "FALSE"]
    decoys = [s for (s, l) in data_lines if l.upper() == "TRUE"]
    invalid = [(s, l) for (s, l) in data_lines if l.upper() not in ("FALSE", "TRUE")]
elif header == ["score", "decoy"]:
    data_lines = [(np.log(float(s.strip())), l.strip()) for (s, l) in data_lines]
    targets = [s for (s, l) in data_lines if l.upper() == "FALSE"]
    decoys = [s for (s, l) in data_lines if l.upper() == "TRUE"]
    invalid = [(s, l) for (s, l) in data_lines if l.upper() not in ("FALSE", "TRUE")]
else:
    raise NotImplementedError("this kind of header %r not supported yet" % header)


print
print "COUNTS"
print
print "targets           =", len(targets)
print "decoys            =", len(decoys)
print "targests + decoys =", len(targets) + len(decoys)
print "data lines        =", len(data_lines)
print "invalid lines     =", len(invalid)

df, __, __ = stats.get_error_stat_from_null(targets, decoys, 0.4)

errt= stats.summary_err_table(df, (0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, .08, .09, .1,
                                   .15, .2, .3, .4, .5))

errt["exp_cutoff"] = np.exp(errt.cutoff.values.astype(float))

print
print "STATS"
print
print errt.to_string()  # avoid line break
