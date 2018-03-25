import pandas as pd
import numpy as np
import random
import sys
import os

from .optimized import find_top_ranked, rank
from .config import CONFIG
from .std_logger import logging

try:
    profile
except NameError:
    def profile(fun):
        return fun


def isSQLite3(filename):
    # https://stackoverflow.com/questions/12932607/how-to-check-with-python-and-sqlite3-if-one-sqlite-database-file-exists
    from os.path import isfile, getsize

    if not isfile(filename):
        return False
    if getsize(filename) < 100: # SQLite database file header is 100 bytes
        return False

    with open(filename, 'rb') as fd:
        header = fd.read(100)

    if 'SQLite format 3' in str(header):
        return True
    else:
        return False


def check_sqlite_table(con, table):
    table_present = False
    c = con.cursor()
    c.execute('SELECT count(name) FROM sqlite_master WHERE type="table" AND name="' + table + '"')
    if c.fetchone()[0] == 1:
        table_present = True
    else:
        table_present = False
    c.fetchall()

    return(table_present)


def check_for_unique_blocks(tg_ids):
    seen = set()
    last_tg_id = None
    for tg_id in tg_ids:
        if last_tg_id != tg_id:
            last_tg_id = tg_id
            if tg_id in seen:
                return False
            seen.add(tg_id)
    return True


@profile
def cleanup_and_check(df):
    score_columns = ["main_score"] + [c for c in df.columns if c.startswith("var_")]
    # this is fast but not easy to read
    # find peak groups with in valid scores:
    sub_df = df.loc[:, score_columns]
    flags = ~pd.isnull(sub_df)
    valid_rows = flags.all(axis=1)

    df_cleaned = df.loc[valid_rows, :]

    # decoy / non decoy sub tables
    df_decoy = df_cleaned[df_cleaned["is_decoy"].eq(True)]
    df_target = df_cleaned[df_cleaned["is_decoy"].eq(False)]

    # groups
    decoy_groups = set(df_decoy["tg_id"])
    target_groups = set(df_target["tg_id"])

    n_decoy = len(decoy_groups)
    n_target = len(target_groups)

    msg = "Data set contains %d decoy and %d target groups." % (n_decoy, n_target)
    logging.info(msg)
    if n_decoy < 10 or n_target < 10:
        sys.exit("Error: At least 10 decoy groups and 10 target groups are required.")

    return df_cleaned


def prepare_data_table(table, tg_id_name="transition_group_id",
                       decoy_name="decoy",
                       main_score_name=None,
                       score_columns=None,
                       ):
    N = len(table)
    if not N:
        sys.exit("Error: Empty input file supplied.")
    header = table.columns.values
    if score_columns is not None:
        missing = set(score_columns) - set(header)
        if missing:
            missing_txt = ", ".join(["'%s'" % m for m in missing])
            msg = "Error: Column(s) %s missing in input file to apply scorer." % missing_txt
            sys.exit(msg)

    assert tg_id_name in header, "Error: Column %s is not in input file(s)." % tg_id_name
    assert decoy_name in header, "Error: Column %s is not in input file(s)." % decoy_name

    if score_columns is not None:
        # we assume there is exactly one main_score in score_columns as we checked that in
        # the run which persisted the classifier:
        var_column_names = [c for c in score_columns if c.startswith("var_")]
        main_score_name = [c for c in score_columns if c.startswith("main_")][0]
    else:
        if main_score_name is not None:
            assert main_score_name in header, "Error: Column %s is not in input file(s)." % main_score_name

        # if no main_score_name provided, look for unique column with name
        # starting with "main_":
        else:
            main_columns = set(c for c in header if c.startswith("main_"))
            if not main_columns:
                sys.exit("Error: No column \"main_*\" is in input file(s).")
            if len(main_columns) > 1:
                sys.exit("Error: Multiple columns with name \"main_*\" are in input file(s).")
            main_score_name = main_columns.pop()

        # get all other score columns, name beginning with "var_"
        var_column_names = tuple(h for h in header if h.startswith("var_"))

        if not var_column_names:
            raise Exception("Error: No column \"var_*\" is in input file(s).")

    # collect needed data:
    empty_col = [0] * N
    empty_none_col = [None] * N

    tg_ids = table[tg_id_name]

    if not check_for_unique_blocks(tg_ids):
        sys.exit("Error: " + tg_id_name + " values do not form unique blocks in input file(s).")

    tg_map = dict()
    for i, tg_id in enumerate(tg_ids.unique()):
        tg_map[tg_id] = i
    tg_num_ids = [tg_map[tg_id] for tg_id in tg_ids]

    data = dict(tg_id=tg_ids.values,
                tg_num_id=tg_num_ids,
                is_decoy=table[decoy_name].values.astype(bool),
                is_top_peak=empty_col,
                is_train=empty_none_col,
                main_score=table[main_score_name].values,
                )

    column_names = ["tg_id", "tg_num_id", "is_decoy", "is_top_peak", "is_train", "main_score"]
    used_var_column_names = []
    for i, v in enumerate(var_column_names):
        col_name = "var_%d" % i
        col_data = table[v]
        if pd.isnull(col_data).all():
            logging.warn("Warning: Column %s contains only invalid/missing values. Column will be dropped." % v)
            continue
        else:
            used_var_column_names.append(v)

        data[col_name] = col_data
        column_names.append(col_name)

    data["classifier_score"] = empty_col
    column_names.append("classifier_score")

    # build data frame:
    df = pd.DataFrame(data, columns=column_names)

    all_score_columns = (main_score_name,) + tuple(used_var_column_names)
    df = cleanup_and_check(df)
    return df, all_score_columns


class Experiment(object):

    @profile
    def __init__(self, df):
        self.df = df

    def log_summary(self):
        logging.info("Summary of input data:")
        logging.info("   %d peak groups" % len(self.df))
        logging.info("   %d group ids" % len(self.df.tg_id.unique()))
        logging.info("   %d scores including main score" % (len(self.df.columns.values) - 6))

    def __getitem__(self, *args):
        return self.df.__getitem__(*args)

    def __setitem__(self, *args):
        return self.df.__setitem__(*args)

    def __setattr__(self, name, value):
        if name not in ["df", ]:
            sys.exit("Error: Use '[...]' syntax to set input file columns.")
        object.__setattr__(self, name, value)

    def set_and_rerank(self, col_name, scores):
        self.df.loc[:, col_name] = scores
        self.rank_by(col_name)

    def rank_by(self, score_col_name):
        flags = find_top_ranked(self.df.tg_num_id.values, self.df[score_col_name].values.astype(np.float32, copy=False))
        self.df.is_top_peak = flags

    def get_top_test_peaks(self):
        df = self.df
        return Experiment(df[(df.is_train == False) & (df.is_top_peak == True)])

    def get_decoy_peaks(self):
        return Experiment(self.df[self.df.is_decoy == True])

    def get_target_peaks(self):
        return Experiment(self.df[self.df.is_decoy == False])

    def get_top_decoy_peaks(self):
        ix_top = self.df.is_top_peak == True
        return Experiment(self.df[(self.df.is_decoy == True) & ix_top])

    def get_top_target_peaks(self):
        ix_top = self.df.is_top_peak == True
        return Experiment(self.df[(self.df.is_decoy == False) & ix_top])

    def get_feature_matrix(self, use_main_score):
        min_col = 5 if use_main_score else 6
        return self.df.iloc[:, min_col:-1].values

    def filter_(self, idx):
        return Experiment(self.df[idx])

    @profile
    def add_peak_group_rank(self):
        ids = self.df.tg_num_id.values
        scores = self.df.d_score.values
        peak_group_ranks = rank(ids, scores)
        self.df["peak_group_rank"] = peak_group_ranks

    @profile
    def split_for_xval(self, fraction, is_test):
        df = self.df
        decoy_ids = df[df.is_decoy == True].tg_id.unique()
        target_ids = df[df.is_decoy == False].tg_id.unique()

        if not is_test:
            random.shuffle(decoy_ids)
            random.shuffle(target_ids)
        else:
            decoy_ids = sorted(decoy_ids)
            target_ids = sorted(target_ids)

        decoy_ids = decoy_ids[:int(len(decoy_ids) * fraction) + 1]
        target_ids = target_ids[:int(len(target_ids) * fraction) + 1]
        learn_ids = np.concatenate((decoy_ids, target_ids))
        ix_learn = df.tg_id.isin(learn_ids)
        df.is_train.loc[ix_learn] = True
        df.is_train.loc[~ix_learn] = False

    def get_train_peaks(self):
        df = self.df[self.df.is_train == True]
        return Experiment(df)


# Filter a sqMass chromatogram file by given input labels
def filterChromByLabels(infile, outfile, labels):
    import sqlite3
    conn = sqlite3.connect(infile)
    c = conn.cursor()

    labels = [ "'" + l + "'" for l in labels]
    labels_stmt = get_ids_stmt(labels)

    stmt = "SELECT ID FROM CHROMATOGRAM WHERE NATIVE_ID IN %s" % labels_stmt
    keep_ids = [i[0] for i in list(c.execute(stmt))]
    print("Keep %s chromatograms" % len(keep_ids) )

    nr_chrom = list(c.execute("SELECT COUNT(*) FROM CHROMATOGRAM"))[0][0]
    nr_spec = list(c.execute("SELECT COUNT(*) FROM SPECTRUM"))[0][0]

    assert(nr_chrom > 0)
    assert(nr_spec == 0)

    copyDatabase(c, conn, outfile, keep_ids)


def copy_table(c, conn, keep_ids, tbl, id_col):
    stmt = "CREATE TABLE other.%s AS SELECT * FROM %s WHERE %s IN " % (tbl, tbl, id_col)
    stmt += get_ids_stmt(keep_ids) + ";"
    c.execute(stmt)
    conn.commit()


def copyDatabase(c, conn, outfile, keep_ids):
    c.execute("ATTACH DATABASE '%s' AS other;" % outfile)

    # Tables: 
    #  - DATA
    #  - SPECTRUM
    #  - RUN
    #  - RUN_EXTRA
    #  - CHROMATOGRAM
    #  - PRODUCT
    #  - PRECURSOR

    # copy over data that matches the selected ids
    copy_table(c, conn, keep_ids, "PRECURSOR", "CHROMATOGRAM_ID")
    copy_table(c, conn, keep_ids, "PRODUCT", "CHROMATOGRAM_ID")
    copy_table(c, conn, keep_ids, "DATA", "CHROMATOGRAM_ID")
    copy_table(c, conn, keep_ids, "CHROMATOGRAM", "ID")

    # copy over data and create indices
    c.execute("CREATE TABLE other.RUN AS SELECT * FROM RUN");
    c.execute("CREATE TABLE other.SPECTRUM AS SELECT * FROM SPECTRUM");
    c.execute("CREATE TABLE other.RUN_EXTRA AS SELECT * FROM RUN_EXTRA");

    c.execute("CREATE INDEX other.data_chr_idx ON DATA(CHROMATOGRAM_ID);")
    c.execute("CREATE INDEX other.data_sp_idx ON DATA(SPECTRUM_ID);")
    c.execute("CREATE INDEX other.spec_rt_idx ON SPECTRUM(RETENTION_TIME);")
    c.execute("CREATE INDEX other.spec_mslevel ON SPECTRUM(MSLEVEL);")
    c.execute("CREATE INDEX other.spec_run ON SPECTRUM(RUN_ID);")
    c.execute("CREATE INDEX other.chrom_run ON CHROMATOGRAM(RUN_ID);")

    conn.commit()


def get_ids_stmt(keep_ids):
    ids_stmt = "("
    for myid in keep_ids:
        ids_stmt += str(myid) + ","
    ids_stmt = ids_stmt[:-1]
    ids_stmt += ")"
    return ids_stmt 

