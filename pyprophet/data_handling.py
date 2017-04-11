# encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

import random

import pandas as pd
import numpy as np
from optimized import find_top_ranked, rank
from config import CONFIG

try:
    profile
except NameError:
    def profile(fun):
        return fun

from std_logger import logging


def setup_csv_dtypes(columns):
    dtype = {}
    for col_name in columns:
        if col_name.startswith("main_") or col_name.startswith("var_"):
            dtype[col_name] = np.float32
    return dtype


def read_csv(path, delim):
    header = pd.read_csv(path, delim, nrows=1).columns
    dtype = setup_csv_dtypes(header)
    return pd.read_csv(path, delim, na_values=["NA", "NaN", "infinite"], engine="c", dtype=dtype)


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

    msg = "data set contains %d decoy and %d target transition groups" % (n_decoy, n_target)
    logging.info(msg)
    if n_decoy < 10 or n_target < 10:
        logging.error("need at least 10 decoy groups ans 10 non decoy groups")
        raise Exception("need at least 10 decoy groups ans 10 non decoy groups. %s" % msg)

    return df_cleaned


def prepare_data_table(table, tg_id_name="transition_group_id",
                       decoy_name="decoy",
                       main_score_name=None,
                       score_columns=None,
                       ):
    N = len(table)
    if not N:
        raise Exception("got empty input file")
    header = table.columns.values
    if score_columns is not None:
        missing = set(score_columns) - set(header)
        if missing:
            missing_txt = ", ".join(["'%s'" % m for m in missing])
            msg = "column(s) %s missing in input file for applying existing scorer" % missing_txt
            raise Exception(msg)

    assert tg_id_name in header, "colum %s not in table" % tg_id_name
    assert decoy_name in header, "colum %s not in table" % decoy_name

    if score_columns is not None:
        # we assume there is exactly one main_score in score_columns as we checked that in
        # the run which persisted the classifier:
        var_column_names = [c for c in score_columns if c.startswith("var_")]
        main_score_name = [c for c in score_columns if c.startswith("main_")][0]
    else:
        if main_score_name is not None:
            assert main_score_name in header, "colum %s not in table" % main_score_name

        # if no main_score_name provided, look for unique column with name
        # starting with "main_":
        else:
            main_columns = set(c for c in header if c.startswith("main_"))
            if not main_columns:
                raise Exception("no column with main_* in table(s)")
            if len(main_columns) > 1:
                raise Exception("multiple columns with name main_* in table(s)")
            main_score_name = main_columns.pop()

        # get all other score columns, name beginning with "var_"
        var_column_names = tuple(h for h in header if h.startswith("var_"))

        if not var_column_names:
            raise Exception("no column with name var_* in table(s)")

    # collect needed data:
    empty_col = [0] * N
    empty_none_col = [None] * N

    tg_ids = table[tg_id_name]

    if not check_for_unique_blocks(tg_ids):
        raise Exception("transition group ids do not form unique blocks in data file")

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

    ignore_invalid_scores = CONFIG["ignore.invalid_score_columns"]
    column_names = ["tg_id", "tg_num_id", "is_decoy", "is_top_peak", "is_train", "main_score"]
    for i, v in enumerate(var_column_names):
        col_name = "var_%d" % i
        col_data = table[v]
        if pd.isnull(col_data).all():
            msg = "column %s contains only invalid/missing values" % v
            if ignore_invalid_scores:
                logging.warn("%s. pyprophet skips this.")
                continue
            raise Exception("%s. you may use --ignore.invalid_score_columns")
        data[col_name] = col_data
        column_names.append(col_name)

    data["classifier_score"] = empty_col
    column_names.append("classifier_score")

    # build data frame:
    df = pd.DataFrame(data, columns=column_names)

    all_score_columns = (main_score_name,) + tuple(var_column_names)
    df = cleanup_and_check(df)
    return df, all_score_columns


def check_header(path, delim, check_cols):
    header = pd.read_csv(path, delim, nrows=1).columns
    if check_cols:
        missing = set(check_cols) - set(header)
        if missing:
            missing = sorted(missing)
            raise Exception("columns %s are missing in input file" % ", ".join(missing))

    if not any(name.startswith("main_") for name in header):
        raise Exception("no column starting with 'main_' found in input file")

    if not any(name.startswith("var_") for name in header):
        raise Exception("no column starting with 'var_' found in input file")

    return header


@profile
def prepare_data_tables(tables, tg_id_name="transition_group_id",
                        decoy_name="decoy",
                        main_score_name=None,
                        score_columns=None,
                        ):

    all_score_columns = set()
    dfs = []
    for table in tables:
        df, score_columns = prepare_data_table(table, tg_id_name, decoy_name, main_score_name,
                                               score_columns)
        all_score_columns.add(tuple(score_columns))
        dfs.append(df)

    if len(all_score_columns) > 1:
        raise Exception("score columns in input tables are not consistent (order and/or naming)")

    return dfs, all_score_columns.pop()


def sample_data_tables(pathes, delim,  sampling_rate=0.1, tg_id_name="transition_group_id",
                       decoy_name="decoy"):
    tg_target_ids = set()
    tg_decoy_ids = set()
    for path in pathes:
        for chunk in pd.read_csv(path, delim, iterator=True, chunksize=100000,
                                 usecols=[tg_id_name, decoy_name]):
            ids = chunk[chunk[decoy_name].eq(False)][tg_id_name].values
            tg_target_ids.update(ids)
            ids = chunk[chunk[decoy_name].eq(True)][tg_id_name].values
            # remove "DECOY_" in the beginnings:
            ids = [id_.partition("DECOY_")[2] for id_ in ids]
            tg_decoy_ids.update(ids)

    assert len(pathes) > 0, "no input files !?"
    header = pd.read_csv(pathes[0], delim, nrows=1).columns

    tg_ids = tg_target_ids.intersection(tg_decoy_ids)
    if not len(tg_ids):
        raise Exception("did not find an intersection of target and decoy ids")

    # subsample from targets
    if sampling_rate < 1.0:
        tg_ids = random.sample(tg_ids, int(sampling_rate * len(tg_ids)))
    else:
        tg_ids = list(tg_ids)
    # add corresponding decoys
    tg_ids += ["DECOY_" + id_ for id_ in tg_ids]
    # convert to set for faster lookup below:
    tg_ids = set(tg_ids)

    dtype = setup_csv_dtypes(header)
    chunks = []
    for path in pathes:
        for chunk in pd.read_csv(path, delim, iterator=True, na_values=["NA", "NaN", "infinite"],
                                 chunksize=100000, dtype=dtype):
            chunk = chunk[chunk[tg_id_name].isin(tg_ids)]
            chunks.append(chunk)

    return prepare_data_tables(chunks)


class Experiment(object):

    @profile
    def __init__(self, df):
        self.df = df

    def log_summary(self):
        logging.info("summary input file:")
        logging.info("   %d lines" % len(self.df))
        logging.info("   %d transition groups" % len(self.df.tg_id.unique()))
        logging.info("   %d scores including main score" % (len(self.df.columns.values) - 6))

    def __getitem__(self, *args):
        return self.df.__getitem__(*args)

    def __setitem__(self, *args):
        return self.df.__setitem__(*args)

    def __setattr__(self, name, value):
        if name not in ["df", ]:
            raise Exception("for setting table columns use '[...]' syntax")
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
