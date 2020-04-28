import pandas as pd
import numpy as np
import random
import click
import sys
import os
import multiprocessing

from .optimized import find_top_ranked, rank

try:
    profile
except NameError:
    def profile(fun):
        return fun

def use_metabolomics_scores():
    return [
            "var_ms1_isotope_overlap_score",
            "var_ms1_xcorr_coelution_contrast",
            "var_ms1_massdev_score",
            "var_ms1_xcorr_coelution",  
            "var_ms1_isotope_correlation_score",
            "var_isotope_overlap_score",
            "var_isotope_correlation_score",
            "var_intensity_score",
            "var_massdev_score",
            "var_library_corr",
            "var_norm_rt_score"
            ]

# Parameter transformation functions
def transform_pi0_lambda(ctx, param, value):
    if value[1] == 0 and value[2] == 0:
        pi0_lambda = value[0]
    elif 0 <= value[0] < 1 and value[0] <= value[1] <= 1 and 0 < value[2] < 1:
        pi0_lambda = np.arange(value[0], value[1], value[2])
    else:
        raise click.ClickException('Wrong input values for pi0_lambda. pi0_lambda must be within [0,1).')
    return(pi0_lambda)


def transform_threads(ctx, param, value):
    if value == -1:
        value = multiprocessing.cpu_count()
    return(value)


def transform_subsample_ratio(ctx, param, value):
    if value < 0 or value > 1:
      raise click.ClickException('Wrong input values for subsample_ratio. subsample_ratio must be within [0,1].')
    return(value)


def is_sqlite_file(filename):
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
    c.execute('SELECT count(name) FROM sqlite_master WHERE type="table" AND name="%s"' % table)
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

    click.echo("Info: Data set contains %d decoy and %d target groups." % (n_decoy, n_target))
    if n_decoy < 10 or n_target < 10:
        raise click.ClickException("At least 10 decoy groups and 10 target groups are required.")

    return df_cleaned


def prepare_data_table(table,
                       ss_score_filter,
                       tg_id_name="transition_group_id",
                       decoy_name="decoy",
                       main_score_name=None,
                       score_columns=None,
                       ):
    N = len(table)
    if not N:
        raise click.ClickException("Empty input file supplied.")
    header = table.columns.values
    if score_columns is not None:
        missing = set(score_columns) - set(header)
        if missing:
            missing_txt = ", ".join(["'%s'" % m for m in missing])
            raise click.ClickException("Column(s) %s missing in input file to apply scorer." % missing_txt)

    assert tg_id_name in header, "Column %s is not in input file(s)." % tg_id_name
    assert decoy_name in header, "Column %s is not in input file(s)." % decoy_name

    if score_columns is not None:
        # we assume there is exactly one main_score in score_columns as we checked that in
        # the run which persisted the classifier:
        var_columns_available = [c for c in score_columns if c.startswith("var_")]
        main_score_name = [c for c in score_columns if c.startswith("main_")][0]
    else:
        if main_score_name is not None:
            assert main_score_name in header, "Column %s is not in input file(s)." % main_score_name

        # if no main_score_name provided, look for unique column with name
        # starting with "main_":
        else:
            main_columns = set(c for c in header if c.startswith("main_"))
            if not main_columns:
                raise click.ClickException("No column \"main_*\" is in input file(s).")
            if len(main_columns) > 1:
                raise click.ClickException("Multiple columns with name \"main_*\" are in input file(s).")
            main_score_name = main_columns.pop()

        # get all other score columns, name beginning with "var_"
        var_columns_available = tuple(h for h in header if h.startswith("var_"))

        if not var_columns_available:
            raise Exception("No column \"var_*\" is in input file(s).")

    # filter columns based on input score names (unless default is set)
    var_column_names = []
    if ss_score_filter != '':
        input_scores = ss_score_filter.split(sep=',')

        # use metabolomics scores and allows to add scores in addition specified by ss_score_filter
        if 'metabolomics' in ss_score_filter:
            input_scores.remove('metabolomics')
            metabolomics_scores = use_metabolomics_scores()
            input_scores += metabolomics_scores
            input_scores = list(set(input_scores))

        # remove main score from filter list if it was specified twice (main, var)
        if main_score_name.strip("main_") in input_scores:
            input_scores.remove(main_score_name.strip("main_"))

        var_columns_available_s = set(var_columns_available)
        score_not_found = []

        for score in input_scores:
            if score in var_columns_available_s:
                var_column_names.append(score)
            else:
                score_not_found.append(score)
        if score_not_found:
            not_found = ", ".join(["'%s'" % m for m in score_not_found])
            raise click.ClickException("Column(s) %s not found in input file. Please check your score filter (--ss_score_filter)" % not_found)

    else:
        var_column_names = var_columns_available

    # collect needed data:
    empty_col = [0] * N
    empty_none_col = [None] * N

    tg_ids = table[tg_id_name]

    if not check_for_unique_blocks(tg_ids):
        raise click.ClickException("" + tg_id_name + " values do not form unique blocks in input file(s).")

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
            click.echo("Warning: Column %s contains only invalid/missing values. Column will be dropped." % v)
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
        self.df = df.copy()

    def log_summary(self):
        click.echo("Info: Summary of input data:")
        click.echo("Info: %d peak groups" % len(self.df))
        click.echo("Info: %d group ids" % len(self.df.tg_id.unique()))
        click.echo("Info: %d scores including main score" % (len(self.df.columns.values) - 6))

    def __getitem__(self, *args):
        return self.df.__getitem__(*args)

    def __setitem__(self, *args):
        return self.df.__setitem__(*args)

    def __setattr__(self, name, value):
        if name not in ["df", ]:
            raise click.ClickException("Use '[...]' syntax to set input file columns.")
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
        df.loc[ix_learn,'is_train'] = True
        df.loc[~ix_learn,'is_train'] = False

    def get_train_peaks(self):
        df = self.df[self.df.is_train == True]
        return Experiment(df)

