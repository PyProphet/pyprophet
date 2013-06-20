#encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

import pandas as pd
import numpy as np
from optimized import find_top_ranked

try:
    profile
except:
    def profile(x):
        return x


@profile
def prepare_data_table(table, tg_id_name = "transition_group_id",
                              decoy_name = "decoy",
                              main_score_name = None,
                              **extra_args_to_dev_null
                              ):

    column_names = table.columns.values

    # check if given column names appear in table:
    assert tg_id_name in column_names, "colum %s not in table" % tg_id_name
    assert decoy_name in column_names, "colum %s not in table" % decoy_name
    if main_score_name is not None:
        assert main_score_name in column_names,\
                                      "colum %s not in table" % main_score_name

    # if no main_score_name provided, look for unique column with name
    # starting with "main_":
    else:
        main_columns = [c for c in column_names if c.startswith("main_")]
        if not main_columns:
            raise Exception("no column with main_* in table")
        if len(main_columns)>1:
            raise Exception("multiple columns with name main_* in table")
        main_score_name = main_columns[0]

    # get all other score columns, name beginning with "var_"
    var_columns = [c for c in column_names if c.startswith("var_")]
    if not var_columns:
        raise Exception("no column with name var_* in table")

    # collect needed data:
    column_names = """tg_id tg_num_id is_decoy is_top_peak is_train
                      main_score""".split()
    N = len(table)
    empty_col = [0] * N
    empty_none_col = [None] * N

    tg_ids = table[tg_id_name]

    tg_map = dict()
    for i, tg_id in enumerate(tg_ids.unique()):
        tg_map[tg_id] = i
    tg_num_ids = [ tg_map[tg_id] for tg_id in tg_ids]

    data = dict(tg_id = tg_ids.values,
                tg_num_id = tg_num_ids,
                is_decoy = table[decoy_name].values.astype(bool),
                is_top_peak = empty_col,
                is_train = empty_none_col,
                main_score = table[main_score_name].values,
                )

    for i, v in enumerate(var_columns):
        col_name = "var_%d" % i
        data[col_name] = table[v].values
        column_names.append(col_name)

    data["classifier_score"] = empty_col
    column_names.append("classifier_score")

    # build data frame:
    df = pd.DataFrame(data, columns=column_names)

    # for each transition group: enumerate peaks in this group, and
    # add peak_rank where increasing rank corresponds to decreasing main
    # score. peak_rank == 0 is peak with max main score
    return df


class Experiment(object):

    def __init__(self, df):
        self.df = df


    def print_summary(self):
        print len(self.df), 'lines in input data'
        print len(self.df.tg_id.unique()), 'transition groups in input data'
        print len(self.df.columns.values) - 6, 'scores including main score'


    def get_top_decoy_peaks(self):
        df = self.df[(self.df.is_decoy == False) & (self.df.is_top_peak == True)]
        return Experiment(df)


    def __getitem__(self, *args):
        return self.df.__getitem__(*args)

    def __setitem__(self, *args):
        return self.df.__setitem__(*args)

    def __setattr__(self, name, value):
        if name not in ["df", ]:
            raise Exception("for setting table columns use '[...]' syntax")
        object.__setattr__(self, name, value)

    def set_and_rerank(self, col_name, scores):
        self.df[col_name] = scores
        self.rank_by(col_name)


    def rank_by(self, score_col_name):
        flags = find_top_ranked(self.df.tg_num_id.values, self.df[score_col_name].values)
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


    def split_for_xval(self, fraction):
        df = self.df
        decoy_ids = df[df.is_decoy == True].tg_id.unique()
        target_ids = df[df.is_decoy == False].tg_id.unique()
        decoy_ids = decoy_ids[:int(len(decoy_ids) * fraction) + 1]
        target_ids = target_ids[:int(len(target_ids) * fraction) + 1]
        learn_ids = np.concatenate((decoy_ids, target_ids))
        ix_learn = df.tg_id.isin(learn_ids)
        df.is_train[ix_learn] = True
        df.is_train[~ix_learn] = False

    split_for_xval = profile(split_for_xval)

    def get_train_peaks(self):
        df = self.df[self.df.is_train == True]
        return Experiment(df)


