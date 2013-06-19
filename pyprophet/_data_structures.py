import pandas as pd
import numpy as np
import random

try:
    profile
except:
    profile = lambda x: x

profile = lambda x: x
from numba import jit, i8, f8



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

@profile
def read_csv(path, sep=None, **extra_args_to_dev_null):
    table = pd.read_csv(path, sep)
    return table

class Experiment(object):

    @staticmethod
    def from_csv(path, **kw):
        raw = read_csv(path, **kw)
        return Experiment.from_data_frame(raw, **kw)

    @staticmethod
    def from_data_frame(df, **kw):
        return Experiment(prepare_data_table(df), **kw)

    @profile
    def __init__(self, df, **kw_to_dev_null):

        var_columns = [c for c in df.columns.values if c.startswith("var_")]
        self.score_columns = var_columns + [ "main_score"]

        # LATER OBSOLETE !
        df.sort("tg_id", ascending=True, inplace=True)

        flags = find_top_ranked(df.tg_num_id.values, df.main_score.values)
        df.is_top_peak = flags
        self.df = df


    @profile
    def split(self, frac):
        decoy_ids = self.df[self.df.is_decoy==True].tg_num_id.unique()
        target_ids = self.df[self.df.is_decoy==False].tg_num_id.unique()

        #random.shuffle(decoy_ids)
        #random.shuffle(target_ids)

        decoy_ids = decoy_ids[:int(len(decoy_ids) * frac)+1]
        target_ids = target_ids[:int(len(target_ids) * frac)+1]

        learn_ids = np.concatenate((decoy_ids,target_ids))
        ix_learn = self.df.tg_num_id.isin(learn_ids)

        self.df.is_train[ix_learn] = True
        self.df.is_train[~ix_learn] = False
