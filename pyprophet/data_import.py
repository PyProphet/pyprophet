import pandas as pd

try:
    profile
except:
    profile = lambda x: x

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
    column_names = """tg_id is_decoy main_score peak_rank
                        is_train global_peak_id tg_peak_id""".split()
    N = len(table)
    empty_col = [0] * N
    empty_none_col = [None] * N

    data = dict(tg_id = table[tg_id_name].values,
                is_decoy = table[decoy_name].values.astype(bool),
                main_score = table[main_score_name].values,
                peak_rank = empty_col,
                is_train = empty_none_col,
                global_peak_id = empty_none_col,
                tg_peak_id = empty_none_col,
                )
    for i, v in enumerate(var_columns):
        col_name = "var_%d" % i
        col_name = v
        data[col_name] = table[v].values
        column_names.append(col_name)

    # build data frame:
    df_orig = pd.DataFrame(data, columns=column_names)
    df_orig["global_peak_id"] = df_orig.index  # for reference with full table

    df_orig.sort(("tg_id", "main_score"), ascending=(True, False),
            inplace=True)

    # for each transition group: enumerate peaks in this group, and
    # add peak_rank where increasing rank corresponds to decreasing main
    # score. peak_rank == 0 is peak with max main score
    return df_orig
