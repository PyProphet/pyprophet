import pandas as pd

def prepare_data_table(table, tgid_name = "transition_group_id",
                              decoy_name = "decoy",
                              main_score_name = None,
                              ):

    column_names = table.columns.values
    # check if given column names appear in table:
    assert tgid_name in column_names, "colum %s not in table" % tgid_name
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
    column_names = "transition_group_id decoy main_score".split()
    data = dict(transition_group_id = table[tgid_name].values,
                decoy = table[decoy_name].values,
                main_score = table[main_score_name].values
                )
    for i, v in enumerate(var_columns):
        col_name = "var_%d" % i
        data[col_name] = table[v].values
        column_names.append(col_name)

    # build data frame:
    df_orig = pd.DataFrame(data, columns=column_names)
    df_orig["row_id"] = df_orig.index    # for reference with full table

    # for each transition group: enumerate peaks in this group, and
    # add peak_rank where increasing rank corresponds to decreasing main
    # score. peak_rank == 0 is peak with max main score
    collected = []
    for x, subdf in df_orig.groupby("transition_group_id"):
        # drop means: forget existing index
        subdf.reset_index(inplace=True, drop=True)
        subdf["peak_id"] = subdf.index
        subdf.sort("main_score", ascending=False, inplace=True)
        subdf.reset_index(inplace=True, drop=True)
        subdf["peak_rank"] = subdf.index
        collected.append(subdf)

    return var_columns, df_orig, pd.concat(collected)
