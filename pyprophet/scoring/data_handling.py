"""
This module provides utilities for handling and processing data in PyProphet for semi-supervised scoring

It includes functions for cleaning and validating data, preparing data tables,
and managing feature scaling and ranking. Additionally, it defines the `Experiment`
class, which encapsulates data operations for peak groups, decoys, and targets.

Classes:
    - Experiment: Encapsulates data operations for peak groups, decoys, and targets.

Functions:
    - use_metabolomics_scores: Returns a list of metabolomics-specific score columns.
    - check_for_unique_blocks: Checks if transition group IDs form unique blocks.
    - cleanup_and_check: Cleans up the input DataFrame and validates its structure.
    - prepare_data_table: Prepares the input data table for scoring and analysis.
    - update_chosen_main_score_in_table: Updates the main score column in the feature table.
"""

import random

import click
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

from ..stats import mean_and_std_dev
from .optimized import find_top_ranked, rank

try:
    profile
except NameError:

    def profile(fun):
        return fun


def use_metabolomics_scores():
    """
    Returns a list of metabolomics-specific score columns.

    These scores are selected for their low cross-correlation in metabolomics scoring.
    """
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
        "var_norm_rt_score",
    ]


def check_for_unique_blocks(tg_ids):
    """
    Checks if group IDs form unique blocks.

    Args:
        tg_ids (iterable): group IDs.

    Returns:
        bool: True if the IDs form unique blocks, False otherwise.
    """
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
    """
    Cleans up the input DataFrame and validates its structure.

    Args:
        df (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: Cleaned and validated data.
    """
    score_columns = ["main_score"] + [c for c in df.columns if c.startswith("var_")]
    # this is fast but not easy to read
    # find peak groups with in valid scores:
    sub_df = df.loc[:, score_columns]
    flags = ~pd.isnull(sub_df)
    valid_rows = flags.all(axis=1)
    logger.trace(f"{valid_rows.sum()} valid rows out of {len(df)}")
    df_cleaned = df.loc[valid_rows, :]

    # decoy / non decoy sub tables
    df_decoy = df_cleaned[df_cleaned["is_decoy"].eq(True)]
    df_target = df_cleaned[df_cleaned["is_decoy"].eq(False)]

    # groups
    decoy_groups = set(df_decoy["tg_id"])
    target_groups = set(df_target["tg_id"])

    n_decoy = len(decoy_groups)
    n_target = len(target_groups)

    logger.info(
        "Data set contains %d decoy and %d target groups." % (n_decoy, n_target)
    )
    if n_decoy < 10 or n_target < 10:
        print(sub_df)
        raise click.ClickException(
            "At least 10 decoy groups and 10 target groups are required."
        )

    return df_cleaned


def prepare_data_table(
    table,
    ss_score_filter,
    tg_id_name="transition_group_id",
    decoy_name="decoy",
    main_score_name=None,
    score_columns=None,
    level=None,
):
    """
    Prepares the input data table for scoring and analysis.

    Args:
        table (pd.DataFrame): Input data table.
        ss_score_filter (str): Semi-supervised score filter.
        tg_id_name (str): Name of the transition group ID column.
        decoy_name (str): Name of the decoy column.
        main_score_name (str, optional): Name of the main score column.
        score_columns (list, optional): List of score columns.
        level (str, optional): Analysis level (e.g., "alignment").

    Returns:
        tuple: Prepared DataFrame, list of all score columns, and used variable column IDs.
    """
    N = len(table)
    if not N:
        raise click.ClickException("Empty input file supplied.")
    header = table.columns.values
    if score_columns is not None:
        missing = set(score_columns) - set(header)
        if missing:
            missing_txt = ", ".join(["'%s'" % m for m in missing])
            raise click.ClickException(
                "Column(s) %s missing in input file to apply scorer." % missing_txt
            )

    assert tg_id_name in header, "Column %s is not in input file(s)." % tg_id_name
    assert decoy_name in header, "Column %s is not in input file(s)." % decoy_name

    if score_columns is not None:
        # we assume there is exactly one main_score in score_columns as we checked that in
        # the run which persisted the classifier:
        var_columns_available = [c for c in score_columns if c.startswith("var_")]
        main_score_name = [c for c in score_columns if c.startswith("main_")][0]
    else:
        if main_score_name is not None:
            assert main_score_name in header, (
                "Column %s is not in input file(s)." % main_score_name
            )

        # if no main_score_name provided, look for unique column with name
        # starting with "main_":
        else:
            main_columns = set(c for c in header if c.startswith("main_"))
            if not main_columns:
                raise click.ClickException('No column "main_*" is in input file(s).')
            if len(main_columns) > 1:
                raise click.ClickException(
                    'Multiple columns with name "main_*" are in input file(s).'
                )
            main_score_name = main_columns.pop()

        # get all other score columns, name beginning with "var_"
        var_columns_available = tuple(h for h in header if h.startswith("var_"))

        if not var_columns_available:
            raise Exception('No column "var_*" is in input file(s).')

    # filter columns based on input score names (unless default is set)
    var_column_names = []
    if ss_score_filter != "":
        input_scores = ss_score_filter.split(sep=",")

        # use metabolomics scores and allows to add scores in addition specified by ss_score_filter
        if "metabolomics" in ss_score_filter:
            input_scores.remove("metabolomics")
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
            raise click.ClickException(
                "Column(s) %s not found in input file. Please check your score filter (--ss_score_filter)"
                % not_found
            )

    else:
        var_column_names = var_columns_available

    # collect needed data:
    empty_col = [0] * N
    empty_none_col = [None] * N

    tg_ids = table[tg_id_name]

    if not check_for_unique_blocks(tg_ids) and level != "alignment":
        raise click.ClickException(
            "" + tg_id_name + " values do not form unique blocks in input file(s)."
        )

    tg_map = dict()
    for i, tg_id in enumerate(tg_ids.unique()):
        tg_map[tg_id] = i
    tg_num_ids = [tg_map[tg_id] for tg_id in tg_ids]

    data = dict(
        tg_id=tg_ids.values,
        tg_num_id=tg_num_ids,
        is_decoy=table[decoy_name].values.astype(bool),
        is_top_peak=empty_col,
        is_train=empty_none_col,
        main_score=table[main_score_name].values,
    )

    column_names = [
        "tg_id",
        "tg_num_id",
        "is_decoy",
        "is_top_peak",
        "is_train",
        "main_score",
    ]
    used_var_column_names = []
    used_var_column_ids = []
    for i, v in enumerate(var_column_names):
        col_name = "var_%d" % i
        col_data = table[v]
        if pd.isnull(col_data).all():
            logger.debug(
                f"Column {v} contains only invalid/missing values. Column will be dropped."
            )
            continue
        else:
            used_var_column_names.append(v)
            used_var_column_ids.append(col_name)

        data[col_name] = col_data
        column_names.append(col_name)

    data["classifier_score"] = empty_col
    column_names.append("classifier_score")

    # build data frame:
    df = pd.DataFrame(data, columns=column_names)

    all_score_columns = (main_score_name,) + tuple(used_var_column_names)
    df = cleanup_and_check(df)
    return df, all_score_columns, used_var_column_ids


def update_chosen_main_score_in_table(train, score_columns, use_as_main_score):
    """
    Updates the main score column in the feature table.

    Args:
        train (Experiment): The training data.
        score_columns (list): List of score columns.
        use_as_main_score (str): The column to use as the new main score.

    Returns:
        tuple: Updated training data and score columns.
    """
    # Get current main score column name
    old_main_score_column = [col for col in score_columns if "main" in col][0]
    # Get tables aliased score variable name
    df_column_score_alias = [
        col
        for col in train.df.columns
        if col
        not in [
            "tg_id",
            "tg_num_id",
            "is_decoy",
            "is_top_peak",
            "is_train",
            "classifier_score",
        ]
    ]
    # Generate mapping to rename columns in table
    mapper = {
        alias_col: col for alias_col, col in zip(df_column_score_alias, score_columns)
    }
    # Rename columns with actual feature score names
    train.df.rename(columns=mapper, inplace=True)
    # Update coulmns to set new main score column based on most important feature column
    updated_score_columns = [
        col.replace("main_", "") if col == old_main_score_column else col
        for col in score_columns
    ]
    updated_score_columns = [
        col.replace("var", "main_var") if col == use_as_main_score else col
        for col in updated_score_columns
    ]
    updated_score_columns = sorted(
        updated_score_columns,
        key=lambda x: (x != use_as_main_score.replace("var", "main_var"), x),
    )
    updated_score_columns = [
        (
            old_main_score_column
            if old_main_score_column.replace("main_", "") == col
            else col
        )
        for col in updated_score_columns
    ]
    # Rename columns with feature aliases
    mapper = {
        v: "var_{0}".format(i)
        for i, v in enumerate(updated_score_columns[1 : len(updated_score_columns)])
    }
    mapper[updated_score_columns[0].replace("main_", "")] = "main_score"
    train.df.rename(columns=mapper, inplace=True)
    # Re-order main_score column index
    temp_col = train.df.pop("main_score")
    train.df.insert(5, temp_col.name, temp_col)
    logger.debug(
        f"Updated main score column from {old_main_score_column} to {use_as_main_score}..."
    )
    return train, tuple(updated_score_columns)


class Experiment(object):
    """
    Encapsulates data operations for peak groups, decoys, and targets.

    Attributes:
        df (pd.DataFrame): The underlying data.
    """

    @profile
    def __init__(self, df):
        self.df = df.copy()

    def log_summary(self):
        """
        Logs a summary of the input data, including the number of peak groups,
        group IDs, and scores.
        """
        logger.info("Summary of input data:")
        logger.info("%d peak groups" % len(self.df))
        logger.info("%d group ids" % len(self.df.tg_id.unique()))
        logger.info(
            "%d scores including main score" % (len(self.df.columns.values) - 6)
        )

    def __getitem__(self, *args):
        return self.df.__getitem__(*args)

    def __setitem__(self, *args):
        return self.df.__setitem__(*args)

    def __setattr__(self, name, value):
        if name not in [
            "df",
        ]:
            raise click.ClickException("Use '[...]' syntax to set input file columns.")
        object.__setattr__(self, name, value)

    def scale_features(self, score_columns):
        """
        Scales the features to the [0, 1] range.

        Args:
            score_columns (list): List of columns to be scaled.
        """

        scaler = StandardScaler()
        # Get the feature matrix from the DataFrame
        feature_matrix = self.df[score_columns].values
        # Fit the scaler to the feature matrix
        scaler.fit(feature_matrix)
        # Transform the feature matrix
        scaled_features = scaler.transform(feature_matrix)
        # Update the DataFrame with the scaled features
        for i, col in enumerate(score_columns):
            logger.trace(
                f"Column {col} original range: min={feature_matrix[:, i].min()}, max={feature_matrix[:, i].max()}, mean={feature_matrix[:, i].mean()}, std={feature_matrix[:, i].std()}"
            )
            logger.trace(
                f"Column {col} scaled range: min={scaled_features[:, i].min()}, max={scaled_features[:, i].max()}, mean={scaled_features[:, i].mean()}, std={scaled_features[:, i].std()}"
            )
            self.df[col] = scaled_features[:, i]

    def set_and_rerank(self, col_name, scores):
        """
        Sets a column with new scores and re-ranks the data.

        Args:
            col_name (str): Name of the column to update.
            scores (array-like): New scores to assign.
        """
        pass
        self.df.loc[:, col_name] = scores
        self.rank_by(col_name)

    def rank_by(self, score_col_name):
        """
        Ranks the data by the specified score column.

        Args:
            score_col_name (str): Name of the score column to rank by.
        """
        flags = find_top_ranked(
            self.df.tg_num_id.values,
            self.df[score_col_name].values.astype(np.float32, copy=False),
        )
        self.df.is_top_peak = flags

    def get_top_test_peaks(self):
        """
        Retrieves the top test peaks.

        Returns:
            Experiment: A new Experiment containing the top test peaks.
        """
        df = self.df
        return Experiment(df[(df.is_train == False) & (df.is_top_peak == True)])

    def get_decoy_peaks(self):
        """
        Retrieves the decoy peaks.

        Returns:
            Experiment: A new Experiment containing the decoy peaks.
        """
        return Experiment(self.df[self.df.is_decoy == True])

    def get_target_peaks(self):
        """
        Retrieves the target peaks.

        Returns:
            Experiment: A new Experiment containing the target peaks.
        """
        return Experiment(self.df[self.df.is_decoy == False])

    def get_top_decoy_peaks(self):
        """
        Retrieves the top decoy peaks.

        Returns:
            Experiment: A new Experiment containing the top decoy peaks.
        """
        ix_top = self.df.is_top_peak == True
        return Experiment(self.df[(self.df.is_decoy == True) & ix_top])

    def get_top_target_peaks(self):
        """
        Retrieves the top target peaks.

        Returns:
            Experiment: A new Experiment containing the top target peaks.
        """
        ix_top = self.df.is_top_peak == True
        return Experiment(self.df[(self.df.is_decoy == False) & ix_top])

    def get_feature_matrix(self, use_main_score):
        """
        Retrieves the feature matrix for scoring.

        Args:
            use_main_score (bool): Whether to include the main score.

        Returns:
            np.ndarray: The feature matrix.
        """
        min_col = 5 if use_main_score else 6
        return self.df.iloc[:, min_col:-1].values

    def normalize_score_by_decoys(self, score_col_name):
        """
        Normalizes the decoy scores to mean 0 and standard deviation 1,
        and scales the target scores accordingly.

        Args:
            score_col_name (str): Name of the score column to normalize.
        """
        td_scores = self.get_top_decoy_peaks()[score_col_name]
        mu, nu = mean_and_std_dev(td_scores)

        if nu == 0:
            raise ValueError(
                "Warning: Standard deviation of decoy scores is zero. Cannot normalize scores."
            )

        self.df.loc[:, score_col_name] = (self.df[score_col_name] - mu) / nu

    def filter_(self, idx):
        """
        Filters the data based on the given index.

        Args:
            idx (array-like): Boolean index for filtering.

        Returns:
            Experiment: A new Experiment containing the filtered data.
        """
        return Experiment(self.df[idx])

    @profile
    def add_peak_group_rank(self):
        """
        Adds a peak group rank column to the data.
        """
        ids = self.df.tg_num_id.values
        scores = self.df.d_score.values
        peak_group_ranks = rank(ids, scores.astype(np.float32, copy=False))
        self.df["peak_group_rank"] = peak_group_ranks

    @profile
    def split_for_xval(self, fraction, is_test):
        """
        Splits the data for cross-validation.

        Args:
            fraction (float): Fraction of data to use for training.
            is_test (bool): Whether this is a test split.
        """
        df = self.df
        decoy_ids = df[df.is_decoy == True].tg_id.unique()
        target_ids = df[df.is_decoy == False].tg_id.unique()

        if not is_test:
            random.shuffle(decoy_ids)
            random.shuffle(target_ids)
        else:
            decoy_ids = sorted(decoy_ids)
            target_ids = sorted(target_ids)

        decoy_ids = decoy_ids[: int(len(decoy_ids) * fraction) + 1]
        target_ids = target_ids[: int(len(target_ids) * fraction) + 1]
        learn_ids = np.concatenate((decoy_ids, target_ids))
        ix_learn = df.tg_id.isin(learn_ids)
        df.loc[ix_learn, "is_train"] = True
        df.loc[~ix_learn, "is_train"] = False

    def get_train_peaks(self):
        """
        Retrieves the training peaks.

        Returns:
            Experiment: A new Experiment containing the training peaks.
        """
        df = self.df[self.df.is_train == True]
        return Experiment(df)
