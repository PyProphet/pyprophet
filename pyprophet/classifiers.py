from typing import List
import numpy as np
import pandas as pd
import click
from loguru import logger
import inspect

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    cross_val_score,
    KFold,
)
import xgboost as xgb
from .data_handling import Experiment

from hyperopt import hp, tpe
from hyperopt.fmin import fmin

import sys


class AbstractLearner(object):

    def learn(self, decoy_peaks, target_peaks, use_main_score=True):
        raise NotImplementedError()

    def score(self, peaks, use_main_score):
        raise NotImplementedError()

    def get_parameters(self):
        raise NotImplementedError()

    def set_parameters(self, param):
        raise NotImplementedError()

    @classmethod
    def averaged_learner(clz, params, **kwargs):
        raise NotImplementedError()


class LinearLearner(AbstractLearner):

    def score(self, peaks, use_main_score):
        X = peaks.get_feature_matrix(use_main_score)
        result = np.dot(X, self.get_parameters()).astype(np.float32)
        return result

    @classmethod
    def averaged_learner(clz, params, **kwargs):
        assert LinearLearner in inspect.getmro(clz)
        learner = clz(**kwargs)
        avg_param = np.vstack(params).mean(axis=0)
        learner.set_parameters(avg_param)
        return learner

    def set_parameters(self, classifier):
        self.classifier = classifier
        return self

    def get_weights(self, features: List[str]) -> pd.DataFrame:
        """
        Return a DataFrame with feature names and their weights, sorted by absolute weight.
        """
        if self.classifier is None:
            raise ValueError("Classifier has not been trained yet.")

        weights = self.classifier.coef_
        intercept = self.classifier.intercept_

        assert weights.shape[0] == 1
        assert weights.shape[1] == len(features)
        assert len(intercept) == 1

        weights = weights.flatten()
        df = (
            pd.DataFrame(
                {"feature": features, "weight": weights, "abs_weight": abs(weights)}
            )
            .sort_values("abs_weight", ascending=False)
            .drop(columns="abs_weight")
        )

        # Add intercept row (optional)
        df = pd.concat(
            [df, pd.DataFrame([{"feature": "intercept", "weight": intercept[0]}])],
            ignore_index=True,
        )

        return df


class LDALearner(LinearLearner):

    def __init__(self):
        self.classifier = None
        self.scalings = None
        self.autotune = None

    def learn(self, decoy_peaks, target_peaks, use_main_score=True):
        assert isinstance(decoy_peaks, Experiment)
        assert isinstance(target_peaks, Experiment)

        X0 = decoy_peaks.get_feature_matrix(use_main_score)
        X1 = target_peaks.get_feature_matrix(use_main_score)
        X = np.vstack((X0, X1))
        y = np.zeros((X.shape[0],))
        y[X0.shape[0] :] = 1.0
        classifier = LinearDiscriminantAnalysis()
        classifier.fit(X, y)
        self.classifier = classifier
        self.scalings = classifier.scalings_.flatten()
        return self

    def get_parameters(self):
        return self.scalings

    def set_parameters(self, w):
        self.scalings = w
        return self


class SVMLearner(LinearLearner):
    """
    Learner that uses Linear Support Vector Classification (SVC).
    """

    def __init__(self, C, max_iter=1000, autotune=False):
        self.classifier = None
        self.weights = None
        self.C = C
        self.max_iter = max_iter
        self.autotune = autotune

    def tune(
        self, decoy_peaks, target_peaks, use_main_score=True, cv_splits=3, n_jobs=-1
    ):
        """
        Tune hyperparameters (C, max_iter) using GridSearchCV.
        """
        assert isinstance(decoy_peaks, Experiment)
        assert isinstance(target_peaks, Experiment)

        # Prepare feature matrices
        X0 = decoy_peaks.get_feature_matrix(use_main_score)
        X1 = target_peaks.get_feature_matrix(use_main_score)
        X = np.vstack((X0, X1))

        y = np.zeros((X.shape[0],))
        y[X0.shape[0] :] = 1.0

        # Set the parameter grid for C and max_iter
        param_grid = {"C": [0.1, 1.0, 10.0], "max_iter": [1000, 2000, 5000]}

        logger.info("Tuning hyperparameters for LinearSVC.")

        # Set up GridSearchCV with LinearSVC
        grid_search = GridSearchCV(
            LinearSVC(dual=False),
            param_grid,
            cv=KFold(cv_splits, shuffle=True, random_state=42),
            n_jobs=n_jobs,
            scoring="roc_auc",
            verbose=3,
        )

        # Perform the grid search
        grid_search.fit(X, y)

        # Log the best parameters
        logger.info(
            f"Best hyperparameters found: {[f'{key}: {value} | ' for key, value in grid_search.best_params_.items()]}"
        )

        # Set the best parameters found
        self.C = grid_search.best_params_["C"]
        self.max_iter = grid_search.best_params_["max_iter"]

        # Optionally, retrain the model with the best parameters
        # self.learn(decoy_peaks, target_peaks, use_main_score)

        return self

    def learn(self, decoy_peaks, target_peaks, use_main_score=True):
        assert isinstance(decoy_peaks, Experiment)
        assert isinstance(target_peaks, Experiment)

        X0 = decoy_peaks.get_feature_matrix(use_main_score)
        X1 = target_peaks.get_feature_matrix(use_main_score)
        X = np.vstack((X0, X1))

        y = np.zeros((X.shape[0],))
        y[X0.shape[0] :] = 1.0

        classifier = LinearSVC(dual=False, C=self.C, max_iter=self.max_iter)
        classifier.fit(X, y)

        self.classifier = classifier
        # self.weights = classifier.coef_.flatten()

        return self

    def score(self, peaks, use_main_score):
        X = peaks.get_feature_matrix(use_main_score)
        return self.classifier.decision_function(X)

    def get_parameters(self):
        return self.classifier

    def set_parameters(self, classifier):
        # self.weights = w
        self.classifier = classifier
        return self


class XGBLearner(AbstractLearner):

    def __init__(
        self, autotune, xgb_hyperparams, xgb_params, xgb_params_space, threads
    ):
        self.classifier = None
        self.importance = None
        self.autotune = autotune
        self.xgb_hyperparams = xgb_hyperparams
        self.xgb_params = xgb_params
        self.xgb_params_space = xgb_params_space
        self.xgb_params_tuned = xgb_params
        self.threads = threads
        self.xgb_params["nthread"] = self.threads

    def tune(self, decoy_peaks, target_peaks, use_main_score=True):
        def objective(params):
            params = {
                "eta": "{:.3f}".format(params["eta"]),
                "gamma": "{:.3f}".format(params["gamma"]),
                "max_depth": int(params["max_depth"]),
                "min_child_weight": int(params["min_child_weight"]),
                "subsample": "{:.3f}".format(params["subsample"]),
                "colsample_bytree": "{:.3f}".format(params["colsample_bytree"]),
                "colsample_bylevel": "{:.3f}".format(params["colsample_bylevel"]),
                "colsample_bynode": "{:.3f}".format(params["colsample_bynode"]),
                "lambda": "{:.3f}".format(params["lambda"]),
                "alpha": "{:.3f}".format(params["alpha"]),
                "scale_pos_weight": "{:.3f}".format(params["scale_pos_weight"]),
            }

            clf = xgb.XGBClassifier(
                random_state=42,
                verbosity=0,
                objective="binary:logitraw",
                eval_metric="auc",
                **params,
            )

            score = cross_val_score(
                clf,
                X,
                y,
                scoring="roc_auc",
                n_jobs=self.threads,
                cv=KFold(n_splits=3, shuffle=True, random_state=42),
            ).mean()
            # click.echo("Info: AUC: {:.3f} hyperparameters: {}".format(score, params))
            return score

        click.echo("Info: Autotuning of XGB hyperparameters.")

        assert isinstance(decoy_peaks, Experiment)
        assert isinstance(target_peaks, Experiment)

        X0 = decoy_peaks.get_feature_matrix(use_main_score)
        X1 = target_peaks.get_feature_matrix(use_main_score)
        X = np.vstack((X0, X1))
        y = np.zeros((X.shape[0],))
        y[X0.shape[0] :] = 1.0

        # Tune complexity hyperparameters
        xgb_params_complexity = self.xgb_params_tuned
        xgb_params_complexity.update(
            {k: self.xgb_params_space[k] for k in ("max_depth", "min_child_weight")}
        )

        rng = np.random.default_rng(42)
        best_complexity = fmin(
            fn=objective,
            space=xgb_params_complexity,
            algo=tpe.suggest,
            max_evals=self.xgb_hyperparams["autotune_num_rounds"],
            rstate=rng,
        )
        best_complexity["max_depth"] = int(best_complexity["max_depth"])
        best_complexity["min_child_weight"] = int(best_complexity["min_child_weight"])

        self.xgb_params_tuned.update(best_complexity)

        # Tune gamma hyperparameter
        xgb_params_gamma = self.xgb_params_tuned
        xgb_params_gamma["gamma"] = self.xgb_params_space["gamma"]

        best_gamma = fmin(
            fn=objective,
            space=xgb_params_gamma,
            algo=tpe.suggest,
            max_evals=self.xgb_hyperparams["autotune_num_rounds"],
            rstate=rng,
        )

        self.xgb_params_tuned.update(best_gamma)

        # Tune subsampling hyperparameters
        xgb_params_subsampling = self.xgb_params_tuned
        xgb_params_subsampling.update(
            {
                k: self.xgb_params_space[k]
                for k in (
                    "subsample",
                    "colsample_bytree",
                    "colsample_bylevel",
                    "colsample_bynode",
                )
            }
        )

        best_subsampling = fmin(
            fn=objective,
            space=xgb_params_subsampling,
            algo=tpe.suggest,
            max_evals=self.xgb_hyperparams["autotune_num_rounds"],
            rstate=rng,
        )

        self.xgb_params_tuned.update(best_subsampling)

        # Tune regularization hyperparameters
        xgb_params_regularization = self.xgb_params_tuned
        xgb_params_regularization.update(
            {k: self.xgb_params_space[k] for k in ("lambda", "alpha")}
        )

        best_regularization = fmin(
            fn=objective,
            space=xgb_params_regularization,
            algo=tpe.suggest,
            max_evals=self.xgb_hyperparams["autotune_num_rounds"],
            rstate=rng,
        )

        self.xgb_params_tuned.update(best_regularization)

        # Tune learning rate
        xgb_params_learning = self.xgb_params_tuned
        xgb_params_learning["eta"] = self.xgb_params_space["eta"]

        best_learning = fmin(
            fn=objective,
            space=xgb_params_learning,
            algo=tpe.suggest,
            max_evals=self.xgb_hyperparams["autotune_num_rounds"],
            rstate=rng,
        )

        self.xgb_params_tuned.update(best_learning)
        click.echo("Info: Optimal hyperparameters: {}".format(self.xgb_params_tuned))

        self.xgb_params = self.xgb_params_tuned

        return self

    def learn(self, decoy_peaks, target_peaks, use_main_score=True):
        assert isinstance(decoy_peaks, Experiment)
        assert isinstance(target_peaks, Experiment)

        X0 = decoy_peaks.get_feature_matrix(use_main_score)
        X1 = target_peaks.get_feature_matrix(use_main_score)
        X = np.vstack((X0, X1))
        y = np.zeros((X.shape[0],))
        y[X0.shape[0] :] = 1.0

        # prepare training and validation data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.xgb_hyperparams["test_size"], random_state=42
        )
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # learn model
        classifier = xgb.train(
            params=self.xgb_params,
            dtrain=dtrain,
            num_boost_round=self.xgb_hyperparams["num_boost_round"],
            evals=[(dval, "validation")],
            early_stopping_rounds=self.xgb_hyperparams["early_stopping_rounds"],
            verbose_eval=False,
        )

        self.importance = classifier.get_score(importance_type="gain")
        self.classifier = classifier
        return self

    def score(self, peaks, use_main_score):
        X = peaks.get_feature_matrix(use_main_score)
        dtest = xgb.DMatrix(X)
        result = self.classifier.predict(dtest)
        return result.astype(np.float32)

    def get_parameters(self):
        return self.classifier

    def set_parameters(self, classifier):
        self.classifier = classifier
        self.importance = classifier.get_score(importance_type="gain")
        return self
