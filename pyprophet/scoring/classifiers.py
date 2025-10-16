"""
This module defines various classifiers (learners) for statistical scoring and
error estimation in targeted proteomics and glycoproteomics data analysis.

Classes:
    - AbstractLearner: Base class for defining a learner interface.
    - LinearLearner: Implements a linear classifier for scoring.
    - LDALearner: Implements a Linear Discriminant Analysis (LDA) learner.
    - SVMLearner: Implements a Support Vector Machine (SVM) learner.
    - XGBLearner: Implements an XGBoost-based learner for scoring.

Each learner provides methods for training, scoring, and parameter management.
"""

import inspect
from typing import List

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier

from .data_handling import Experiment


class AbstractLearner(object):
    """
    Abstract base class for defining a learner interface.

    Methods:
        - learn: Abstract method for training the learner.
        - score: Abstract method for scoring data.
        - get_parameters: Abstract method for retrieving model parameters.
        - set_parameters: Abstract method for setting model parameters.
        - averaged_learner: Abstract method for creating an averaged learner.
    """

    def learn(self, decoy_peaks, target_peaks, use_main_score=True):
        """Train the learner using decoy and target peaks."""
        raise NotImplementedError()

    def score(self, peaks, use_main_score):
        """Score the given peaks using the trained model."""
        raise NotImplementedError()

    def get_parameters(self):
        """Retrieve the parameters of the trained model."""
        raise NotImplementedError()

    def set_parameters(self, param):
        """Set the parameters of the model."""
        raise NotImplementedError()

    @classmethod
    def averaged_learner(clz, params, **kwargs):
        """Create an averaged learner from multiple parameter sets."""
        raise NotImplementedError()


class LinearLearner(AbstractLearner):
    """
    Implements a linear classifier for scoring.

    Methods:
        - score: Score the given peaks using the linear model.
        - averaged_learner: Create an averaged learner from multiple parameter sets.
        - set_parameters: Set the parameters of the linear model.
        - get_weights: Retrieve feature weights from the model.
    """

    def score(self, peaks, use_main_score):
        """Score the given peaks using the linear model."""
        X = peaks.get_feature_matrix(use_main_score)
        result = np.dot(X, self.get_parameters()).astype(np.float32)
        return result

    @classmethod
    def averaged_learner(clz, params, **kwargs):
        """Create an averaged learner from multiple parameter sets."""
        assert LinearLearner in inspect.getmro(clz)
        learner = clz(**kwargs)
        avg_param = np.vstack(params).mean(axis=0)
        learner.set_parameters(avg_param)
        return learner

    def set_parameters(self, classifier):
        """Set the parameters of the linear model."""
        self.classifier = classifier
        return self

    def get_weights(self, features: List[str]) -> pd.DataFrame:
        """
        Return a DataFrame with feature names and their weights, sorted by absolute weight.

        Args:
            features (List[str]): List of feature names.

        Returns:
            pd.DataFrame: DataFrame containing feature names and weights.
        """
        if self.classifier is None:
            raise ValueError("Classifier has not been trained yet.")

        elif isinstance(self.classifier, np.ndarray):
            # If classifier is a numpy array, assume it contains weights
            weights = self.classifier
            assert weights.shape[0] == len(features)
        elif isinstance(self.classifier, LinearSVC):
            weights = self.classifier.coef_
            _intercept = self.classifier.intercept_
            assert weights.shape[0] == 1
            assert weights.shape[1] == len(features)

        weights = weights.flatten()
        df = pd.DataFrame({"score": features, "weight": weights})

        # # Add intercept row (optional)
        # df = pd.concat(
        #     [df, pd.DataFrame([{"score": "intercept", "weight": intercept[0]}])],
        #     ignore_index=True,
        # )

        return df


class LDALearner(LinearLearner):
    """
    Implements a Linear Discriminant Analysis (LDA) learner.

    Methods:
        - learn: Train the LDA model using decoy and target peaks.
        - get_parameters: Retrieve the scaling parameters of the LDA model.
        - set_parameters: Set the scaling parameters of the LDA model.
    """

    def __init__(self):
        self.classifier = None
        self.scalings = None
        self.autotune = None

    def learn(self, decoy_peaks, target_peaks, use_main_score=True):
        """Train the LDA model using decoy and target peaks."""
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
        """Retrieve the scaling parameters of the LDA model."""
        return self.scalings

    def set_parameters(self, w):
        """Set the scaling parameters of the LDA model."""
        self.scalings = w
        return self


class SVMLearner(LinearLearner):
    """
    Implements a Support Vector Linear Classification (SVM) learner.

    Methods:
        - tune: Tune hyperparameters (C, max_iter) using GridSearchCV.
        - learn: Train the SVM model using decoy and target peaks.
        - score: Score the given peaks using the SVM model.
        - get_parameters: Retrieve the parameters of the SVM model.
        - set_parameters: Set the parameters of the SVM model.
    """

    def __init__(self, C, max_iter=1000, autotune=False):
        self.classifier = None
        self.weights = None
        self.C = C
        self.max_iter = max_iter
        self.class_weight = None
        self.autotune = autotune

    def tune(
        self, decoy_peaks, target_peaks, use_main_score=True, cv_splits=3, n_jobs=-1
    ):
        """
        Tune hyperparameters (C, max_iter) using GridSearchCV.

        Args:
            decoy_peaks (Experiment): Decoy peaks data.
            target_peaks (Experiment): Target peaks data.
            use_main_score (bool): Whether to use the main score.
            cv_splits (int): Number of cross-validation splits.
            n_jobs (int): Number of parallel jobs.

        Returns:
            SVMLearner: The tuned learner instance.
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
        param_grid = {
            "C": [0.1, 1.0, 10.0],
            "max_iter": [1000, 2000, 5000],
            "class_weight": [
                {1: neg, 0: pos} for neg in (0.1, 1, 10) for pos in (0.1, 1, 10)
            ]
            + ["balanced"],
        }

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

        best_params_str = [
            f"{key}: {str(value).replace('{', '[').replace('}', ']')} | "
            for key, value in grid_search.best_params_.items()
        ]

        # Log the best parameters
        logger.info(f"Best hyperparameters found: {best_params_str}")

        # Set the best parameters found
        self.C = grid_search.best_params_["C"]
        self.max_iter = grid_search.best_params_["max_iter"]
        self.class_weight = grid_search.best_params_["class_weight"]

        return self

    def learn(self, decoy_peaks, target_peaks, use_main_score=True):
        """Train the SVM model using decoy and target peaks."""
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
        """Score the given peaks using the SVM model."""
        X = peaks.get_feature_matrix(use_main_score)
        # Check if self.classifier is loaded weights (an numpy.ndarray object)
        if isinstance(self.classifier, np.ndarray):
            return np.dot(X, self.classifier).astype(np.float32)
        elif isinstance(self.classifier, LinearSVC):
            return self.classifier.decision_function(X)

    def get_parameters(self):
        """Retrieve the parameters of the SVM model."""
        return self.classifier

    def set_parameters(self, classifier):
        """Set the parameters of the SVM model."""
        # self.weights = w
        self.classifier = classifier
        return self


class HistGBCLearner(AbstractLearner):
    """
    Implements a scikit-learn HistGradientBoostingClassifier-based learner for scoring.

    Methods:
        - tune: Tune hyperparameters using RandomizedSearchCV.
        - learn: Train the HistGradientBoosting model using decoy and target peaks.
        - score: Score the given peaks using the trained model.
        - get_parameters: Retrieve the parameters of the model.
        - set_parameters: Set the parameters of the model.
    """

    def __init__(self, autotune=False, hgb_params=None, threads=1):
        self.classifier = None
        self.importance = None
        self.autotune = autotune
        self.threads = threads
        self.hgb_params = hgb_params or {}

    def tune(
        self, decoy_peaks, target_peaks, use_main_score=True, cv_splits=3, n_jobs=-1
    ):
        """
        Tune hyperparameters using RandomizedSearchCV.
        """
        logger.info(
            "Autotuning of HistGradientBoosting hyperparameters using RandomizedSearchCV."
        )

        assert isinstance(decoy_peaks, Experiment)
        assert isinstance(target_peaks, Experiment)

        X0 = decoy_peaks.get_feature_matrix(use_main_score)
        X1 = target_peaks.get_feature_matrix(use_main_score)
        X = np.vstack((X0, X1))
        y = np.zeros((X.shape[0],))
        y[X0.shape[0] :] = 1.0

        param_dist = {
            "learning_rate": np.linspace(0.01, 0.3, num=30),
            "max_depth": list(range(2, 9)),
            "max_leaf_nodes": [None] + list(range(15, 65, 5)),
            "min_samples_leaf": [1, 2, 3, 5, 10],
            "l2_regularization": np.linspace(0.0, 1.0, num=20),
        }

        base_model = HistGradientBoostingClassifier(random_state=42)

        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=10,
            cv=KFold(n_splits=cv_splits, shuffle=True, random_state=42),
            n_jobs=n_jobs,
            scoring="roc_auc",
            verbose=3,
            random_state=42,
        )

        random_search.fit(X, y)

        best_params = random_search.best_params_
        self.hgb_params.update(best_params)

        best_params_str = [
            f"{key}: {str(value).replace('{', '[').replace('}', ']')} | "
            for key, value in self.hgb_params.items()
        ]
        logger.info(f"Optimal hyperparameters: {best_params_str}")

        return self

    def learn(self, decoy_peaks, target_peaks, use_main_score=True):
        """Train the HistGradientBoosting model using decoy and target peaks."""
        assert isinstance(decoy_peaks, Experiment)
        assert isinstance(target_peaks, Experiment)


        X0 = decoy_peaks.get_feature_matrix(use_main_score)
        X1 = target_peaks.get_feature_matrix(use_main_score)
        X = np.vstack((X0, X1))
        y = np.zeros((X.shape[0],))
        y[X0.shape[0] :] = 1.0

        # Configure classifier with user params or defaults
        clf_params = dict(self.hgb_params)
        clf_params.setdefault("random_state", 42)
        clf_params.setdefault("max_iter", 100)
        clf_params.setdefault("early_stopping", True)
        clf_params.setdefault("validation_fraction", 0.1)

        # Filter out any params not accepted by HistGradientBoostingClassifier
        import inspect
        valid_params = inspect.signature(HistGradientBoostingClassifier.__init__).parameters
        clf_params = {k: v for k, v in clf_params.items() if k in valid_params}

        classifier = HistGradientBoostingClassifier(**clf_params)
        classifier.fit(X, y)

        self.classifier = classifier
        # Store feature importances as dict keyed by f{index} to match XGBoost format
        feats = classifier.feature_importances_
        self.importance = {f"f{i}": float(v) for i, v in enumerate(feats)}

        return self

    def score(self, peaks, use_main_score):
        """Score the given peaks using the HistGradientBoosting model."""
        X = peaks.get_feature_matrix(use_main_score)
        # Use decision_function for compatibility with XGBoost scoring
        result = self.classifier.decision_function(X)
        return result.astype(np.float32)

    def get_parameters(self):
        """Retrieve the parameters of the model."""
        return self.classifier

    def set_parameters(self, classifier):
        """Set the parameters of the model."""
        self.classifier = classifier
        if hasattr(classifier, "feature_importances_"):
            feats = classifier.feature_importances_
            self.importance = {f"f{i}": float(v) for i, v in enumerate(feats)}
        else:
            self.importance = {}
        return self


class XGBLearner(AbstractLearner):
    """
    Implements an XGBoost-based learner for scoring.

    Methods:
        - tune: Tune hyperparameters using RandomizedSearchCV.
        - learn: Train the XGBoost model using decoy and target peaks.
        - score: Score the given peaks using the XGBoost model.
        - get_parameters: Retrieve the parameters of the XGBoost model.
        - set_parameters: Set the parameters of the XGBoost model.
    """

    def __init__(self, autotune, xgb_params, threads):
        self.classifier = None
        self.importance = None
        self.autotune = autotune
        self.xgb_hyperparams = {
            "autotune_num_rounds": 10,
            "num_boost_round": 100,
            "early_stopping_rounds": 10,
            "test_size": 0.33,
        }
        self.xgb_params = xgb_params
        self.xgb_params_tuned = xgb_params
        self.threads = threads
        self.xgb_params["nthread"] = self.threads

    def tune(
        self, decoy_peaks, target_peaks, use_main_score=True, cv_splits=3, n_jobs=-1
    ):
        """
        Tune hyperparameters using RandomizedSearchCV for faster optimization.

        Args:
            decoy_peaks (Experiment): Decoy peaks data.
            target_peaks (Experiment): Target peaks data.
            use_main_score (bool): Whether to use the main score.
            cv_splits (int): Number of cross-validation splits.
            n_jobs (int): Number of parallel jobs.

        Returns:
            XGBLearner: The tuned learner instance.
        """
        logger.info("Autotuning of XGB hyperparameters using RandomizedSearchCV.")

        assert isinstance(decoy_peaks, Experiment)
        assert isinstance(target_peaks, Experiment)

        # Prepare feature matrices
        X0 = decoy_peaks.get_feature_matrix(use_main_score)
        X1 = target_peaks.get_feature_matrix(use_main_score)
        X = np.vstack((X0, X1))
        y = np.zeros((X.shape[0],))
        y[X0.shape[0] :] = 1.0

        # Define parameter distributions for RandomizedSearchCV
        param_dist = {
            "learning_rate": np.linspace(0.0, 0.3, num=100),
            "gamma": np.linspace(0.0, 0.5, num=100),
            "max_depth": list(range(2, 9)),  # 2 to 8 inclusive
            "min_child_weight": list(range(1, 6)),  # 1 to 5 inclusive
            "subsample": [1.0],  # Fixed value
            "colsample_bytree": [1.0],  # Fixed value
            "reg_lambda": np.linspace(0.0, 1.0, num=100),
            "reg_alpha": np.linspace(0.0, 1.0, num=100),
            "scale_pos_weight": [1.0],  # Fixed value
        }

        # Create base model with fixed parameters
        xgb_model = xgb.XGBClassifier(
            objective="binary:logitraw",
            eval_metric="auc",
            nthread=self.threads,
            verbosity=0,
            random_state=42,
            colsample_bylevel=1.0,  # Fixed params moved here
            colsample_bynode=1.0,  # Fixed params moved here
        )

        # Set up RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_dist,
            n_iter=self.xgb_hyperparams["autotune_num_rounds"],
            cv=KFold(n_splits=cv_splits, shuffle=True, random_state=42),
            n_jobs=n_jobs,
            scoring="roc_auc",
            verbose=3,
            random_state=42,
        )

        # Perform the random search
        random_search.fit(X, y)

        # Get the best parameters and convert to original parameter names
        best_params = random_search.best_params_
        self.xgb_params_tuned.update(
            {
                "eta": best_params["learning_rate"],
                "gamma": best_params["gamma"],
                "max_depth": best_params["max_depth"],
                "min_child_weight": best_params["min_child_weight"],
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "colsample_bylevel": 1.0,
                "colsample_bynode": 1.0,
                "lambda": best_params["reg_lambda"],
                "alpha": best_params["reg_alpha"],
                "scale_pos_weight": 1.0,
            }
        )

        self.xgb_params = self.xgb_params_tuned
        best_params_str = [
            f"{key}: {str(value).replace('{', '[').replace('}', ']')} | "
            for key, value in self.xgb_params_tuned.items()
        ]
        logger.info(f"Optimal hyperparameters: {best_params_str}")

        return self

    def learn(self, decoy_peaks, target_peaks, use_main_score=True):
        """Train the XGBoost model using decoy and target peaks."""
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
        """Score the given peaks using the XGBoost model."""
        X = peaks.get_feature_matrix(use_main_score)
        dtest = xgb.DMatrix(X)
        result = self.classifier.predict(dtest)
        return result.astype(np.float32)

    def get_parameters(self):
        """Retrieve the parameters of the XGBoost model."""
        return self.classifier

    def set_parameters(self, classifier):
        """Set the parameters of the XGBoost model."""
        self.classifier = classifier
        self.importance = classifier.get_score(importance_type="gain")
        return self
