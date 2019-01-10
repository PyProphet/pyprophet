import numpy as np
import click
import inspect

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score, KFold
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
    def averaged_learner(clz, params):
        raise NotImplementedError()


class LinearLearner(AbstractLearner):

    def score(self, peaks, use_main_score):
        X = peaks.get_feature_matrix(use_main_score)
        result = np.dot(X, self.get_parameters()).astype(np.float32)
        return result

    @classmethod
    def averaged_learner(clz, params):
        assert LinearLearner in inspect.getmro(clz)
        learner = clz()
        avg_param = np.vstack(params).mean(axis=0)
        learner.set_parameters(avg_param)
        return learner

    def set_parameters(self, classifier):
        self.classifier = classifier
        return self


class LDALearner(LinearLearner):

    def __init__(self):
        self.classifier = None
        self.scalings = None

    def learn(self, decoy_peaks, target_peaks, use_main_score=True):
        assert isinstance(decoy_peaks, Experiment)
        assert isinstance(target_peaks, Experiment)

        X0 = decoy_peaks.get_feature_matrix(use_main_score)
        X1 = target_peaks.get_feature_matrix(use_main_score)
        X = np.vstack((X0, X1))
        y = np.zeros((X.shape[0],))
        y[X0.shape[0]:] = 1.0
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


class XGBLearner(AbstractLearner):

    def __init__(self, xgb_hyperparams, xgb_params, xgb_params_space, threads):
        self.classifier = None
        self.importance = None
        self.xgb_hyperparams = xgb_hyperparams
        self.xgb_params = xgb_params
        self.xgb_params_space = xgb_params_space
        self.threads = threads

    def tune(self, decoy_peaks, target_peaks, use_main_score=True):
        def objective(params):
            params = {
                'eta': "{:.3f}".format(params['eta']),
                'gamma': "{:.3f}".format(params['gamma']),
                'max_depth': int(params['max_depth']),
                'min_child_weight': int(params['min_child_weight']),
                'subsample': "{:.3f}".format(params['subsample']),
                'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
                'colsample_bylevel': '{:.3f}'.format(params['colsample_bylevel']),
                'colsample_bynode': '{:.3f}'.format(params['colsample_bynode']),
                'lambda': "{:.3f}".format(params['lambda']),
                'alpha': "{:.3f}".format(params['alpha']),
                'scale_pos_weight': "{:.3f}".format(params['scale_pos_weight']),
            }
            
            clf = xgb.XGBClassifier(silent=1, objective='binary:logitraw', eval_metric='auc', **params)

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0, random_state=42)

            score = cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=KFold()).mean()
            # click.echo("Info: AUC: {:.3f} hyperparameters: {}".format(score, params))
            return score

        click.echo("Info: Autotuning of XGB hyperparameters.")

        assert isinstance(decoy_peaks, Experiment)
        assert isinstance(target_peaks, Experiment)

        X0 = decoy_peaks.get_feature_matrix(use_main_score)
        X1 = target_peaks.get_feature_matrix(use_main_score)
        X = np.vstack((X0, X1))
        y = np.zeros((X.shape[0],))
        y[X0.shape[0]:] = 1.0

        best = fmin(fn=objective, space=self.xgb_params_space, algo=tpe.suggest, max_evals=self.xgb_hyperparams['autotune_num_rounds'])
        best['max_depth'] = int(best['max_depth'])
        best['min_child_weight'] = int(best['min_child_weight'])

        click.echo("Info: Optimal hyperparameters: {}".format(best))

        self.xgb_params = best
        self.xgb_params['silent'] = 1
        self.xgb_params['objective'] = 'binary:logitraw'
        self.xgb_params['eval_metric'] = 'auc'

        return self

    def learn(self, decoy_peaks, target_peaks, use_main_score=True):
        assert isinstance(decoy_peaks, Experiment)
        assert isinstance(target_peaks, Experiment)

        X0 = decoy_peaks.get_feature_matrix(use_main_score)
        X1 = target_peaks.get_feature_matrix(use_main_score)
        X = np.vstack((X0, X1))
        y = np.zeros((X.shape[0],))
        y[X0.shape[0]:] = 1.0

        # prepare training and validation data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.xgb_hyperparams['test_size'], random_state=42)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # learn model
        classifier = xgb.train(params=self.xgb_params, dtrain=dtrain, num_boost_round=self.xgb_hyperparams['num_boost_round'], evals=[(dval,"validation")], early_stopping_rounds=self.xgb_hyperparams['early_stopping_rounds'], verbose_eval=False)

        self.importance = classifier.get_score(importance_type='gain')
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
        self.importance = classifier.get_score(importance_type='gain')
        return self
