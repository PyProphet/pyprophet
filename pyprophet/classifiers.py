import numpy as np
import inspect

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from .data_handling import Experiment

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


class SVMLearner(AbstractLearner):

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
        classifier = NuSVC(nu=0.1, probability=True)
        classifier.fit(X, y)
        self.classifier = classifier
        return self

    def score(self, peaks, use_main_score):
        X = peaks.get_feature_matrix(use_main_score)
        result = self.classifier.predict_proba(X)
        return result[:,1].astype(np.float32)

    def get_parameters(self):
        return self.classifier

    def set_parameters(self, w):
        self.classifier = w
        return self


class RFLearner(AbstractLearner):

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
        classifier = RandomForestClassifier()
        classifier.fit(X, y)
        self.classifier = classifier
        self.scalings = classifier.feature_importances_.flatten()
        return self

    def score(self, peaks, use_main_score):
        X = peaks.get_feature_matrix(use_main_score)
        result = self.classifier.predict_proba(X)
        return result[:,1].astype(np.float32)

    def get_parameters(self):
        return self.classifier

    def set_parameters(self, w):
        self.classifier = w
        return self

class XGBLearner(AbstractLearner):

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

        # prepare training data
        dtrain = xgb.DMatrix(X, label=y)

        # specify parameters
        param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
        param['nthread'] = 4
        param['eval_metric'] = 'auc'

        # learning
        num_round = 10
        classifier = xgb.train(param, dtrain, num_round)

        self.classifier = classifier
        # self.scalings = xgb.importance(feature_names = sparse_matrix@Dimnames[[2]], model = classifier)
        return self

    def score(self, peaks, use_main_score):
        X = peaks.get_feature_matrix(use_main_score)
        dtest = xgb.DMatrix(X)
        result = self.classifier.predict(dtest)
        return result.astype(np.float32)

    def get_parameters(self):
        return self.classifier

    def set_parameters(self, w):
        self.classifier = w
        return self
