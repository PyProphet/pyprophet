# encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

try:
    profile
except NameError:
    profile = lambda x: x

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import inspect

from data_handling import Experiment


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
