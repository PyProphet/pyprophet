# encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
os.putenv("OPENBLAS_NUM_THREADS", "1")

try:
    profile
except NameError:
    profile = lambda x: x

import sklearn.lda
import sklearn.linear_model
import sklearn.svm
import sklearn.preprocessing
import numpy as np
import inspect

from config import CONFIG
import logging

from data_handling import Experiment
from scaler import Scaler
from scaler import NonScaler


class Predictor(object):

    def __init__(self):
        self.post_scaler = NonScaler()

    def score(self, peaks, use_main_score):
        X = peaks.get_feature_matrix(use_main_score)
        return self.post_scaler.scale(self.score_by_matrix(X))

    def score_by_matrix(self, feature_matrix):
        raise NotImplementedError()

    def simplistic(self):
        return False
    


class AbstractLearner(Predictor):

    def format_train_data(self, decoy_peaks, target_peaks, use_main_score=True):
        assert isinstance(decoy_peaks, Experiment)
        assert isinstance(target_peaks, Experiment)
        X0 = decoy_peaks.get_feature_matrix(use_main_score)
        X1 = target_peaks.get_feature_matrix(use_main_score)
        X = np.vstack((X0, X1))
        y = np.zeros((X.shape[0],))
        y[X0.shape[0]:] = 1.0
        return X, y

    def learn(self, decoy_peaks, target_peaks, use_main_score=True):
        raise NotImplementedError()

    def get_coefs(self):
        raise NotImplementedError()



class LinearLearner(AbstractLearner):

    def __init__(self):
    	AbstractLearner.__init__(self)
        self.coefs = None

    def score_by_matrix(self, feature_matrix):
        return np.dot(feature_matrix, self.coefs)

    def simplistic(self):
        return isinstance(self.post_scaler, NonScaler)

    def get_coefs(self):
        return self.coefs



class ConsensusPredictor(Predictor):

    def __init__(self, preds):
    	Predictor.__init__(self)
        linCoefs = [pred.get_coefs() for pred in preds if pred.simplistic() ]
        self.predictors = [pred for pred in preds if not pred.simplistic() ]
        
#        print "lincoefs", linCoefs
#        print "predictors", self.predictors
        
        if len(linCoefs) > 0:
            avg_coefs = np.vstack(linCoefs).mean(axis=0)
            avg_pred = LinearPredictor(avg_coefs)
            self.predictors.append(avg_pred)


    def score_by_matrix(self, feature_matrix):
        scores = [pred.score_by_matrix(feature_matrix) for pred in self.predictors]
        return np.mean(scores, axis=0)


    def get_coefs(self):
        linCoefs = [pred.get_coefs() for pred in self.predictors if isinstance(pred, LinearLearner) ]
        if len(linCoefs) > 0:
            avg_coefs = np.vstack(linCoefs).mean(axis=0)
            return avg_coefs
        else:
            return []



class LinearPredictor(Predictor):

    def __init__(self, coefs):
    	Predictor.__init__(self)
        self.coefs = coefs

    def score_by_matrix(self, feature_matrix):
        return np.dot(feature_matrix, self.coefs)

    def get_coefs(self):
        return self.coefs




class LDALearner(LinearLearner):
    
    def __init__(self):
    	LinearLearner.__init__(self)
        self.classifier = sklearn.lda.LDA()
        logging.info("===> doing LDA")
    
    def learn(self, decoy_peaks, target_peaks, use_main_score=True):
        X, y = self.format_train_data(decoy_peaks, target_peaks, use_main_score)
        self.classifier.fit(X, y)
        self.coefs = self.classifier.scalings_.flatten()
        #scores = np.dot(X, self.scalings)
        return self




class SGDLearner(LinearLearner):

    def __init__(self):
    	LinearLearner.__init__(self)
        if CONFIG.get("classifier.weight_classes"):
            logging.info("===> doing weighted SGD")
            self.classifier = sklearn.linear_model.SGDClassifier(shuffle=True, n_iter=10, class_weight="auto")
        else:
            logging.info("===> doing non-weighted SGD")
            self.classifier = sklearn.linear_model.SGDClassifier(shuffle=True, n_iter=10)
        self.scaler = sklearn.preprocessing.StandardScaler()
        

    def learn(self, decoy_peaks, target_peaks, use_main_score=True):
        X, y = self.format_train_data(decoy_peaks, target_peaks, use_main_score)
        if CONFIG.get("classifier.scale_subscores"):
            self.scaler.fit(X)
            X = self.scaler.transform(X)
        self.classifier.fit(X, y)
        self.coefs = self.classifier.coef_.flatten()
        #scores = np.dot(X, self.scalings)
        return self
    
    
    def simplistic(self):
        return not CONFIG.get("classifier.scale_subscores")
    
     
    def score_by_matrix(self, feature_matrix):
        if CONFIG.get("classifier.scale_subscores"):
            feature_matrix = self.scaler.transform(feature_matrix)
        return np.dot(feature_matrix, self.coefs)
        



class LinearSVMLearner(LinearLearner):

    def __init__(self):
    	LinearLearner.__init__(self)
        c_size = int(CONFIG.get("classifier.cache_size", "500"))
        if CONFIG.get("classifier.weight_classes"):
            logging.info("===> doing weighted SVM")
            self.classifier = sklearn.svm.SVC(kernel='linear', cache_size=c_size, class_weight="auto")
        else:
            logging.info("===> doing non-weighted SVM")
            self.classifier = sklearn.svm.SVC(kernel='linear', cache_size=c_size)
        self.scaler = sklearn.preprocessing.StandardScaler()


    def learn(self, decoy_peaks, target_peaks, use_main_score=True):
        X, y = self.format_train_data(decoy_peaks, target_peaks, use_main_score)
        if CONFIG.get("classifier.scale_subscores"):
            self.scaler = self.scaler.fit(X)
            X = self.scaler.transform(X)
        self.classifier.fit(X, y)
        self.coefs = self.classifier.coef_.flatten()
        #scores = np.dot(X, self.scalings)
        return self
    
     
    def simplistic(self):
        return not CONFIG.get("classifier.scale_subscores")
     
     
    def score_by_matrix(self, feature_matrix):
        if CONFIG.get("classifier.scale_subscores"):
            feature_matrix = self.scaler.transform(feature_matrix)
        return np.dot(feature_matrix, self.coefs)
        



class RbfSVMLearner(AbstractLearner):

    def __init__(self):
    	AbstractLearner.__init__(self)
        c_size = int(CONFIG.get("classifier.cache_size", "500"))
        if CONFIG.get("classifier.weight_classes"):
            logging.info("===> doing weighted rbfSVM")
            self.classifier = sklearn.svm.SVC(cache_size=c_size, class_weight="auto")
        else:
            logging.info("===> doing non-weighted rbfSVM")
            self.classifier = sklearn.svm.SVC(cache_size=c_size)
        self.scaler = sklearn.preprocessing.StandardScaler()


    def learn(self, decoy_peaks, target_peaks, use_main_score=True):
        X, y = self.format_train_data(decoy_peaks, target_peaks, use_main_score)
        if CONFIG.get("classifier.scale_subscores"):
            self.scaler = self.scaler.fit(X)
            X = self.scaler.transform(X)
        self.classifier.fit(X, y)
        #scores = np.dot(X, self.scalings)
        return self
     
     
    def score_by_matrix(self, feature_matrix):
        if CONFIG.get("classifier.scale_subscores"):
            feature_matrix = self.scaler.transform(feature_matrix)
        return self.classifier.decision_function(feature_matrix)
 
            
        


class PolySVMLearner(AbstractLearner):
            
    def __init__(self):
        AbstractLearner.__init__(self)
        c_size = int(CONFIG.get("classifier.cache_size", "500"))
        if CONFIG.get("classifier.weight_classes"):
            logging.info("===> doing weighted polySVM")
            self.classifier = sklearn.svm.SVC(cache_size=c_size, kernel="poly", class_weight="auto")
        else:
            logging.info("===> doing non-weighted polySVM")
            self.classifier = sklearn.svm.SVC(cache_size=c_size, kernel="poly")
        self.scaler = sklearn.preprocessing.StandardScaler()
            
        
    def learn(self, decoy_peaks, target_peaks, use_main_score=True):
        X, y = self.format_train_data(decoy_peaks, target_peaks, use_main_score)
        if CONFIG.get("classifier.scale_subscores"):
            self.scaler = self.scaler.fit(X)
            X = self.scaler.transform(X)
        self.classifier.fit(X, y)
        #scores = np.dot(X, self.scalings)
        return self
        
        
    def score_by_matrix(self, feature_matrix):
        if CONFIG.get("classifier.scale_subscores"):
            feature_matrix = self.scaler.transform(feature_matrix)
        return self.classifier.decision_function(feature_matrix)



class LogitLearner(LinearLearner):

    def __init__(self):
        AbstractLearner.__init__(self)
        if CONFIG.get("classifier.weight_classes"):
            logging.info("===> doing weighted logit")
            self.classifier = sklearn.linear_model.LogisticRegression(class_weight='auto')
        else:
            logging.info("===> doing non-weighted logit")
            self.classifier = sklearn.linear_model.LogisticRegression()
        self.scaler = sklearn.preprocessing.StandardScaler()


    def learn(self, decoy_peaks, target_peaks, use_main_score=True):
        X, y = self.format_train_data(decoy_peaks, target_peaks, use_main_score)
        if CONFIG.get("classifier.scale_subscores"):
            self.scaler = self.scaler.fit(X)
            X = self.scaler.transform(X)
        self.classifier.fit(X, y)
        self.coefs = self.classifier.coef_.flatten()
        #scores = np.dot(X, self.scalings)
        return self
    
     
    def simplistic(self):
        return not CONFIG.get("classifier.scale_subscores")
     
     
    def score_by_matrix(self, feature_matrix):
        if CONFIG.get("classifier.scale_subscores"):
            feature_matrix = self.scaler.transform(feature_matrix)
        return np.dot(feature_matrix, self.coefs)
