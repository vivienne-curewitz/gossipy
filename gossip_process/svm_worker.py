import sklearn
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import numpy as np
from random import randint
import pickle

def copy_sgd(s1):
    new_sgd = SGDClassifier(
        loss=s1.loss,
        alpha=s1.alpha,
        learning_rate=s1.learning_rate,
        eta0=s1.eta0,
    )
    new_sgd.coef_ = s1.coef_
    new_sgd.intercept_ = s1.intercept_
    new_sgd.classes_ = s1.classes_
    new_sgd.t_ = s1.t_
    new_sgd.n_iter_ = s1.n_iter_
    return new_sgd


class SVMWorker:
    def __init__(self, data, inqueue, outqueue):
        self.iters = 1
        self.xtrain = data.iloc[:, :-1].values
        self.ytrain = data.iloc[:, -1].values
        self.svm = self.init_primal_svm()
        self.inqueue = inqueue
        self.outqueue = outqueue

    @property
    def model(self):
        return self.pack_model()

    def run(self):
        print("Starting SVM worker loop...")
        self.outqueue.put(self.model)  # send initial model
        while True:
            s2 = self.inqueue.get()
            self.combine_packed(s2)
            self.svm.partial_fit(self.xtrain, self.ytrain)
            self.iters += 1
            packed = self.model
            self.outqueue.put(packed)

    def pack_model(self):
        output = (
            self.svm.coef_.tolist(),
            self.svm.intercept_.tolist(),
            self.svm.t_,
            self.svm.n_iter_,
        )
        return output


    # test the combination
    def init_primal_svm(self):
        primal = SGDClassifier(loss='hinge', alpha=0.0001, learning_rate="optimal")
        primal.partial_fit(self.xtrain, self.ytrain, np.array([0,1]))
        return primal
    
    def combine_sgds(self, s2):
        w_merged = (self.svm.coef_ + s2.coef_)/2 # double check learning rates later
        b_merged = (self.svm.intercept_ + s2.intercept_)/2
        self.svm.coef_ = w_merged
        self.svm.intercept_ = b_merged
        self.svm.t_ = max(self.svm.t_, s2.t_)
        self.svm.n_iter_ = max(self.svm.n_iter_, s2.n_iter)

    def combine_packed(self, packed):
        w2, b2, t2, n2 = packed
        w_merged = (self.svm.coef_ + np.array(w2))/2
        b_merged = (self.svm.intercept_ + np.array(b2))/2
        self.svm.coef_ = w_merged
        self.svm.intercept_ = b_merged
        self.svm.t_ = max(self.svm.t_, t2)
        self.svm.n_iter_ = max(self.svm.n_iter_, n2)

    def infer(self, xdata):
        x_array = np.array(xdata)
        preds = self.svm.predict(x_array)
        return preds.tolist()


def scale_df(df):
    scaler = StandardScaler()
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    xs = scaler.fit_transform(x)
    feature_cols = df.columns[:-1]   # all but last column
    label_col = df.columns[-1]       # last column
    
    df_scaled = pd.DataFrame(xs, columns=feature_cols)
    df_scaled[label_col] = y.values
    return df_scaled

