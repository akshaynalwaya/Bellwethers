from __future__ import division

import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from smote import SMOTE


import warnings

with warnings.catch_warnings():
    # Shut those god damn warnings up!
    warnings.filterwarnings("ignore")

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from old.Prediction import rforest

from data.handler import *

import warnings

with warnings.catch_warnings():
    # Shut those god damn warnings up!
    warnings.filterwarnings("ignore")


def getTunings(fname):
    raw = pd.read_csv(root + '/old/tunings.csv').transpose().values.tolist()
    formatd = pd.DataFrame(raw[1:], columns=raw[0])
    try:
        return formatd[fname].values.tolist()
    except KeyError:
        return None


def df2thing(dframe):
    dframe.to_csv('temp.csv', index=False)
    new_thing = createTbl(['temp.csv'], isBin=True)
    os.remove('temp.csv')
    return new_thing


def rf_model(source, target):
    clf = RandomForestClassifier(n_estimators=100, random_state=1)
    # Binarize source
    # source.loc[source[source.columns[-1]] > 0, source.columns[-1]] = 1
    source = SMOTE(source)
    features = source.columns[:-1]
    klass = source[source.columns[-1]]
    clf.fit(source[features], klass)
    preds = clf.predict(target[target.columns[:-1]])
    distr = clf.predict_proba(target[target.columns[:-1]])
    return preds, distr[:,1]


def rf_model_old(source, target):

    def df2thing(dframe):
        dframe.to_csv('temp.csv', index=False)
        new_thing = createTbl(['temp.csv'], isBin=True)
        os.remove('temp.csv')
        return new_thing

    train = df2thing(source)
    test = df2thing(target)

    return rforest(train, test)



def rf_model0(source, target):
    train = df2thing(source)
    test = df2thing(target)
    return rforest(train, test, tunings=None)


def nbayes(source, target):
    """ Naive Bayes Classifier
    """
    clf = GaussianNB()
    source.loc[source[source.columns[-1]] > 0, source.columns[-1]] = 1
    source.loc[source[source.columns[-1]] < 1, source.columns[-1]] = 0
    # set_trace()
    # source = SMOTE(source, k=1)
    # set_trace()
    features = source.columns[:-1]
    klass = source[source.columns[-1]]
    clf.fit(source[features], klass)
    preds = clf.predict(target[target.columns[:-1]])
    distr = clf.predict_proba(target[target.columns[:-1]])
    return preds, distr[:,1]

def _test_model():
    pass


if __name__ == '__main__':
    _test_model()
