from __future__ import division

import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)


def getTunings(fname):
    raw = pd.read_csv(root + '/old/tunings.csv').transpose().values.tolist()
    formatd = pd.DataFrame(raw[1:], columns=raw[0])
    try:
        return formatd[fname].values.tolist()
    except KeyError:
        return None


def rf_model(source, target):
    # Binarize source
    clf = RandomForestRegressor(n_estimators=100, random_state=1)
    features = source.columns[:-1]
    klass = source[source.columns[-1]]
    clf.fit(source[features], klass)
    preds = clf.predict(target[target.columns[:-1]])
    return preds


def rf_model0(source, target, name):
    return rf_model(source, target)


def linear_model(source, target):
    # Binarize source
    clf = LinearRegression()
    features = source.columns[:-1]
    klass = source[source.columns[-1]]
    clf.fit(source[features], klass)
    preds = clf.predict(target[target.columns[:-1]])
    return preds


def bayesRegr(source, target):
    # Binarize source
    clf = BayesianRidge()
    features = source.columns[:-1]
    klass = source[source.columns[-1]]
    clf.fit(source[features], klass)
    preds = clf.predict(target[target.columns[:-1]])
    return preds


def _test_model():
    pass


if __name__ == '__main__':
    _test_model()
