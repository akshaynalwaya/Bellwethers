from __future__ import division

import os
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from smote import SMOTE

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)


def nbayes(source, target):
    """
    Naive Bayes
    :param source:
    :param target:
    :return:
    """
    clf = GaussianNB()
    # Binarize source
    source.loc[source[source.columns[-1]] == True, source.columns[-1]] = 1
    source.loc[source[source.columns[-1]] == False, source.columns[-1]] = 0
    target.loc[target[target.columns[-1]] == True, target.columns[-1]] = 1
    target.loc[target[target.columns[-1]] == False, target.columns[-1]] = 0

    features = source.columns[:-1]
    klass = source[source.columns[-1]]
    clf.fit(source[features], klass)
    preds = clf.predict(target[target.columns[:-1]])
    distr = clf.predict_proba(target[target.columns[:-1]])
    return preds, distr[:, 1]


def rf_model(source, target):
    """
    Random Forest
    :param source:
    :param target:
    :return:
    """
    clf = RandomForestClassifier(n_estimators=90, random_state=1)

    # Binarize source
    source.loc[source[source.columns[-1]] == True, source.columns[-1]] = 1
    source.loc[source[source.columns[-1]] == False, source.columns[-1]] = 0
    target.loc[target[target.columns[-1]] == True, target.columns[-1]] = 1
    target.loc[target[target.columns[-1]] == False, target.columns[-1]] = 0

    features = source.columns[:-1]
    klass = source[source.columns[-1]]
    clf.fit(source[features], klass)
    preds = clf.predict(target[target.columns[:-1]])
    distr = clf.predict_proba(target[target.columns[:-1]])
    return preds, distr[:, 1]


def logistic_model(source, target):
    """
    Logistic Regression
    :param source:
    :param target:
    :return:
    """
    # Binarize source
    clf = LogisticRegression()
    source.loc[source[source.columns[-1]] > 0, source.columns[-1]] = 1
    source.loc[source[source.columns[-1]] < 1, source.columns[-1]] = 0
    source = SMOTE(source, k=1)
    features = source.columns[:-1]
    klass = [1 if k > 0 else 0 for k in source[source.columns[-1]]]
    clf.fit(source[features], klass)
    preds = clf.predict(target[target.columns[:-1]])
    distr = clf.predict_proba(target[target.columns[:-1]])
    return preds, distr[:, 1]


def _test_model():
    pass


if __name__ == '__main__':
    _test_model()
