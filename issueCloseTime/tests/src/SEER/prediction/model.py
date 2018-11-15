from __future__ import division
import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from smote import SMOTE

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from StringIO import StringIO
from old.Prediction import rforest
from data.handler import *


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

# def df2thing(dframe):
#     output = StringIO()
#     dframe.to_csv(output)
#     output.seek(0)
#     set_trace()
#     new_thing = createTbl([output], isBin=True)
#     return new_thing


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


def rf_model0(source, target):
    train = df2thing(source)
    test = df2thing(target)
    return rforest(train, test, smote=False)


#
# def logistic_model(source, target):
#     train = df2thing(source)
#     test = df2thing(target)
#     return logistic_regression(train, test)


def _test_model():
    pass


if __name__ == '__main__':
    _test_model()
