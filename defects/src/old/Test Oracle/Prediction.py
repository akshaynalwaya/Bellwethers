from __future__ import division

from sklearn.ensemble import RandomForestClassifier

from methods1 import *
from smote import *


def formatData(tbl):
    """ Convert Tbl to Pandas DataFrame

    :param tbl: Thing object created using function createTbl
    :returns table in a DataFrame format
    """
    Rows = [i.cells for i in tbl._rows]
    headers = [i.name for i in tbl.headers]
    return pd.DataFrame(Rows, columns=headers)


def Bugs(tbl):
    cells = [i.cells[-2] for i in tbl._rows]
    return cells


def rforest(train, test, tunings=None):
    """ Random Forest

    :param train:   Thing object created using function createTbl
    :param test:    Thing object created using function createTbl
    :param tunings: List of tunings obtained from Differential Evolution
                    tunings=[n_estimators, max_features, min_samples_leaf, min_samples_split]
    :return preds: Predicted bugs
    """

    assert type(train) is Thing, "Train is not a Thing object"
    assert type(test) is Thing, "Test is not a Thing object"
    train = SMOTE(train, atleast=50, atmost=101, resample=True)
    if not tunings:
        clf = RandomForestClassifier(n_estimators=100, random_state=1)
    else:
        clf = RandomForestClassifier(n_estimators=int(tunings[0]),
                                     max_features=tunings[1] / 100,
                                     min_samples_leaf=int(tunings[2]),
                                     min_samples_split=int(tunings[3]))

    train_DF = formatData(train)
    test_DF = formatData(test)
    features = train_DF.columns[:-2]
    klass = train_DF[train_DF.columns[-2]]
    clf.fit(train_DF[features], klass)
    preds = clf.predict(test_DF[test_DF.columns[:-2]])
    return preds


def _RF():
    "Test RF"
    dir = 'Data/Jureczko'
    one, two = explore(dir)
    train, test = createTbl(one[0]), createTbl(two[0])
    actual = Bugs(test)
    predicted = rforest(train, test)
    set_trace()


if __name__ == '__main__':
    _RF()
