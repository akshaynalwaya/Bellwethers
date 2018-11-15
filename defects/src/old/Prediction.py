from __future__ import division

from sklearn.ensemble import RandomForestClassifier
from random import randint, shuffle
from methods1 import *
from smote import *


def formatdata(tbl, picksome=False, addsome=None):
    """ Convert Tbl to Pandas DataFrame

    :param tbl: Thing object created using function createTbl
    :returns table in a DataFrame format
    """
    some = []
    Rows = [i.cells for i in tbl._rows]
    if picksome:
        shuffle(Rows)
        for i in xrange(int(len(Rows)*0.1)):
            some.append(Rows.pop())
    headers = [i.name+"_"+str(randint(0,10000)) for i in tbl.headers]

    if addsome:
        Rows.extend(addsome)

    if picksome:
        return pd.DataFrame(Rows), some
    else:
        return pd.DataFrame(Rows)


def Bugs(tbl):
    cells = [i.cells[-2] for i in tbl._rows]
    return cells


def rforest(train, test, smote=True, tunings=None, picksome=False):
    """ Random Forest

    :param train:   Thing object created using function createTbl
    :param test:    Thing object created using function createTbl
    :param tunings: List of tunings obtained from Differential Evolution
                    tunings=[n_estimators, max_features, min_samples_leaf, min_samples_split]
    :return preds: Predicted bugs
    """

    assert type(train) is Thing, "Train is not a Thing object"
    assert type(test) is Thing, "Test is not a Thing object"
    if smote:
        train = SMOTE(train, atleast=50, atmost=101, resample=True)
    if not tunings:
        clf = RandomForestClassifier(n_estimators=100, random_state=1)
    else:
        clf = RandomForestClassifier(n_estimators=int(tunings[0]),
                                     max_features=tunings[1] / 100,
                                     min_samples_leaf=int(tunings[2]),
                                     min_samples_split=int(tunings[3]))

    if picksome:
        train_DF, some = formatdata(train, True)
        test_DF = formatdata(test, False, some)
    else:
        train_DF = formatdata(train, False, None)
        test_DF = formatdata(test, False, None)

    set_trace()
    features = train_DF.columns[:-2]
    klass = train_DF[train_DF.columns[-2]]
    clf.fit(train_DF[features], klass)
    try:
        preds = clf.predict(test_DF[test_DF.columns[:-2]])
    except:
        set_trace()

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
