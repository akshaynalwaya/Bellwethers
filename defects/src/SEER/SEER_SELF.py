from __future__ import print_function, division

import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from data.handler import get_all_projects
from prediction.model import rf_model
from old.stats import ABCD
from old.sk import rdivDemo
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from numpy.random import choice
import numpy as np
from matching.bahsic.hsic import CHSIC
from utils import *


def cause_effect(x, y):
    """
    Run a non-parametric cause-effect test
    :param x: IID samples
    :param y: IID samples
    :return: A tuple (sign, delta-HSI). If X->Y: (+1, c_xy-c_yx) or if Y->X:(-1, ) else 0
    """

    def pack():
        """
        Split data into train test and pack them as tuples
        :return:
        """

        N = int(min(len(x), len(y)))
        "Make sure x0, y0 are the same size. Note: This is rerun multiple times."
        if len(x) > len(y):
            x0, y0 = choice(x, size=N), y
        elif len(x) < len(y):
            x0, y0 = x, choice(y, size=N)
        else:
            x0, y0 = x, y

        # Defaults to a 0.75/0.25 split.
        x_train, y_train, x_test, y_test = train_test_split(x0, y0)

        train = [(a, b) for a, b in zip(x_train, y_train)]
        test = [(a, b) for a, b in zip(x_test, y_test)]

        return train, test

    def unpack(lst, axis):
        return np.atleast_2d([l[axis] for l in lst]).T

    def residue(train, test, fwd=True):
        mdl = LinearRegression()
        if fwd:
            X = unpack(train, axis=0)
            y = unpack(train, axis=1)
            x_hat = unpack(test, axis=0)
            y_hat = unpack(test, axis=1)
        else:
            X = unpack(train, axis=1)
            y = unpack(train, axis=0)
            x_hat = unpack(test, axis=1)
            y_hat = unpack(test, axis=0)

        mdl.fit(X, y)
        return y_hat - mdl.predict(x_hat)

    train, test = pack()
    e_y = residue(train, test, fwd=True)
    e_x = residue(train, test, fwd=False)
    x_val = unpack(test, axis=0)
    y_val = unpack(test, axis=1)
    hsic = CHSIC()
    c_xy = hsic.UnBiasedHSIC(x_val, e_y)
    c_yx = hsic.UnBiasedHSIC(y_val, e_x)

    return abs(c_xy - c_yx) if c_xy < c_yx else 0


def metrics_match(src, tgt):
    s_col = [col for col in src.columns[:-1] if not '?' in col]
    t_col = [col for col in tgt.columns[:-1] if not '?' in col]
    selected_col = dict()
    for t_metric in t_col:
        hi = -1e32
        for s_metric in s_col:
            value = np.median([cause_effect(src[s_metric], tgt[t_metric]) for _ in xrange(3)])
            if value > hi:
                selected_col.update({t_metric: (s_metric, value)})
                hi = value

    return selected_col


def predict_defects(train, test):
    """

    :param train:
    :param test:
    :return:
    """
    # Binarize data
    train.loc[train[train.columns[-1]] > 0, train.columns[-1]] = 1
    test.loc[test[test.columns[-1]] > 0, test.columns[-1]] = 1

    actual = test[test.columns[-1]].values.tolist()

    predicted = rf_model(train, test)
    return actual, predicted


def seer0(source, target):
    """
    seer0: Causal Inference Learning
    :param source:
    :param target:
    :return:
    """
    for tgt_name, tgt_path in target.iteritems():
        PD, PF, ED = [], [], []
        print("Target Project: {}\n".format(tgt_name), end="```\n")
        for src_name, src_path in source.iteritems():
            if not src_name == tgt_name:
                src = list2dataframe(src_path.data)
                tgt = list2dataframe(tgt_path.data)
                pd, pf, ed = [src_name], [src_name], [src_name]
                for n in xrange(1):

                    matched_src = metrics_match(src, tgt)

                    target_columns = []
                    source_columns = []

                    all_columns = [(key, val[0], val[1]) for key, val in matched_src.iteritems() if val[1] > 1]
                    all_columns = sorted(all_columns, key=lambda x: x[-1])[::-1]  # Sort descending

                    # Filter all columns to remove dupes
                    for elem in all_columns:
                        if not elem[1] in source_columns:
                            target_columns.append(elem[0])
                            source_columns.append(elem[1])

                    _train = df_norm(src[source_columns + [src.columns[-1]]])
                    __test = df_norm(tgt[target_columns + [tgt.columns[-1]]])

                    actual, predicted = predict_defects(train=_train, test=__test)
                    p_buggy = [a for a in ABCD(before=actual, after=predicted)()]
                    pd.append(p_buggy[1].stats()[0])
                    pf.append(p_buggy[1].stats()[1])
                    ed.append(p_buggy[1].stats()[-1])

                PD.append(pd)
                PF.append(pf)
                ED.append(ed)
                # set_trace()
        rdivDemo(ED, isLatex=False)
        print('```')


def execute():
    """
    This method performs HDP.
    :return:
    """
    all_projects = get_all_projects()  # Get a dictionary of all projects and their respective pathnames.
    result = {}  # Store results here

    for target in all_projects.keys():
        for source in all_projects.keys():
            if source == target:  # This ensures transfer happens within community
                print("Target Community: {} | Source Community: {}".format(target, source))
                seer0(all_projects[source], all_projects[target])


def __test_seer():
    """
    A test case goes here.
    :return:
    """


if __name__ == "__main__":
    execute()
