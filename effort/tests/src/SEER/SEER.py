from __future__ import print_function, division

import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from oracle.models import rf_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from numpy.random import choice
import numpy as np
from metrics.abcd import abcd
from matching.bahsic.hsic import CHSIC
from sklearn.decomposition import TruncatedSVD as KernelSVD
from utils import *
from pdb import set_trace
from time import time
import pandas
from py_weka.classifier import classify
from tabulate import tabulate


def get_kernel_matrix(dframe, n_components=5):
    """
    This returns a Kernel Transformation Matrix $\Theta$ -> This is basically a mapping using PCA-SVD
    :param dframe: input data as a pandas dataframe.
    :param n_dim: Number of dimensions for the kernel matrix
    :return: $\Theta$ matrix
    """
    kernel = KernelSVD(n_components=5)
    kernel.fit(dframe)
    return kernel


def map_transform(training_instance, test_instance, n_components=5):
    """
    Run a map and transform x and y onto a new space using PCA
    :param src: IID samples
    :param tgt: IID samples
    :return: Mapped x and y
    """
    head = training_instance.columns
    new_train = training_instance[head[:-1]]

    new_train = (new_train - test_instance[head[:-1]].min()) / (test_instance[head[:-1]].max() - test_instance[head[:-1]].min())
    new_train[head[-1]] = training_instance[head[-1]]
    try:
        new_test = (test_instance - test_instance[head[:-1]].min()) / (test_instance[head[:-1]].max() - test_instance[head[:-1]].min())
        new_test[head[-1]] = test_instance[head[-1]]
    except:
        set_trace()
    return new_train, new_test[new_train.columns]


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


def metrics_match(src, tgt, n_redo):
    s_col = [col for col in src.columns[:-1] if not '?' in col]
    t_col = [col for col in tgt.columns[:-1] if not '?' in col]
    selected_col = dict()
    for t_metric in t_col:
        hi = -1e32
        for s_metric in s_col:
            value = np.median([cause_effect(src[s_metric], tgt[t_metric]) for _ in xrange(n_redo)])
            if value > hi:
                selected_col.update({t_metric: (s_metric, value)})
                hi = value

    return selected_col


def predict_defects(train, test, weka=False):
    """

    :param train:
    :type train:
    :param test:
    :type test:
    :param weka:
    :type weka:
    :return:
    """

    # actual = test[test.columns[-1]].values.tolist()
    # actual = [1 if act == "T" else 0 for act in actual]
    #
    # if weka:
    #     train.to_csv('./tmp/train.csv', index=False)
    #     test.to_csv('./tmp/test.csv', index=False)
    #
    #     predicted = classify(train=os.path.abspath('./tmp/train.csv'),
    #                          test=os.path.abspath('./tmp/test.csv'),
    #                          name="rf")
    #
    #     # set_trace()
    #
    #     predicted = [1 if p > 0.5 else 0 for p in predicted]
    #
    #     # Remove temporary csv files to avoid conflicts
    #     os.remove('./tmp/train.csv')
    #     os.remove('./tmp/test.csv')
    # else:
    #     # Binarize data
    #     train.loc[train[train.columns[-1]] > 0, train.columns[-1]] = 1
    #     test.loc[test[test.columns[-1]] > 0, test.columns[-1]] = 1
    #     predicted, distr = rf_model(train, test)
    #
    # return actual, predicted, distr

    actual = test[test.columns[-1]].values.tolist()
    actual = [1 if act == "T" else 0 for act in actual]
    predicted, distr = rf_model(train, test)
    return actual, predicted, distr


def seer(source, target, verbose=False, n_rep=20, n_redo=5):
    """
    seer: Causal Inference Learning
    :param source:
    :param target:
    :return: result: A dictionary of estimated
    """
    result = dict()
    t0 = time()
    for tgt_name, tgt_path in target.iteritems():
        stats = []
        if verbose: print("{} \r".format(tgt_name[0].upper() + tgt_name[1:]))
        for src_name, src_path in source.iteritems():
            if not src_name == tgt_name:
                src = list2dataframe(src_path.data)
                tgt = list2dataframe(tgt_path.data)
                pd, pf, g, auc = [], [], [], []

                matched_src = metrics_match(src, tgt, n_redo)

                for n in xrange(n_rep):
                    target_columns = []
                    source_columns = []

                    all_columns = [(key, val[0], val[1]) for key, val in matched_src.iteritems() if val[1] > 1]
                    all_columns = sorted(all_columns, key=lambda x: x[-1], reverse=True)  # Sort descending

                    # Filter all columns to remove dupes
                    for elem in all_columns:
                        if not elem[1] in source_columns:
                            target_columns.append(elem[0])
                            source_columns.append(elem[1])
                    selected_col = list(set(target_columns).intersection(source_columns))
                    _train, __test = map_transform(src[selected_col + [src.columns[-1]]],
                                             tgt[selected_col + [tgt.columns[-1]]])

                    # _train, __test = src[source_columns + [src.columns[-1]]], \
                    #                  tgt[target_columns + [tgt.columns[-1]]]

                    # set_trace()
                    actual, predicted, distribution = predict_defects(train=_train, test=__test)
                    p_d, p_f, p_r, rc, f_1, e_d, _g, auroc = abcd(actual, predicted, distribution)
                    pd.append(p_d)
                    pf.append(p_f)
                    g.append(e_d)
                    auc.append(int(auroc))

                stats.append([src_name, int(np.mean(pd)), int(np.std(pd)),
                              int(np.mean(pf)), int(np.std(pf)),
                              int(np.mean(auc)), int(np.std(auc))])  # ,

        stats = pandas.DataFrame(sorted(stats, key=lambda lst: lst[-2], reverse=True),  # Sort by G Score
                                 columns=["Name", "Pd (Mean)", "Pd (Std)",
                                          "Pf (Mean)", "Pf (Std)",
                                          "AUC (Mean)", "AUC (Std)"])  # ,
        # "G (Mean)", "G (Std)"])
        if verbose: print(tabulate(stats,
                       headers=["Name", "Pd (Mean)", "Pd (Std)",
                                "Pf (Mean)", "Pf (Std)",
                                "AUC (Mean)", "AUC (Std)"],
                       showindex="never",
                       tablefmt="fancy_grid"))

        result.update({tgt_name: stats})
    return result



"""
Test case: Apache Datasets
"""


def seer_jur():
    from data.handler import get_all_projects
    all = get_all_projects()
    apache = all["Apache"]
    return seer(apache, apache, verbose=False, n_rep=10, n_redo=5)


if __name__ == "__main__":
    seer_jur()
