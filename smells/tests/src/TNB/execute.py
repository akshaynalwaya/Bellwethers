from __future__ import print_function, division

import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from oracle.models import nbayes
from utils import *
from metrics.abcd import abcd
from pdb import set_trace
import numpy as np
import pandas
from tabulate import tabulate
from datasets.handler2 import get_all_datasets


def target_details(test_set):
    """ Return Max and Min and 'Mass' from the test set """
    test_set = test_set[test_set.columns[3:-1]]
    hi, lo = test_set.max().values, test_set.min().values
    mass = test_set.size
    return lo, hi, mass


def get_weights(train_set, test_set, maxs, mins):
    train_set = train_set[train_set.columns[:-1]]
    mass = len(test_set)
    k = len(train_set.columns)
    w_i = []
    for i in xrange(len(train_set)):
        s = np.sum([1 if lo <= val < hi else 0 for lo, val, hi in zip(mins, train_set.ix[i].values, maxs)]) / k
        w_i.append((k * s * mass) / (k - s + 1) ** 2)
    return w_i


def weight_training(weights, training_instance, test_instance):
    weighted = []
    head = training_instance.columns
    for i in xrange(len(training_instance)):
        weighted.append([weights[i] * val for val in training_instance.ix[i].values[:-1]])
    new_train = pd.DataFrame(weighted, columns=head[:-1])
    new_train = (new_train - new_train.min()) / (new_train.max() - new_train.min())
    new_train[head[-1]] = training_instance[head[-1]]
    new_train.dropna(axis=1, inplace=True)
    tgt = new_train.columns
    new_test = (test_instance[tgt[:-1]] - test_instance[tgt[:-1]].min()) / (
        test_instance[tgt[:-1]].max() - test_instance[tgt[:-1]].min())

    new_test[tgt[-1]] = test_instance[tgt[-1]]
    new_test.dropna(axis=1, inplace=True)
    columns = list(set(tgt[:-1]).intersection(new_test.columns[:-1])) + [tgt[-1]]
    return new_train[columns], new_test[columns]


def predict_defects(train, test):
    """
    Prediction scheme
    """
    actual = test[test.columns[-1]].values.tolist()
    predicted, distr = nbayes(train, test)
    return actual, predicted, distr


def tnb(source, target, verbose=False, n_rep=12):
    """
    TNB: Transfer Naive Bayes
    :param source:
    :param target:
    :param n_rep: number of repeats
    :return: result
    """
    result = dict()
    for tgt_name, tgt_path in target.iteritems():
        stats = []
        if verbose: print("{} \r".format(tgt_name[0].upper() + tgt_name[1:]))
        val = []
        for src_name, src_path in source.iteritems():
            if not src_name == tgt_name:
                src = pandas.read_csv(src_path)
                tgt = pandas.read_csv(tgt_path)
                pd, pf, pr, f1, g, auc = [], [], [], [], [], []
                for _ in xrange(n_rep):
                    lo, hi, test_mass = target_details(tgt)
                    weights = get_weights(maxs=hi, mins=lo, train_set=src, test_set=tgt)
                    _train, __test = weight_training(weights=weights, training_instance=src, test_instance=tgt)
                    actual, predicted, distribution = predict_defects(train=_train, test=__test)
                    p_d, p_f, p_r, rc, f_1, e_d, _g, auroc = abcd(actual, predicted, distribution)

                    pd.append(p_d)
                    pf.append(p_f)
                    pr.append(p_r)
                    f1.append(f_1)
                    g.append(_g)
                    auc.append(int(auroc))

                stats.append([src_name, int(np.mean(pd)), int(np.mean(pf)),
                              int(np.mean(pr)), int(np.mean(f1)),
                              int(np.mean(g)), int(np.mean(auc))])  # ,

        stats = pandas.DataFrame(sorted(stats, key=lambda lst: lst[-2], reverse=True),  # Sort by G Score
                                 columns=["Name", "Pd", "Pf", "Prec", "F1", "G", "AUC"])  # ,

        if verbose: print(tabulate(stats,
                       headers=["Name", "Pd", "Pf", "Prec", "F1", "G", "AUC"],
                       showindex="never",
                       tablefmt="fancy_grid"))

        result.update({tgt_name: stats})

    return result


def tnb_jur():
    all = get_all_datasets()
    for name, paths in all.iteritems():
        tnb(paths, paths, verbose=False, n_rep=10)
        print("\n\n")
        # set_trace()


if __name__ == "__main__":
    tnb_jur()
