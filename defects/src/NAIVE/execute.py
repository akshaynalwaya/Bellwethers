from __future__ import print_function, division

import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src/defects')
if root not in sys.path:
    sys.path.append(root)

from prediction.model import rf_model
from utils import *
from metrics.abcd import abcd
import numpy as np
import pandas
from tabulate import tabulate


def weight_training(test_instance, training_instance):
    head = training_instance.columns
    new_train = training_instance[head[:-1]]
    new_train = (new_train - test_instance[head[:-1]].mean()) / test_instance[head[:-1]].std()
    new_train[head[-1]] = training_instance[head[-1]]
    new_train.dropna(axis=1, inplace=True)
    tgt = new_train.columns
    new_test = (test_instance[tgt[:-1]] - test_instance[tgt[:-1]].mean()) / (
        test_instance[tgt[:-1]].std())

    new_test[tgt[-1]] = test_instance[tgt[-1]]
    new_test.dropna(axis=1, inplace=True)
    columns = list(set(tgt[:-1]).intersection(new_test.columns[:-1])) + [tgt[-1]]
    return new_train[columns], new_test[columns]


def predict_defects(train, test):
    actual = test[test.columns[-1]].values.tolist()
    actual = [1 if act == "T" else 0 for act in actual]
    predicted, distr = rf_model(train, test)
    return actual, predicted, distr


def bellw(source, target, n_rep=12, verbose=False):
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
        charts = []
        print("{} \r".format(tgt_name[0].upper() + tgt_name[1:]))
        val = []
        for src_name, src_path in source.iteritems():
            if not src_name == tgt_name:

                src = list2dataframe(src_path.data)
                tgt = list2dataframe(tgt_path.data)

                pd, pf, pr, f1, g, auc = [], [], [], [], [], []
                for _ in xrange(n_rep):
                    _train, __test = weight_training(test_instance=tgt, training_instance=src)
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
    from data.handler import get_all_projects
    all = get_all_projects()
    # set_trace()
    apache = all["Apache"]
    return bellw(apache, apache, n_rep=10)


if __name__ == "__main__":
    tnb_jur()
