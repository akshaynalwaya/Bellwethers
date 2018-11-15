from __future__ import print_function, division
import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

import warnings
from oracle.models import nbayes, rf_model0
from py_weka.classifier import classify
from utils import *
from metrics.abcd import abcd
from metrics.recall_vs_loc import get_curve
from pdb import set_trace
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas
from plot.effort_plot import effort_plot
from tabulate import tabulate

import warnings

with warnings.catch_warnings():
    # Shut those god damn warnings up!
    warnings.filterwarnings("ignore")


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
        s = np.sum([1 if lo <= val < hi else 0 for lo, val, hi in zip(mins, train_set.ix[i].values, maxs)])
        w_i.append((k * s * mass) / (k - s + 1) ** 2)
    return w_i


def weight_training(weights, training_instance):
    weighted = []
    head = training_instance.columns
    for i in xrange(len(training_instance)):
        weighted.append([weights[i] * val for val in training_instance.ix[i].values[:-1]])
    new_train = pd.DataFrame(weighted, columns=head[:-1])
    new_train = (new_train - new_train.min()) / (new_train.max() - new_train.min())
    new_train[head[-1]] = training_instance[head[-1]]
    return new_train


def predict_defects(train, test, weka=True):
    actual = test[test.columns[-1]].values.tolist()
    actual = [1 if act == "T" else 0 for act in actual]
    if weka:
        train.to_csv(root + '/TNB/tmp/train.csv', index=False)
        test.to_csv(root + '/TNB/tmp/test.csv', index=False)

        __, distr = classify(train=os.path.abspath(root + '/TNB/tmp/train.csv'),
                             test=os.path.abspath(root + '/TNB/tmp/test.csv'),
                             name="rf", tuning=False)

        # set_trace()
        predicted = [1 if d > 0.6 else 0 for d in distr]

        # Remove temporary csv files to avoid conflicts
        os.remove(root + '/TNB/tmp/train.csv')
        os.remove(root + '/TNB/tmp/test.csv')

    else:
        # Binarize data
        # predicted, distr = nbayes(train, test)
        predicted, distr = rf_model0(train, test)
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
    plot_data =[("Xalan", "Log4j", "Lucene", "Poi", "Velocity")]
    for tgt_name, tgt_path in target.iteritems():
        stats = []
        charts = []
        if verbose: print("{} \r".format(tgt_name[0].upper() + tgt_name[1:]))
        val = []
        for src_name, src_path in source.iteritems():
            if not src_name == tgt_name:
                # print("{}  \r".format(src_name[0].upper() + src_name[1:]))
                src = list2dataframe(src_path.data)
                tgt = list2dataframe(tgt_path.data)
                pd, pf, g, auc = [], [], [], []
                lo, hi, test_mass = target_details(tgt)
                weights = get_weights(maxs=hi, mins=lo, train_set=src, test_set=tgt)
                _train = weight_training(weights=weights, training_instance=src)
                __test = (tgt[tgt.columns[:-1]] - tgt[tgt.columns[:-1]].min()) / (
                    tgt[tgt.columns[:-1]].max() - tgt[tgt.columns[:-1]].min())
                __test[tgt.columns[-1]] = tgt[tgt.columns[-1]]
                actual, predicted, distribution = predict_defects(train=_train, test=__test)
                loc = tgt["$loc"].values
                loc = loc * 100 / np.max(loc)
                recall, loc, au_roc = get_curve(loc, actual, predicted)
                effort_plot(recall, loc,
                            save_dest=os.path.abspath(os.path.join(root, "plot", "plots", tgt_name)),
                            save_name=src_name)
                p_d, p_f, p_r, rc, f_1, e_d, _g = abcd(actual, predicted, distribution)

                pd.append(p_d)
                pf.append(p_f)
                g.append(_g)
                auc.append(int(au_roc))
                stats.append([src_name, int(np.mean(pd)), int(np.std(pd)),
                              int(np.mean(pf)), int(np.std(pf)),
                              int(np.mean(auc)), int(np.std(auc))])  # ,
                # int(np.mean(g)), int(np.std(g))])
                # print("")
        stats = pandas.DataFrame(sorted(stats, key=lambda lst: lst[0]),  # Sort by G Score
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

    set_trace()
    return result


def tnb_jur():
    from data.handler import get_all_projects
    all = get_all_projects()
    apache = all["Apache"]
    return tnb(apache, apache, verbose=False, n_rep=1)


if __name__ == "__main__":
    tnb_jur()
