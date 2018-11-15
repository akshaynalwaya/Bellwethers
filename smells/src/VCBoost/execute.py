from __future__ import print_function, division

import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from utils import *
from metrics.abcd import abcd
import numpy as np
from collections import Counter
import pandas
from tabulate import tabulate
from random import random as rand, choice
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from datasets.handler2 import get_all_datasets


def target_details(test_set):
    """ Return Max and Min and 'Mass' from the test set """
    test_set = test_set[test_set.columns[:-1]]
    hi, lo = test_set.max().values, test_set.min().values
    return lo, hi


def get_weights(train_set, maxs, mins):
    s_i = []
    for i in xrange(len(train_set)):
        s_i.append((train_set.ix[i].values,
                    np.sum(
                        [1 if lo <= val < hi else 0 for lo, val, hi in
                         zip(mins, train_set.ix[i].values[:-1], maxs)]) / len(
                        train_set.columns[:-1])))
    return s_i


def weight_training(train, test, verbose=False):
    def train_validation_split():
        """ Split training data into X_train and X_validation"""
        sorted_train = sorted(train_w, key=lambda x: x[1], reverse=True)
        N = len(sorted_train)
        train0, validation = sorted_train[int(N * 2 / 5):], sorted_train[:int(N * 2 / 5)]
        return train0, validation

    def multiply_dframe(dframe, weight):
        assert len(weight) == len(dframe)
        N = len(dframe.columns) - 1
        wt_array = pd.DataFrame(np.array(N * [weight]).T, columns=dframe.columns[:-1])
        new_dframe = dframe.multiply(wt_array)
        new_dframe[dframe.columns[-1]] = dframe[dframe.columns[-1]]
        return new_dframe[dframe.columns].dropna(axis=1, inplace=False)

    def ensemble_measure(lst, classifiers, weigths):

        def norm_lst(lst):
            import numpy as np
            s = np.sum(lst)
            arr = np.array(lst) / s
            return np.nan_to_num(arr)

        tst = pd.DataFrame([t[0] for t in lst], columns=train.columns)
        X = tst[tst.columns[:-1]]
        y = tst[tst.columns[-1]]
        y_hat = []
        y_pred = []

        for clf in classifiers:
            y_hat.append(clf.decision_function(X))

        if len(y_hat) == 1:
            y = [1 if p == True else -1 for p in y]
            auc = roc_auc_score(y, y_hat[0])

        else:
            for pred, wgt in zip(y_hat, norm_lst(weigths)):
                y_pred.append([wgt * p for p in pred])

            y_pred = np.sum(np.array(y_pred).T, axis=1)
            y = [1 if p == True else -1 for p in y]
            try:
                auc = roc_auc_score(y, y_pred)
            except:
                set_trace()
        return auc

    def resample(train0, weights):
        """ The name says it all; resample training set"""

        def oversample(lst):
            new_lst = []
            while len(new_lst) < N:
                # set_trace()
                a = choice(lst)
                b = choice(lst)
                c = choice(lst)
                r = rand()
                new = [x + r * (y - z) for x, y, z in zip(a[0][0][:-1], b[0][0][:-1], c[0][0][:-1])] + [a[0][0][-1]]
                new_lst.append(((new, (a[0][1] + b[0][1] + c[0][1]) / 3), a[1] + r * (b[1] - c[1])))

            return new_lst

        def undersample(lst):
            return [choice(lst) for _ in xrange(len(lst))]

        klass = [t[0][-1] for t in train0]
        count = Counter(klass)
        # set_trace()
        [major, minor] = sorted(count)[::-1]
        N = int(0.5 * (count[minor] + count[major]))

        oversamp = []
        undersmp = []
        therest = []
        w_cutoff = np.median(weights)

        for w, b in zip(weights, train0):
            if b[1] <= w_cutoff and b[0][-1] is minor:
                oversamp.append((b, w))
            else:
                therest.append((b, w))

            if b[1] >= w_cutoff and b[0][-1] is major:
                undersmp.append((b, w))
            else:
                therest.append((b, w))
        try:
            therest.extend(undersample(undersmp))
            therest.extend(oversample(oversamp))
        except:
            pass

        weights = [t[1] for t in therest]
        therest = [t[0] for t in therest]
        return therest, weights

    lo, hi = target_details(test)
    train_w = get_weights(train, hi, lo)
    train0, validation = train_validation_split()
    rho_best = 0
    h = []
    a_m = []
    lam = 0.5  # Penalty for each iteration
    train1 = train0
    w = len(train1) * [1]
    a_best = a_m
    trn_best = train1
    for iter in xrange(5):
        if verbose: print("Interation number: {}".format(iter))
        train1, w = resample(train1, w)
        sim = [t[1] for t in train1]
        try:
            trn = pd.DataFrame([t[0] for t in train1], columns=train.columns)
        except:
            trn = pd.DataFrame([t for t in train1], columns=train.columns)
        w_trn = multiply_dframe(trn, w)
        # Create an SVM model
        X = w_trn[w_trn.columns[:-1]]
        y = w_trn[w_trn.columns[-1]]
        clf = LinearSVC()
        clf.fit(X, y)
        h.append(clf)
        y_prd = h[-1].predict(X)
        e_m = np.sum([w0 if not y_hat == y_act else 0 for w0, y_hat, y_act in zip(w, y_prd, y)]) / np.sum(
            w)
        a_m.append(lam * np.log((1 - e_m) / (e_m)))
        w = [w0 * np.exp(a_m[-1]) if not y_hat == y_act else w0 for w0, y_hat, y_act in zip(w, y_prd, y)]
        p_m = ensemble_measure(validation, h, a_m)
        if p_m >= rho_best:
            if verbose: print("Found better Rho. Previously: {0:.2f} | Now: {1:.2f}".format(rho_best, p_m))
            rho_best = p_m
            a_best = a_m
            trn_best = w_trn
    if verbose: print("Boosting terminated. Best Rho={}".format(rho_best))
    return trn_best, a_best, h


def predict_defects(test, weights, classifiers):
    def norm_lst(lst):
        import numpy as np
        s = np.sum(lst)
        arr = np.array(lst) / s
        return np.nan_to_num(arr)

    X = test[test.columns[:-1]]
    y = test[test.columns[-1]]
    y_hat = []
    y_pred = []

    for clf in classifiers:
        y_hat.append(clf.decision_function(X))

    if len(y_hat) == 1:
        actuals = [1 if p > 0 else 0 for p in y]
        distribution = y_hat[0]
        predicted = [1 if p > 0 else 0 for p in distribution]
    else:
        for pred, wgt in zip(y_hat, norm_lst(weights)):
            y_pred.append([wgt * p for p in pred])

        distribution = np.sum(np.array(y_pred).T, axis=1)
        set_trace()
        actuals = [1 if p is "T" else 0 for p in y]
        predicted = [1 if p > 0 else 0 for p in distribution]

    return actuals, predicted, distribution


def vcb(source, target, verbose=False, n_rep=12):
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
                    _train, clf_w, classifiers = weight_training(train=src, test=tgt)
                    actual, predicted, distribution = predict_defects(tgt, clf_w, classifiers)
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
        vcb(paths, paths, verbose=False, n_rep=10)
        print("\n\n")


if __name__ == "__main__":
    tnb_jur()
