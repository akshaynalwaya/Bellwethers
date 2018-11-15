from __future__ import print_function, division

import numpy as np
from sklearn.metrics import *
from pdb import set_trace


def get_curve(loc, actual, predicted, distribution):
    sorted_loc = np.array(loc)[np.argsort(loc)]
    sorted_act = np.array(actual)[np.argsort(loc)]

    try:
        fpr, tpr, thresholds = roc_curve(actual, distribution)
        for a, b, c in zip(fpr, tpr, thresholds):
            if a < 0.31:
                threshold = c
        predicted = [1 if val>threshold else 0 for val in distribution]
    except:
        pass
        # predicted = [1 if val is "T" else 0 for val in predicted]

    sorted_prd = np.array(predicted)[np.argsort(loc)]
    recall, loc = [], []
    tp, fn, Pd = 0, 0, 0
    for a, p, l in zip(sorted_act, sorted_prd, sorted_loc):
        tp += 1 if (a == 1 and p == 1) or (a == "T" and p == "T") else 0
        fn += 1 if (a == 1 and p == 0) or (a == "T" and p == "F") else 0
        Pd = tp / (tp + fn) if (tp + fn) > 0 else Pd
        loc.append(l)
        recall.append(int(Pd * 100))
    auc = np.trapz(recall, loc) / 100
    # set_trace()
    return recall, loc, auc
