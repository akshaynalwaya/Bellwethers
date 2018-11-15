from __future__ import print_function, division

import numpy as np
from pdb import set_trace

def get_curve(loc, actual, predicted):
    sorted_loc = np.array(loc)[np.argsort(loc)]
    sorted_act = np.array(actual)[np.argsort(loc)]

    sorted_prd = np.array(predicted)[np.argsort(loc)]
    recall, loc = [], []
    tp, fn, Pd = 0, 0, 0
    for a, p, l in zip(sorted_act, sorted_prd, sorted_loc):
        tp += 1 if a == 1 and p == 1 else 0
        fn += 1 if a == 1 and p == 0 else 0
        Pd = tp / (tp + fn) if (tp + fn) > 0 else Pd
        loc.append(l)
        recall.append(int(Pd * 100))

    auc = np.trapz(recall, loc) / 100
    return recall, loc, auc
