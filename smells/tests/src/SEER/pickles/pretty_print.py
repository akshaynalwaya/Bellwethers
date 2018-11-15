from __future__ import print_function
import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from stats.scott_knott import rdivDemo
from utils import load_json
from glob import glob
from pdb import set_trace
from sklearn.metrics import auc
import numpy as np


def auc_temp(x, y):
    # auc0 = [np.sqrt(0.7*(1-xx)**2+0.3*(yy)**2) for xx, yy in zip(x,y)]
    # return auc(x, y)
    try:
        g_measure = [2* x_0 * y_0 / (x_0 + y_0) for x_0, y_0 in zip(x, y)]
        return np.mean(g_measure)  # , np.std(g_measure)
    except ZeroDivisionError:
        return 0
        # return np.mean(x), np.mean(y)


def print_sk_charts():
    files = glob(os.path.join(os.path.abspath("./"), '*.json'))
    for f in files:
        data = load_json(f)
        for key, value in data.iteritems():
            print("## {}\n\nname \t Pd \t Pf \t G".format(key))
            stats = []

            for pd, pf in zip(value[0][0], value[0][1]):
                # set_trace()
                stats.append((pd[0], np.median(pd[1:]), np.median(pf[1:]), auc_temp(pd[1:], pf[1:])))

            stats = sorted(stats, key=lambda x: x[-1], reverse=True)

            for v in stats:
                star = "*" if v[1]>0.6 and v[2] <= 0.5 else ""
                name = v[0][:4] if len(v[0]) >=3 else v[0]+(3-len(v[0]))*" "
                print(name+star+' \t ','{0:.2f} \t {1:.2f} \t {2:.2f}'.format(v[1], v[2], v[-1]), sep="")

    return


if __name__ == "__main__":
    print_sk_charts()
