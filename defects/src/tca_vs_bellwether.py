"""
Compares TCA with Bellwether Method (SEER to be added)
"""
from __future__ import print_function, division

import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from old.rq1 import jur
from TCA.execute import tca_jur
import multiprocessing
from pdb import set_trace
from stats.effect_size import hedges_g_2
import pandas
from utils import print_pandas


# Note To Self: REFACTOR THE TRANSFER LEARNER

def compute(method):
    return method()


if __name__ == "__main__":
    methods = [jur, tca_jur]
    pool = multiprocessing.Pool(processes=2)
    a = pool.map(compute, methods)
    projects = a[0].keys()

    for p in projects:
        print("\\textbf{" + str(p) + "}")
        bell = a[0][p].set_index("Name").sort_index()  # Rename index and sort alphabetically
        tca = a[1][p].set_index("Name").sort_index()  # Rename index and sort alphabetically
        both = pandas.concat([tca, bell], axis=1, join_axes=[tca.index])
        all_metrics = hedges_g_2(both)
        print_pandas(all_metrics.set_index("Name"))
        print("\n\n")
