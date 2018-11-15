from __future__ import division, print_function
import os
import sys
from pdb import set_trace
from KSAnalyzer import KSAnalyzer
from filter import weightedBipartite
import pandas as pd


def list2dataframe(lst):
    data = [pd.read_csv(elem) for elem in lst]
    return pd.concat(data, ignore_index=True)


def match_metrics(source, target):
    source = list2dataframe(source.data)
    target = list2dataframe(target.data)
    matches = KSAnalyzer(source, target, cutoff=0.05)
    wbp = weightedBipartite(matches, source=source.columns, target=target.columns)
    if len(wbp)>0:
        return wbp
    else:
        return None
