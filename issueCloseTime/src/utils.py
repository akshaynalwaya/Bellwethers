from __future__ import print_function, division

import os
import sys
from random import shuffle

import pandas as pd

try:
    import cPickle as pickle
except ImportError:
    import pickle

import json
from StringIO import StringIO
import prettytable
from pdb import set_trace

__root__ = os.path.join(os.getcwd().split('HDP')[0], 'HDP')
if __root__ not in sys.path:
    sys.path.append(__root__)

"""
Some awesome utility functions used everywhere
"""


def stringify_pandas(pd):
    """

    :param pd: A Dataframe
    :return:
    """
    output = StringIO()
    pd.to_csv(output)

    pt = prettytable.from_csv(output)
    print(pt)


def print_pandas(pd):
    """
    Prints a dataframe as latex
    :param pd: A Dataframe
    :return:
    """

    prefix = "\\begin{figure}[hp!]\n\\centering\n\\resizebox{\\textwidth}{!}{"
    postfix = "}\n\\end{figure}"
    body = pd.to_latex()
    print(prefix + body + postfix)


def list2dataframe(lst):
    if isinstance(lst, list):
        data = [pd.read_csv(elem) for elem in lst]
        return pd.concat(data, ignore_index=True)
    elif isinstance(lst, str) and os.path.isfile(lst):
        return pd.read_csv(lst)
    else:
        raise TypeError("This is not a path")


def dump_json(result, dir='.', fname='data'):
    with open('{}/{}.json'.format(dir, fname), 'w+') as fp:
        json.dump(result, fp)


def load_json(path):
    with open(path) as data_file:
        data = json.load(data_file)

    return data


def brew_pickle(data, dir='.', fname='data'):
    with open('{}/{}.p'.format(dir, fname), 'w+') as fp:
        pickle.dump(data, fp)


def load_pickle(path):
    return pickle.load(open(path, 'rb'))


def flatten(x):
    """
    Takes an N times nested list of list like [[a,b],[c, [d, e]],[f]]
    and returns a single list [a,b,c,d,e,f]
    """
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def df_norm(dframe, type="normal"):
    """ Normalize a dataframe"""
    col = dframe.columns
    bugs = dframe[dframe.columns[-1]]
    if type == "min_max":
        dframe = (dframe - dframe.min()) / (dframe.max() - dframe.min() + 1e-32)
        dframe[col[-1]] = bugs
        return dframe[col]

    if type == "normal":
        dframe = (dframe - dframe.mean()) / dframe.std()
        dframe[col[-1]] = bugs
        return dframe[col]


def explore(dir):
    datasets = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        datasets.append(dirpath)

    dataset = []
    for k in datasets[1:]:
        data0 = [[dirPath, fname] for dirPath, _, fname in os.walk(k)]
        dataset.append(
            [data0[0][0] + '/' + p for p in data0[0][1] if not p == '.DS_Store'])
    return dataset


def formatData(tbl, picksome=False, addsome=None):
    """ Convert Tbl to Pandas DataFrame

    :param tbl: Thing object created using function createTbl
    :returns table in a DataFrame format
    """
    some = []
    Rows = [i.cells for i in tbl._rows]
    if picksome:
        shuffle(Rows)
        for i in xrange(int(len(Rows) * 0.1)):
            some.append(Rows.pop())
    headers = [i.name for i in tbl.headers]

    if addsome:
        Rows.extend(addsome)

    if picksome:
        return pd.DataFrame(Rows, columns=headers), some
    else:
        return pd.DataFrame(Rows, columns=headers)


def pretty(an_item):
    import pprint
    p = pprint.PrettyPrinter(indent=2)
    p.pprint(an_item)


def pairs(D):
    """
    Returns pairs of (key, values).
    :param D: Dictionary
    :return:
    """
    keys = D.keys()
    last = keys[0]
    for i in keys[1:]:
        yield D[last], D[i]
        last = i


def _test_pretty_print():
    import pandas as pd
    from pdb import set_trace
    aa = [[1, 2, 3, 4],
          [1, 2, 3, 4],
          [5, 6, 7, 8],
          [5, 6, 7, 8],
          ]
    aa = pd.DataFrame(aa)
    stringify_pandas(aa)
    set_trace()


if __name__ == "__main__":
    _test_pretty_print()
