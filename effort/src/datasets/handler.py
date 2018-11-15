from __future__ import print_function, division
from pdb import set_trace
from effort import *
import pandas
import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)
from glob import glob


def pytocsv():
    for mod in [coc81,Mystery1,Mystery2,cocomo,nasa93]:
            inst  = mod.run()
            fname = mod.__name__.split('.')[-1]+'.csv'
            head = inst.indep+[inst.less[0]]
            print(fname, len(head), " ".join(head))
            body = [elem.cells[:24] for elem in inst._rows]
            dframe = pandas.DataFrame(body, columns = head)
            dframe.to_csv(fname, index=False)


def get_all_datasets():
    all = {}
    files = glob(os.path.abspath(os.path.join(root, 'datasets', "*.csv")))
    for f in files:
        all.update({f.split("/")[-1].split('.')[0]: f})

    return all

if __name__ == '__main__':
    get_all_datasets()
