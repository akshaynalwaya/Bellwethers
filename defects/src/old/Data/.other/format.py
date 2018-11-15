from __future__ import print_function

import csv
import re
from os import walk
from os.path import join
from pdb import set_trace
from pandas import DataFrame, read_csv


def do():
    print('Doing')
    for (a, b, c) in walk('./'):
        if not a=='./':
            for file in c:
                old = read_csv(join(a,file))
                newcol = old.columns.values.tolist()
                _ = newcol.pop(0)
                depen = newcol.pop(8)
                newcol.append(depen)
                new=old[newcol]
                new.columns=['$'+n for n in newcol[:-1]]+['$>'+newcol[-1]]
                new.to_csv(join(a,file))


if __name__ == '__main__':
    do()
