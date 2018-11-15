from __future__ import print_function, division

import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from TCA.execute import tca_plus
from TNB.execute import tnb
from NAIVE.execute import bellw
import multiprocessing
from pdb import set_trace
from datasets.handler2 import get_all_datasets


def deploy(elem):
    data = elem[0][0]
    fname = elem[1][0]
    method = elem[1][1]
    project = elem[0][1]
    print("Deploying: Datset: {} | Save Path: {}".format(data.lower(), fname.lower()))
    result = method(source=project, target=project, n_rep=12)
    for tgt_name, stats in result.iteritems():
        path = os.path.join(root, "results", data.lower(), fname.lower(), tgt_name + ".csv")
        stats.to_csv(path, index=False)


def run():
    all = get_all_datasets()
    dir_names = {
        "bell": bellw
        , "tca": tca_plus
        , "tnb": tnb
        # , "vcb": vcb
    }
    tasks = []
    for data, project in all.iteritems():
            if data == "LongMethod":
                for f_name, method in dir_names.iteritems():
                    save_path = os.path.join(root, "results", data.lower(), f_name.lower())
                    if os.path.exists(save_path) is False:
                        os.makedirs(save_path)
                    tasks.append(((data, project), (f_name, method)))

    pool = multiprocessing.Pool(processes=4)
    pool.map(deploy, tasks)
    set_trace()


if __name__ == "__main__":
    run()
