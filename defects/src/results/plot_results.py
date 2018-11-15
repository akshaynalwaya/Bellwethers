from __future__ import print_function, division

import os
from glob import glob
from pdb import set_trace

import numpy as np
from pandas import DataFrame, read_csv
from scipy.stats import iqr
from tabulate import tabulate


def list_communities():
    for var in ["bell", "tnb", "vcb"]:
        print("Method: {}".format(var.upper()))
        files = glob(os.path.abspath(os.path.join(".", "aeeem", var, "*.csv")))
        # set_trace()
        yield files


def find_median(dframe):
    return [int(np.median([v for k, v in enumerate(dframe.ix[i].values) if k != i])) for i in
            xrange(len(dframe))]


def find_iqr(dframe):
    return [int(iqr([v for k, v in enumerate(dframe.ix[i].values) if k != i])) for i
            in
            xrange(len(dframe))]


def find_mean(dframe):
    return [int(sum([v for k, v in enumerate(dframe.ix[i].values) if k != i]) / (len(dframe) - 1)) for i in
            xrange(len(dframe))]


def find_std(dframe):
    mean = find_mean(dframe)
    return [int(sum([abs(v - mean[i]) for k, v in enumerate(dframe.ix[i].values) if k != i]) / (len(dframe) - 2)) for i
            in
            xrange(len(dframe))]


def plot_stuff():
    pd_list = {}
    compare_tl = []
    compare_tl_head = []
    for vars in list_communities():
        for var in vars:
            # set_trace()
            try:
                pd_list.update({var.split("/")[-1].split(".")[0]: DataFrame(sorted(read_csv(var)[["Name", "AUC (Mean)"]].values, key=lambda x: x[0], reverse=True))})
            except:
                pd_list.update({var.split("/")[-1].split(".")[0]: DataFrame(sorted(read_csv(var)[["Name", "AUC"]].values, key=lambda x: x[0], reverse=True))})

        keys = [p for p in sorted(pd_list, reverse=True) if not p == "poi"]  # Find data sets (Sort alphabetically, backwards)
        N = len(keys)  # Find number of elements
        stats = np.zeros((N, N))  # Create a 2-D Array to hold the stats
        for idx, key in enumerate(keys):  # Populate 2-D array
            for i, val in enumerate(pd_list[key][1].values):
                # Ensure self values are set to zero
                if i < idx:
                    stats[i, idx] = val
                if i > idx:
                    stats[i + 1, idx] = val

        stats = DataFrame(stats, columns=keys, index=keys)
        # stats["Mean"] = stats.median(axis=0)
        # set_trace()
        stats["Mean"] = find_median(stats)
        stats["Std"] = find_iqr(stats)
        stats = stats.sort_values(by="Mean", axis=0, ascending=False, inplace=False)
        print(tabulate(stats, showindex=True, headers=stats.columns, tablefmt="fancy_grid"))
        print("\n")
        save_path = os.path.abspath("/".join(var.split("/")[:-2]))
        method = var.split("/")[-2] + ".xlsx"
        stats.to_excel(os.path.join(save_path, method))
        compare_tl.append(stats.sort_index(inplace=False)["Mean"].values.tolist())
        compare_tl_head.append(method)

    compare_tl = DataFrame(np.array(compare_tl).T, columns=compare_tl_head, index=stats.index.sort_values())
    save_path_2 = os.path.join(os.path.abspath("/".join(var.split("/")[:-3])),
                               os.path.abspath("".join(var.split("/")[-3])) + ".xlsx")
    compare_tl.to_excel(save_path_2)


if __name__ == "__main__":
    plot_stuff()
    set_trace()
