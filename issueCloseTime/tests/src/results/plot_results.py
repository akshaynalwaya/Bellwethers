from __future__ import print_function, division
import os, sys
from glob import glob
from pdb import set_trace
from pandas import DataFrame, read_csv
import numpy as np
from tabulate import tabulate


def list_communities():
    for var in ["bell", "tca", "tnb"]:
        print("Method: {}".format(var.upper()))
        files = glob(os.path.abspath(os.path.join(".", "godclass", var, "*.csv")))
        yield files


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
            pd_list.update({var.split("/")[-1].split(".")[0]: DataFrame(
                sorted(read_csv(var)[["Name", "G"]].values, key=lambda x: x[0], reverse=True))})

        N = len(pd_list.keys())  # Find number of elements
        stats = np.zeros((N, N))  # Create a 2-D Array to hold the stats
        keys = sorted(pd_list, reverse=True)  # Find data sets (Sort alphabetically, backwards)
        for idx, key in enumerate(keys):  # Populate 2-D array
            for i, val in enumerate(pd_list[key][1].values):
                if not i == idx:  # Ensure self values are set to zero
                    stats[i, idx] = val

        stats = DataFrame(stats, columns=keys, index=keys)
        # stats["Mean"] = stats.median(axis=0)
        # set_trace()
        stats["Mean"] = find_mean(stats)
        stats["Std"] = find_std(stats)
        stats = stats.sort_values(by="Mean", axis=0, ascending=False, inplace=False)
        print(tabulate(stats, showindex=True, headers=stats.columns, tablefmt="fancy_grid"))
        print("\n")
        save_path = os.path.abspath("/".join(var.split("/")[:-2]))
        method = var.split("/")[-2]+".xlsx"
        stats.to_excel(os.path.join(save_path, method))
        compare_tl.append(stats.sort_index(inplace=False)["Mean"].values.tolist())
        compare_tl_head.append(method)
    # set_trace()
    compare_tl= DataFrame(np.array(compare_tl).T, columns=compare_tl_head, index=stats.index.sort_values())
    save_path_2 = os.path.join(os.path.abspath("/".join(var.split("/")[:-3])), os.path.abspath("".join(var.split("/")[-3]))+".xlsx")
    compare_tl.to_excel(save_path_2)
    # set_trace()


if __name__ == "__main__":
    plot_stuff()
    set_trace()
