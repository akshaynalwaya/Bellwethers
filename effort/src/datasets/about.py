from __future__ import print_function, division
import os
import sys
import pandas as pd
from pdb import set_trace
from glob import glob
from tabulate import tabulate


def details():
    head = ["Dataset", "Samples", " Range (min-max)", "# metrics", "Nature"]
    rows = []
    files = glob(os.path.join(os.path.abspath("./"), "*.csv"))
    nature = "Functional Point"
    for file in files:
        name = file.split("/")[-1].split(".csv")[0]
        name = name.split("-")[0][:10]
        dframe = pd.read_csv(file)
        N = len(dframe)
        n_metrics = len(dframe.columns)
        lo, hi = round(dframe[dframe.columns[-1]].min(), 2), round(dframe[dframe.columns[-1]].max(), 2)
        rows.append([name, N, "{} - {}".format(lo, hi), n_metrics, nature])

    stats = pd.DataFrame(rows, columns=head)
    stats.set_index("Dataset", inplace=False)
    stats.sort_values("Samples", ascending=False, inplace=True)
    stats.to_csv(os.path.abspath(os.path.join(".", "stats_"+file.split("/")[-1])))
    print(tabulate(stats, headers=head, showindex="never"), end="\n\n")
    set_trace()


if __name__ == "__main__":
    details()
