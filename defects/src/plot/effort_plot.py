from __future__ import print_function, division

import csv
import os


def effort_plot(recall, loc, save_dest="./test", save_name="test"):
    if not os.path.exists(save_dest):
        os.mkdir(save_dest)

    with open("{}/{}.dat".format(save_dest, save_name), "w+") as writer:
        csvwriter = csv.writer(writer, delimiter=" ")
        for i, (lines, rec) in enumerate(zip(loc, recall)):
            csvwriter.writerow((i, int(lines), rec))
