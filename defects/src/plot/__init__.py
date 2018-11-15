# from __future__ import print_function, division
#
# import os
#
#
# def effort_plot(recall, loc, save_dest="./test", save_name="test"):
#     import matplotlib.pyplot as plt
#     import numpy as np
#     loc_r, y_r = np.arange(0, 110, 10), np.arange(0, 110, 10)
#     # plt.scatter(loc, recall, marker='+', c="k", linewidths=0.01)
#     plt.plot(loc, recall, c="k")
#     plt.plot(loc_r, y_r, c="r")
#     plt.xlabel('LOC (%)')
#     plt.ylabel('Recall (%)')
#     plt.xlim([0, 100])
#     plt.ylim([0, 100])
#     plt.title(save_name)
#     plt.grid(True)
#
#     if not os.path.exists(save_dest):
#         os.mkdir(save_dest)
#
#     plt.savefig("{}/{}.png".format(save_dest, save_name))
