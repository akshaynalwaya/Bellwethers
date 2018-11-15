from collections import Counter
from pdb import set_trace
from random import choice, uniform as rand

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.neighbors import BallTree


def SMOTE(data=None, atleast=101, atmost=101, k=5, type='Both'):
    """
    Synthetic Minority Oversampling Technique

    :param data: A DataFrame of imbalanced data
    :param atleast: Minimum number of minority class samples
    :param atmost: Maximum number of majority class samples
    :param k: Number of nearest neighbors
    :param type: "Both": Under/oversample
               , "under/down": Only downsample majority class
               , "over/up": Only upsample minority class
               , "none": Do nothing.
    :return: A new balanced dataset.
    """

    def knn(a, b):
        "k nearest neighbors"
        b = np.array([bb[:-1] for bb in b])
        tree = BallTree(b)
        __, indx = tree.query(a[:-1], k=6)

        return [b[i] for i in indx]

    def extrapolate(one, two):
        new = len(one) * [None]
        new[:-1] = [a + rand(0, 1) * (b - a) for
                    a, b in zip(one[:-1], two[:-1])]
        new[-1] = int(one[-1])
        return new

    def populate(data, atleast):
        newData = [dd.tolist() for dd in data]

        if atleast - len(newData) > 0:
            for _ in xrange(atleast - len(newData)):
                one = choice(data)
                neigh = knn(one, data)[1:k + 1]
                try:
                    two = choice(neigh)
                except IndexError:
                    two = one
                newData.append(extrapolate(one, two))
            return newData

    def cull(data):
        return [choice(data).tolist() for _ in xrange(atmost)]

    newCells = []
    klass = lambda df: df[df.columns[-1]]
    count = Counter(klass(data))

    major, minor = count.keys()
    # set_trace()
    for u in count.keys():
        if u == minor:
            try:
                newCells.extend(populate([r for r in data.as_matrix() if r[-1] == u], atleast=atleast))
            except:
                pass
        if u == major:
            newCells.extend(cull([r for r in data.as_matrix() if r[-1] == u]))
        else:
            newCells.extend([r.tolist() for r in data.as_matrix() if r[-1] == u])

    return pd.DataFrame(newCells, columns=data.columns)


def __test_smote():
    """
    A test case goes here
    :return:
    """
    pass


if __name__ == "__main__":
    __test_smote()
