import os
import sys

import pandas as pd

root = os.getcwd().split('src')[0] + 'src'
if root not in sys.path:
    sys.path.append(root)

from sklearn.feature_selection import chi2
from pdb import set_trace


def fSelect(data, top_n=0.15):
    """
    Feature selection using chi-squared.
    See https://goo.gl/3DdgpI
    :param data: A DataFrame (N-1) feature columns and Nth class column.
    :param top_n: Fraction of features to retain. Default 0.15 (15%).
    :return: Top feature dataset.
    """

    X = data[data.columns[3:-1]]
    y = data[data.columns[-1]]
    chi2_score, p_value = chi2(X, y)
    cutoff = sorted(chi2_score, reverse=True)[int(len(chi2_score) * top_n)]
    columns = [name for i, name in zip(chi2_score, data.columns) if i >= cutoff] + [data.columns[-1]]
    return data[columns]


def _test_fselect():
    data_DF = pd.read_csv(os.path.join(root, "data/Jureczko/ant/ant-1.7.csv"))
    new_df = fSelect(data_DF)
    # ----- Debug -----
    set_trace()


if __name__ == "__main__":
    _test_fselect()
