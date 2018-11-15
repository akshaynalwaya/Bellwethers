from __future__ import division, print_function
import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)
from pdb import set_trace
import pandas as pd
from utils import *
from scipy.stats import ks_2samp


def KSAnalyzer(source, target, cutoff=0.05):
    """

    :param source: Source data set. Formatted as a Pandas DataFrame
    :param target: Target data set. Formatted as a Pandas DataFrame
    :param cutoff: P-Value Cutoff.
    :return: Dictionary. Key is a pair (source, target). Value is p-value.
    For example,

    matches =
    {
        ('$cam', '$cam'): 0.45,
        ('$cbo', '$cbo'): 0.40,
        ('$cbo', '$wmc'): 0.27,
        ('$loc', '$loc'): 0.22,
        ('$npm', '$npm'): 0.16,
        ('$rfc', '$rfc'): 0.98,
        ('$wmc', '$wmc'): 0.60
    }

    """
    s_col = [col for col in source.columns[:-1] if not '?' in col]
    t_col = [col for col in target.columns[:-1] if not '?' in col]
    source = df_norm(source[s_col]) # Refactor!
    target = df_norm(target[t_col]) # Refactor!
    matches = dict()
    # set_trace()

    for col_name_src in source:
        # matches.update({col_name_src: []})
        for col_name_tgt in target:
            try:
                _, p_val = ks_2samp(source[col_name_src],
                                    target[col_name_tgt])
            except: set_trace()
            if p_val > cutoff:
                matches.update({(col_name_src, col_name_tgt): p_val})

    return matches


def get_data():
    return pd.read_csv(os.path.join(root, "data/Jureczko/ant/ant-1.6.csv")), \
           pd.read_csv(os.path.join(root, "data/Jureczko/ant/ant-1.7.csv"))


def _test__KSAnalyzer():
    """
    Test Kolmogorov Smirnov Analyzer
    :return:
    """
    print("Testing KSAnalyzer()\n")
    data0, data1 = get_data()
    try:
        matches = KSAnalyzer(source=data0, target=data1, cutoff=0.05)
        import pprint
        pretty = pprint.PrettyPrinter(indent=2)
        pretty.pprint(matches)
        print("Test Succeeded.")
    except:
        print("Test Failed")
        # ----- Debug -----
        set_trace()


if __name__ == "__main__":
    _test__KSAnalyzer()
