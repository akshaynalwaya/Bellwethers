from __future__ import print_function, division
import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from utils import *
from old.Prediction import rforest
from data.handler import get_all_projects
from matching.match_metrics import match_metrics, list2dataframe
from pdb import set_trace
from oracle.models import rf_model, rf_model0
from old.methods1 import createTbl
from old.stats import ABCD
import pickle
from sklearn.metrics import auc


class HDP:
    def __init__(self, source, target):
        self.source = source
        self.target = target

    @staticmethod
    def known_bellwether(datasets):
        """
        Returns the predetermined bellwether for the community
        """
        for key, value in datasets.iteritems():
            if key.lower() in ['lc', 'mc', 'lucene', 'safe']:
                return key, value

    def one2one_matching(self):
        all_matches = {
            "Source": {
                "name": [],
                "path": []
            },
            "Target": {
                "name": [],
                "path": []
            },
            "matches": []
        }

        for tgt_name, tgt_path in self.target.iteritems():
            for src_name, src_path in self.source.iteritems():
                matched = match_metrics(src_path, tgt_path)
                all_matches["Source"]["name"].append(src_name)
                all_matches["Source"]["path"].append(src_path)
                all_matches["Target"]["name"].append(tgt_name)
                all_matches["Target"]["path"].append(tgt_path)
                all_matches["matches"].append(matched)

        return all_matches

    def bellwether_matching(self, bellwether=False):
        bw_matches = []
        for tgt_name, tgt_path in self.target.iteritems():
            src_name, src_path = self.known_bellwether(self.source)
            matched = match_metrics(src_path, tgt_path)
            if matched:
                bw_matches.extend(matched)
            bw_matches = list(set(bw_matches))

        return bw_matches

    def process(self):
        data = self.one2one_matching()
        source, target, matches = data["Source"], data["Target"], data["matches"]
        result_dict = dict()
        for s_name, s_path, t_name, t_path, match in zip(source["name"], source["path"],
                                                         target["name"], target["path"], matches):
            result_dict.update({t_name: {s_name: {"pd": [], "pf": []}}})
            train = list2dataframe(s_path.data)
            test = list2dataframe(t_path.data)
            train_klass = train.columns[-1]
            test_klass = test.columns[-1]

            # set_trace()
            pd, pf = 0, 0
            if match:
                trainCol = [col[0] for col in match]
                testCol = [col[1] for col in match]
                new_head = [str(i) for i in xrange(len(testCol) + 1)]
                train = train[trainCol + [train_klass]]
                test = test[testCol + [test_klass]]
                actual = test[test.columns[-1]].values.tolist()
                predicted = rf_model(train, test, name=t_name)
                p_buggy = [a for a in ABCD(before=actual, after=predicted)()]
                pd, pf = p_buggy[0].stats()[0], p_buggy[0].stats()[1]
            # result_dict[t_name][s_name]["pd"].append(pd)
            # result_dict[t_name][s_name]["pf"].append(pf)
            yield t_name, pd, pf
            # set_trace()


def save(obj, fname):
    """

    :rtype: object
    """
    pickle.dump(obj, open("./picklejar/result_rerun_{}.pkl".format(fname), "wb"))


def load(name):
    return pickle.load(open("./picklejar/{}.pkl".format(name), "rb"))


def run_hdp():
    """
    This method performs HDP.
    :return:
    """
    all_projects = get_all_projects()  # Get a dictionary of all projects and their respective files.
    result = {}  # Store results here

    for _, v in all_projects.iteritems():
        # Create a template for results.
        for kk in v.keys():
            result.update({kk: {
                'pd': [],
                'pf': []
            }})

    for key_t, value_t in all_projects.iteritems():  # <key/value>_s denotes source
        for key_s, value_s in all_projects.iteritems():  # <key/value>_s denotes target
            if not key_s == key_t:  # Ignore cases where source=target
                print("Source: {}, Target: {}".format(key_s, key_t))
                hdp_runner = HDP(value_s, value_t)
                PD, PF = 0, 0
                oldname = newname = None
                for name, pd, pf in hdp_runner.process():
                    result[name]['pd'].append(pd)
                    result[name]['pf'].append(pf)

    return result


def repeated_runs(n_repeat=5):
    assert n_repeat > 0
    for n in xrange(n_repeat):
        save(run_hdp(), fname=n)

    # for name in result.keys():
    #   print(name + ",", auc(result[name]['pd'], result[name]['pf'], reorder=True))

    # save(result)


def get_stats():
    import pickle
    import numpy as np
    result = pickle.load(open('./picklejar/result.pkl', 'rb'))
    for k, v in result.iteritems():
        auc = np.mean(v)
        stdev = np.var(v) ** 0.5
        print("{},{},{}".format(k, round(auc, 2), round(stdev, 2)))


if __name__ == "__main__":
    # get_stats()
    run_hdp()
