from __future__ import print_function, division

import os
from pdb import set_trace
from time import time

import pandas as pd
import weka.core.converters as converters
import weka.core.jvm as jvm
from weka.classifiers import Classifier
from utils import list2dataframe
from tuner.devol import tune, default_opt, weka_instance

classifiers = {
    "rf": "weka.classifiers.trees.RandomForest",
    "c45": "weka.classifiers.trees.J48",
    "j48": "weka.classifiers.trees.J48",
    "cart": "weka.classifiers.trees.SimpleCart",
    "logistic": "weka.classifiers.functions.Logistic",
}


def csv_as_ndarray(fname):
    if isinstance(fname, str):
        dframe = pd.read_csv(fname)
    elif isinstance(fname, pd.core.frame.DataFrame):
        dframe = fname

    # dframe.loc[dframe[dframe.columns[-1]] > 0, dframe.columns[-1]] = 1.0
    # set_trace()
    ndarray = dframe.values.tolist()
    return ndarray


def get_actuals(fname):
    if isinstance(fname, str):
        dframe = pd.read_csv(fname)
    if isinstance(fname, list):
        dframe = list2dataframe(fname)
    elif isinstance(fname, pd.core.frame.DataFrame):
        dframe = fname

    return dframe[dframe.columns[-1]].values.tolist()


def classify(train, test, name="RF", tuning=False):
    jvm.start()

    if isinstance(train, list) and isinstance(test, list):
        train = weka_instance(train)
        trn_data = converters.load_any_file(train)
        test = weka_instance(test)
        tst_data = converters.load_any_file(test)

    elif os.path.isfile(train) and os.path.isfile(test):
        trn_data = converters.load_any_file(train)
        tst_data = converters.load_any_file(test)

    else:
        trn = csv_as_ndarray(train)
        tst = csv_as_ndarray(test)

        trn_data = converters.ndarray_to_instances(trn, relation="Train")
        tst_data = converters.ndarray_to_instances(tst, relation="Test")

    trn_data.class_is_last()
    tst_data.class_is_last()

    # t = time()
    if tuning:
        opt = tune(train)
    else:
        opt = default_opt
    # print("Time to tune: {} seconds".format(time() - t))

    cls = Classifier(classname=classifiers[name.lower()], options=opt)

    cls.build_classifier(trn_data)

    distr = [cls.distribution_for_instance(inst)[1] for inst in tst_data]
    preds = [cls.classify_instance(inst) for inst in tst_data]

    jvm.stop()

    return preds, distr


def __test_learners():
    data_dir = "../data/Jureczko/ant/"
    train = os.path.join(data_dir, "ant-1.3.csv")
    test = os.path.join(data_dir, "ant-1.4.csv")
    act, prd = classify(train, test)
    set_trace()
    return


if __name__ == "__main__":
    data_dir = "../data/Jureczko/ant/"
    __test_learners()
    # classify(data_dir)
    # load_and_print(data_dir)
