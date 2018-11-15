from __future__ import print_function, division

import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

import pandas as pd
import weka.core.jvm as jvm
from utils import list2dataframe
from weka.classifiers import Classifier
import weka.core.converters as converters
from random import uniform, randint, sample, choice, random as rand
from weka.experiments import SimpleCrossValidationExperiment, Tester, ResultMatrix
from pdb import set_trace

default_opt = [u'-P', u'50', u'-I', u'500', u'-M', u'6.0', u'-V', u'0.01']


class CONF:
    POP = 10
    ITER = 10
    LIVES = 5
    XOVER = 0.75
    WEIGHT = 0.5


def weka_instance(fname):
    if isinstance(fname, list):
        dframe = list2dataframe(fname)
    elif isinstance(fname, pd.core.frame.DataFrame):
        dframe = fname
    elif os.path.isfile(fname):
        dframe = pd.read_csv(fname)

    output = os.path.abspath(os.path.join(root, "csvbin/dframe.csv"))
    dframe.to_csv(output, index=False)
    return output


def tune(data, config=CONF, range_limits=None):
    range_limits = {
        "-P": (1, 100),  # Size of each bag, as a percentage of the training set size. (default 100)
        "-I": (10, 1000),  # Number of Iterations
        "-M": (1, 10),  # Minimum samples per leaf
        "-V": (1e-3, 0.1),  # Variance per split
        "-depth": (0, 6),  # Depth of the tree
        "-N": (0, 10)  # Number of folds for back fitting
    } if range_limits is None else range_limits

    typecast = {
        "-P": int,  # Size of each bag, as a percentage of the training set size. (default 100)
        "-I": int,  # Number of Iterations
        "-M": int,  # Minimum samples per leaf
        "-V": float,  # Variance per split
        "-depth": int,  # Depth of the tree
        "-N": int  # Number of folds for back fitting
    }

    def build_classifier(opt=None):
        opt = default_opt if opt is None else opt
        cls = Classifier(classname="weka.classifiers.trees.RandomForest",
                         options=opt)
        return cls

    def eval_using_crossval(opt=None, r=3, f=3):
        """
        Evaluate performance of a given set of configurations using crossval

        :parameter opt: options as an argument list
        :type opt: list (or str)
        :parameter r: runs
        :type r: int
        :parameter f: folds
        :type f: int
        :return: Area Under The Curve
        """

        # Run 5-way cross validation
        try:
            outfile = os.path.abspath("result.arff")  # store results for later analysis

            if os.path.isfile(outfile):
                "Remove result file if if already exists"
                os.remove(outfile)

            classifier = build_classifier(opt)
            exp = SimpleCrossValidationExperiment(classification=True, runs=r, folds=f, datasets=[data],
                                                  classifiers=[classifier], result=outfile)
            exp.setup()
            exp.run()

            # Compute metric scores
            loader = converters.loader_for_file(outfile)
            result = loader.load_file(outfile)
            matrix = ResultMatrix(classname="weka.experiment.ResultMatrixPlainText")
            tester = Tester(classname="weka.experiment.PairedCorrectedTTester")
            tester.resultmatrix = matrix
            comparison_col = result.attribute_by_name("Area_under_ROC").index
            tester.instances = result

            # remove outfile
            os.remove(outfile)

            # Return Area-Under-Curve Score
            return float(str(tester.multi_resultset_full(0, comparison_col)).split('\n')[2].split()[1])

        except Exception as e:
            pass

    def format_options(opt_dict):
        formatted_opt = []
        for key, val in opt_dict.iteritems():
            formatted_opt.extend([key, str(val)])
        return formatted_opt

    def initialize_population():
        """
        Initialize population

        :return:
        """

        def rand_pop():
            return {
                "-P": randint(*range_limits["-P"]),
                "-I": randint(*range_limits["-I"]),
                "-M": randint(*range_limits["-M"]),
                "-V": round(uniform(*range_limits["-V"]), 2),
                "-depth": randint(*range_limits["-depth"]),
                "-N": randint(*range_limits["-N"])
            }

        return [format_options(rand_pop()) for _ in xrange(config.POP)]

    def evaluate_candidate(candidate):
        return eval_using_crossval(opt=candidate)

    def get_best_solution(population):
        return sorted(population, key=lambda member: evaluate_candidate(member), reverse=True)[0]

    def new_sample(old_member, old_population, problem=range_limits.keys(), weighting_factor=config.WEIGHT,
                   xover_rate=config.XOVER):

        pool = [p for p in old_population if p != old_member]
        p1, p2, p3 = tuple(sample(pool, 3))
        cut_point = choice(problem)

        new = {
            "-P": None,
            "-I": None,
            "-M": None,
            "-V": None,
            "-depth": None,
            "-N": None
        }

        for i, opt_key in enumerate(problem):
            id = 2 * i + 1
            if opt_key is not cut_point and rand() < xover_rate:
                new.update({opt_key: str(typecast[opt_key](typecast[opt_key](p3[id]) +
                                                           weighting_factor * (
                                                               typecast[opt_key](p1[id]) - typecast[opt_key](
                                                                   p2[id]))))})
            else:
                new.update({opt_key: old_member[id]})

        return format_options(new)

    """
    Differential Evolution

    :return: list of options to set
    """
    population = initialize_population()
    s_best = get_best_solution(population)
    lives = CONF.LIVES
    for _ in xrange(CONF.ITER):
        if lives <= 0:
            break

        new_population = []
        lives -= 1
        for candidate in population:
            new_candidate = new_sample(candidate, population)

            if eval_using_crossval(new_candidate) >= eval_using_crossval(candidate):
                new_population.append(new_candidate)
            else:
                new_population.append(candidate)
        population = new_population
        c_best = get_best_solution(population)
        if s_best < c_best:
            s_best = c_best
            lives += 1

    return s_best


def __test_tune():
    data_dir = os.path.abspath("../../data/Jureczko/ant/")
    data = [os.path.join(data_dir, "ant-1.{}.csv".format(i)) for i in xrange(3, 7)]
    jvm.start()
    data = weka_instance(data)
    tuned_opt = tune(data)
    jvm.stop()
    return


if __name__ == "__main__":
    __test_tune()
