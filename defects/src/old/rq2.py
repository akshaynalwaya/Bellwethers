#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function, division

from Prediction import *
from logo import logo
from methods1 import *
from sk import rdivDemo
from stats import ABCD


def getTunings(fname):
    raw = pd.read_csv('tunings.csv').transpose().values.tolist()
    formatd = pd.DataFrame(raw[1:], columns=raw[0])
    return formatd[fname].values.tolist()


class data:
    """Hold training and testing data"""

    def __init__(self, dataName='ant', type='jur'):
        if type == 'jur':
            dir = "./Data/Jureczko"
        elif type == 'nasa':
            dir = "./Data/mccabe"
        elif type == 'aeeem':
            dir = "./Data/AEEEM"
        elif type == "relink":
            dir = './Data/Relink'

        try:
            projects = [Name for _, Name, __ in walk(dir)][0]
        except:
            set_trace()
        numData = len(projects)  # Number of data
        one, two = explore(dir)
        data = [one[i] + two[i] for i in xrange(len(one))]

        def whereis():
            for indx, name in enumerate(projects):
                if name == dataName:
                    return indx

        loc = whereis()
        self.train = []
        self.test = []
        for idx, dat in enumerate(data):
            if idx != loc:
                self.train.append(dat)
        self.test = data[loc]


class simulate:
    def __init__(self, file='ant', type='jur', tune=True):
        self.file = file
        self.type = type
        self.param = None if not tune else getTunings(file)
        # set_trace()

    def incrUpdates(self):
        """When to update a reference set"""
        everything = []
        src = data(dataName=self.file, type=self.type)
        set_trace()
        self.test = createTbl(src.test, isBin=True)


    def bellwether(self):
        everything = []
        src = data(dataName=self.file, type=self.type)
        self.test = createTbl(src.test, isBin=True)

        table_rows = [["Dataset", "ED", "G2", "Pd", "Pf"]]

        if len(src.train) == 1:
            train = src.train[0]
        else:
            train = src.train
        header = [" "]
        # onlyMe = [self.file]
        val = []
        for file in train:
            fname = file[0].split('/')[-2]
            e = [fname]
            header.append(fname)
            self.train = createTbl(file, isBin=True)
            actual = Bugs(self.test)
            for _ in xrange(10):
                predicted = rforest(
                    self.train,
                    self.test,
                    tunings=self.param,
                    smoteit=True)
                p_buggy = [a for a in ABCD(before=actual, after=predicted).all()]
                e.append(p_buggy[1].stats()[-2])

                # val.append([fname, "%0.2f" % p_buggy[1].stats()[-2], "%0.2f" % p_buggy[1].stats()[-3]
                #                , "%0.2f" % p_buggy[1].stats()[0], "%0.2f" % p_buggy[1].stats()[1]])
            everything.append(e)

        rdivDemo(everything, isLatex=True)

        # table_rows.extend(sorted(val, key=lambda F: float(F[2]), reverse=True))
        #
        # "Pretty Print Thresholds"
        # table = Texttable()
        # table.set_cols_align(["l", 'l', "l", "l", "l"])
        # table.set_cols_valign(["m", "m", "m", "m", "m"])
        # table.set_cols_dtype(['t', 't', 't', 't', 't'])
        # table.add_rows(table_rows)
        # print(table.draw(), "\n")

        # ---------- DEBUG ----------
        #   set_trace()


def attributes(type='jur'):
    """ Returns total instances, and no. of bug in a file"""
    from Prediction import formatdata
    if type == 'jur':
        for name in ['ant', 'camel', 'ivy', 'jedit', 'log4j',
                     'lucene', 'poi', 'velocity', 'xalan', 'xerces']:
            src = data(dataName=name, type=type)
            test = src.test
            dat = createTbl(test, isBin=True)
            dframe = formatdata(dat)
            bugs = dframe[dframe.columns[-2]].values
            print(name, len(bugs), str(sum(bugs)) + " (%0.2f)" % (sum(bugs) / len(bugs) * 100))
    elif type == 'nasa':
        for name in ["cm", "jm", "kc", "mc", "mw"]:
            src = data(dataName=name, type=type)
            test = src.test
            dat = createTbl(test, isBin=True)
            dframe = formatdata(dat)
            bugs = dframe[dframe.columns[-2]].values
            print(name, len(bugs), str(sum(bugs)) + " (%0.2f)" % (sum(bugs) / len(bugs) * 100))
    elif type == 'aeeem':
        for name in ["EQ", "JDT", "LC", "ML", "PDE"]:
            src = data(dataName=name, type=type)
            test = src.test
            dat = createTbl(test, isBin=True)
            dframe = formatdata(dat)
            bugs = dframe[dframe.columns[-2]].values
            print(name, len(bugs), str(sum(bugs)) + " (%0.2f)" % (sum(bugs) / len(bugs) * 100))
    elif type == 'relink':
        for name in ["Apache", "Safe", "Zxing"]:
            src = data(dataName=name, type=type)
            test = src.test
            dat = createTbl(test, isBin=True)
            dframe = formatdata(dat)
            bugs = dframe[dframe.columns[-2]].values
            print(name, len(bugs), str(sum(bugs)) + " (%0.2f)" % (sum(bugs) / len(bugs) * 100))


def nasa():
    print("NASA\n------\n```")
    for file in ["cm", "jm", "kc", "mc", "mw"]:
        print('### ' + file)
        simulate(file, type='nasa', tune=False).bellwether()
    print('```')


def jur():
    print("Jureczko\n------\n```")
    for file in ['ant', 'camel', 'ivy', 'jedit', 'log4j',
                 'lucene', 'poi', 'velocity', 'xalan', 'xerces']:
        print('### ' + file)
        simulate(file, type='jur').incrUpdates()
    print('```')


def aeeem():
    print("AEEEM\n------\n```")
    for file in ["EQ", "JDT", "LC", "ML", "PDE"]:
        print('### ' + file)
        simulate(file, type='aeeem', tune=False).bellwether()
    print('```')


def relink():
    print("Relink\n------\n```")
    for file in ["Apache", "Safe", "Zxing"]:
        print('### ' + file)
        simulate(file, type='relink', tune=False).bellwether()
    print("```")


if __name__ == "__main__":
    logo()  # Print logo
    # nasa()
    jur()
    # aeeem()
    # relink()
    # attributes('jur')
    # attributes('nasa')
    # attributes('aeeem')
    # print("")
    # attributes('relink')
