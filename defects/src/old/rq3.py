#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function, division

from Prediction import *
from logo import logo
from methods1 import *
from sk import rdivDemo
from stats import ABCD


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
        elif type ==  "barak":
            dir = "./Data/Turhan09"
            projects = [Name for _, __, Name in walk(dir)][0]
            self.train = [dir+'/'+a for a in projects if dataName in a]
            return

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

def knn(train, test):
    "My version of KNN (uses lists, painfully slow, but it works)"
    lot = train._rows
    new = []

    def edist(A, B):
        return np.sqrt(sum([a**2-b**2 for a, b in zip(A.cells[:-2], B.cells[:-2])]))

    for samp in test._rows:
        new.extend(sorted(lot, key=lambda X: edist(X, samp))[:10])

    new = list(set(new))
    return new

class makeDataSets:
    def __init__(self, file='ant', type='jur', tune=True):
        self.file = file
        self.type = type

    def barak09(self):
        src = data(dataName=self.file, type=self.type)
        self.test = createTbl(src.test, isBin=True)

        def flat(lst):
            """Converts [[a,b],[c,d],...] to [a,b,c,d,...]"""
            new = []
            for l in lst:
                new.extend(l)
            return new

        self.train = createTbl(flat(src.train), isBin=True)
        new = knn(self.train, self.test)

        # Save the KNN data as text file
        Rows = [i.cells[:-1] for i in new]
        headers = [i.name for i in self.train.headers]
        try:
            File = pd.DataFrame(Rows, columns=headers[:-1])
        except:
            set_trace()

        try:
            File.to_csv(self.file + '.csv', index=False)
        except:
            set_trace()


class simulate:
    def __init__(self, file='ant', type='jur', tune=True):
        self.file = file
        self.type = type

    def turhan09(self):

        self.train = createTbl(data(dataName=self.file, type='barak').train, isBin=True)
        self.test = createTbl(data(dataName=self.file, type=self.type).test, isBin=True)
        e = ['Turhan09']
        actual = Bugs(self.test)
        for _ in xrange(10):
            predicted = rforest(
                self.train,
                self.test)
            p_buggy = [a for a in ABCD(before=actual, after=predicted)()]
            e.append(p_buggy[1].stats()[-2])
        return e

    def turhan11(self):

        self.train = createTbl(data(dataName=self.file, type='barak').train, isBin=True)
        self.test = createTbl(data(dataName=self.file, type=self.type).test, isBin=True)
        e = ['Turhan11']
        actual = Bugs(self.test)
        for _ in xrange(10):
            predicted = rforest(
                self.train,
                self.test,
                picksome=True)
            p_buggy = [a for a in ABCD(before=actual, after=predicted)()]
            e.append(p_buggy[1].stats()[-2])
        return e

    def bellwether(self):

        if self.type == 'jur':
            dir = 'Data/Jureczko/'
            pos = 5
        elif self.type == 'mccabe':
            dir = 'Data/mccabe/'
            pos = -2
        elif self.type == 'aeeem':
            dir = 'Data/AEEEM/'
            pos = -3
        elif self.type == 'relink':
            dir = 'Data/Relink/'
            pos = -2

        try:
            train, test = explore(dir)
        except:
            set_trace()
        bellwether = train.pop(pos) + test.pop(pos)
        all = [t + tt for t, tt in zip(train, test)]

        def getTest():
          for dat in all:
            if self.file == dat[0].split('/')[-2]:
              return dat

        try:
            self.test = createTbl(getTest(), isBin=True)
        except:
            set_trace()
        self.train = createTbl(bellwether, isBin=True)
        e = ['Bellwether']

        actual = Bugs(self.test)
        for _ in xrange(10):
            predicted = rforest(
                self.train,
                self.test)
            p_buggy = [a for a in ABCD(before=actual, after=predicted)()]
            e.append(p_buggy[1].stats()[-2])
        return e

def nasa():
    print("NASA\n------\n```")
    for file in ["cm", "jm", "kc", "mc", "mw"]:
        print('### ' + file)
        makeDataSets(file, type='nasa', tune=False).barak09()
        # simulate(file, type='nasa', tune=False).bellwether()
    print('```')


def jur():
    print("Jureczko\n------\n```")
    for file in [ 'log4j', 'ant', 'camel', 'ivy', 'jedit'
                 , 'poi', 'velocity', 'xalan', 'xerces']:
        print('### ' + file)
        E = []
        E.append(simulate(file, type='jur').turhan09())
        E.append(simulate(file, type='jur').turhan11())
        E.append(simulate(file, type='jur').bellwether())
        rdivDemo(E, isLatex=True)
    print('```')


def aeeem():
    print("AEEEM\n------\n```")
    for file in ["EQ", "JDT", "ML", "PDE"]:
        print('### ' + file)
        # makeDataSets(file, type='aeeem', tune=False).barak09()
        E = []
        E.append(simulate(file, type='aeeem').turhan09())
        E.append(simulate(file, type='aeeem').turhan11())
        E.append(simulate(file, type='aeeem').bellwether())
        rdivDemo(E, isLatex=True)
    print('```')


def relink():
    print("Relink\n------\n```")
    for file in ["Apache", "Zxing"]:
        print('### ' + file)
        # makeDataSets(file, type='relink', tune=False).barak09()
        E = []
        E.append(simulate(file, type='relink').turhan09())
        E.append(simulate(file, type='relink').turhan11())
        E.append(simulate(file, type='relink').bellwether())
        rdivDemo(E, isLatex=True)
    print('```')


if __name__ == "__main__":
    logo()  # Print logo
    # nasa()
    # jur()
    # aeeem()
    relink()