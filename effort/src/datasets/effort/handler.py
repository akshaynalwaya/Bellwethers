import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)
from utils import explore
from old.methods1 import createTbl
from pdb import set_trace


class _Data:
    """Hold training and testing data"""

    def __init__(self, dataName='ant', type='jur'):
        if type == 'jur':
            dir = os.path.join(root, "data/Jureczko")
        elif type == 'nasa':
            dir = os.path.join(root, "data/mccabe")
        elif type == 'aeeem':
            dir = os.path.join(root, "data/AEEEM")
        elif type == "relink":
            dir = os.path.join(root, "data/Relink")
        elif type == 'other':
            dir = os.path.join(root, "data/other/")

        data = explore(dir)
        for d in data:
            if dataName in d[0]:
                self.data = d


class NASA:
    "NASA"
    def __init__(self):
        self.projects = {}
        for file in ["cm", "jm", "kc", "mc", "mw"]:
            self.projects.update({file: _Data(dataName=file, type='nasa')})


class Jureczko:
    "Apache"
    def __init__(self):
        self.projects = {}
        for file in ['ant', 'camel', 'ivy', 'jedit', 'log4j',
                     'lucene', 'poi', 'velocity', 'xalan', 'xerces']:
             self.projects.update({file: _Data(dataName=file, type='jur')})


class AEEEM:
    "AEEEM"
    def __init__(self):
        self.projects = {}
        for file in ["EQ", "JDT", "LC", "ML", "PDE"]:
            self.projects.update({file: _Data(dataName=file, type='aeeem')})


class ReLink:
    "RELINK"
    def __init__(self):
        self.projects = {}
        for file in ["Apache", "Safe", "Zxing"]:
            self.projects.update({file: _Data(dataName=file, type='relink')})

def get_all_projects():
    all = dict()
    for community in [AEEEM, Jureczko, AEEEM, ReLink, NASA]:
        all.update({community.__doc__: community().projects})
    return all

def _test():
    data = NASA()
    data.projects


if __name__ == "__main__":
    _test()
