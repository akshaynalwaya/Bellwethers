import  sys
sys.dont_write_bytecode = True
from lib import *
from sklearn.tree import DecisionTreeClassifier
from Models.nasa93 import *

def formatForCART(model, trains):
  indep = lambda x: x.cells[:len(model.indep)]
  dep   = lambda x: x.cells[len(model.indep)]
  train_input = []
  train_output = []
  for train in trains:
    train_input+=[indep(train)]
    train_output+=[dep(train)]
  return train_input, train_output

def launchCART(model, settings=None, rows=None, verbose=False):
  if rows is None:
    rows = model._rows
  if settings is None:
    settings = cartSettings().defaults
  train_ip, train_op = formatForCART(model, rows)
  tree = DecisionTreeClassifier(criterion="entropy", max_features=settings.max_features ,
                                max_depth=settings.max_depth, min_samples_split=settings.min_samples_split,
                                min_samples_leaf= settings.min_samples_leaf, random_state=1)
  tree.fit(train_ip, train_op)
  return tree

def predictCART(model, tree, test):
  test_ip = [test.cells[:len(model.indep)]]
  return tree.predict(test_ip)[0]
