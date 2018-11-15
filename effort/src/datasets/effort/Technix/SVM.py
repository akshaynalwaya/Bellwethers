import  sys
sys.dont_write_bytecode = True
from lib import *
from sklearn.svm import SVC
from Models.nasa93 import *

def formatForSVM(model, trains):
  indep = lambda x: x.cells[:len(model.indep)]
  dep   = lambda x: x.cells[len(model.indep)]
  train_input = []
  train_output = []
  for train in trains:
    train_input+=[indep(train)]
    train_output+=[dep(train)]
  return train_input, train_output

def launchSVM(model, settings=None, rows=None, verbose=False):
  if rows is None:
    rows = model._rows
  if settings is None:
    settings = svmSettings().defaults
  train_ip, train_op = formatForSVM(model, rows)
  svm = SVC(C=settings.C, kernel=settings.kernel, degree=settings.degree,
             gamma=settings.gamma, coef0=settings.coef0, probability=settings.probability,
             shrinking=settings.shrinking, tol=settings.tol, random_state=1)
  svm.fit(train_ip, train_op)
  return svm

def predictSVM(model, classifier, test):
  test_ip = [test.cells[:len(model.indep)]]
  return classifier.predict(test_ip)[0]


