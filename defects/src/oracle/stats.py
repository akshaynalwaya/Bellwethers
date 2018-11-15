# from collections import Counter
from __future__ import division
from numpy import sqrt
from pdb import set_trace

class counter():

  def __init__(self, before, after, indx):
    self.indx = indx
    self.actual = before
    self.predicted = after
    self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0
    for a, b in zip(self.actual, self.predicted):
      if a == 1 and b == 1:
        self.TP += 1
      if a == 0 and b == 0:
        self.TN += 1
      if a == 0 and b == 1:
        self.FP += 1
      if a == 1 and b == 0:
        self.FN += 1

  def stats(self):
    try:
      Sen  = self.TP / (self.TP + self.FN)
      Spec = self.TN / (self.TN + self.FP)
      Prec = 0
      Acc  = 0
      # F1   = 0

      # G    = 0
      # Prec = self.TP / (self.TP + self.FP)
      # Acc = (self.TP + self.TN) / (self.TP + self.FN + self.TN + self.FP)
      F1 = 2 * self.TP / (2 * self.TP + self.FP + self.FN)

      G = 2 * Sen * Spec / (Sen + Spec)
      G1 = sqrt(Sen**2+(1-Spec)**2)
    except: set_trace()
    return Sen, 1 - Spec, Prec, Acc, F1, G1, G


class ABCD():

  "Statistics Stuff, confusion matrix, all that jazz..."

  def __init__(self, before, after):
    self.actual = before
    self.predicted = after

  def all(self):
    uniques = set(self.actual)
    for u in list(uniques):
      yield counter(self.actual, self.predicted, indx=u)