import  sys
sys.dont_write_bytecode = True
from lib import *
from sdivUtil import fss
from Models.nasa93 import nasa93

def launchKNN(model, settings=None, rows=None, verbose=False):
  if rows is None:
    rows = model._rows
  if settings is None:
    settings = knnSettings().defaults
  distance = euclid_dist
  if settings.distance == "weighted":
    distance = weighted_dist
  elif settings.distance == "maximal":
    distance = max_dist

  predictor = kMedian
  if settings.adaption == "mean":
    predictor = kMean
  elif settings.adaption == "average_weight":
    predictor = kWeighted

  KNN_object = o(
    rows = rows,
    k = settings.k,
    distance = distance,
    predict = predictor
  )
  if settings.distance == "weighted":
    KNN_object.weights = sdivUtil.fss(model)
  return KNN_object


def predictKNN(model, knnObject, test):
  return knnObject.predict(model, knnObject, test)

def norm(m,c,val) :
  "Normalizes val in col c within model m 0..1"
  return (val- m.lo[c]) / (m.hi[c]- m.lo[c]+ 0.0001)

def euclid_dist(m, knnObject, i, j, what =lambda m: m.decisions):
  "Euclidean distance 0 <= d <= 1 between decisions"
  n      = len(i.cells)
  deltas = 0
  for c in what(m):
    n1 = norm(m, c, i.cells[c])
    n2 = norm(m, c, j.cells[c])
    inc = (n1-n2)**2
    deltas += inc
    n += abs(m.w[c])
  return deltas**0.5 / n**0.5

def weighted_dist(m, knnObject, i, j, what =lambda m: m.decisions):
  "Weighted distance 0 <= d <= 1 between decisions"
  n      = len(i.cells)
  deltas = 0
  for c in what(m):
    n1 = norm(m, c, i.cells[c])
    n2 = norm(m, c, j.cells[c])
    inc = (knnObject.weights[c] * (n1 - n2))**2
    deltas += inc
    n += abs(m.w[c])
  return deltas**0.5 / n**0.5

def max_dist(m, knnObject, i, j, what =lambda m: m.decisions):
  maxDist = 0
  for c in what(m):
    n1 = norm(m, c, i.cells[c])
    n2 = norm(m, c, j.cells[c])
    dist = abs(n1-n2)
    if dist > maxDist:
      dist = maxDist
  return maxDist

def closestN(model,knnObject, test):
  tmp = []
  for row in knnObject.rows:
    if id(test) == id(row): continue
    d = knnObject.distance(model,knnObject,test,row)
    tmp += [ (d,row) ]
  k = min(knnObject.k, len(knnObject.rows))
  return sorted(tmp)[:k]

def kMedian(model, knnObject, test):
  closestK = closestN(model, knnObject, test)
  if len(closestK) % 2 != 0:
    return effort(model, closestK[len(closestK)//2][1])
  else:
    return 0.5*(effort(model, closestK[len(closestK)//2 - 1][1]) + effort(model, closestK[len(closestK)//2][1]))

def kMean(model, knnObject, test):
  closestK = closestN(model, knnObject, test)
  total = sum([effort(model, row) for d, row in closestK])
  return total/len(closestK)

def kWeighted(model, knnObject, test):
  test_effort, sum_wt, eps = 0, 0, 0.000001
  closestK = closestN(model, knnObject, test)
  for dist, row in closestK:
    test_effort += (1/(dist+eps))*effort(model,row)
    sum_wt += (1/(dist+eps))
  return test_effort/sum_wt

# model = nasa93()
# knnObj = launchKNN(model)
# for row in model._rows:
#   print(effort(model,row) , predictKNN(model, knnObj, row))