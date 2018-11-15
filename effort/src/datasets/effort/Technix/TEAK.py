from __future__ import division,print_function
import  sys  
sys.dont_write_bytecode = True
from lib import *
from where2 import launchWhere2, dist, closestN, leaves
from Models.nasa93 import *

def leafTeak(m,one,node):
  if len(node._kids) > 1:
    east = node.east
    west = node.west
    mid_cos = node.mid_cos
    a = dist(m,one,west)
    b = dist(m,one,east)
    c = dist(m,west,east)
    x = (a*a + c*c - b*b)/(2*c+0.000001)
    if (x<mid_cos):
      return leafTeak(m,one,node._kids[0])
    else:
      return leafTeak(m,one,node._kids[1])
  elif len(node._kids) == 1:
    return node._kids[0]
  return node

def predictTeak(model, tree, test):
  eps = 0.000001
  test_leaf = leafTeak(model, test, tree)
  k = model.settings.neighbors
  rows = test_leaf.val
  if k > len(rows):
    k = len(rows)
  nearestN = closestN(model, k, test, rows)
  if len(nearestN) == 1 :
    return effort(model, nearestN[0][1])
  else :
    testEffort, sumWt = eps, eps
    for distance, row in nearestN[:k]:
      testEffort += effort(model, row)/(distance+eps)
      sumWt += 1/(distance + eps)
    return testEffort/sumWt


def launchTeak(d=nasa93(), settings=None, rows=None, verbose=False):
  if rows is None:
    rows = d._rows
  if settings is None:
    settings = teakSettings().defaults.update(
                minSize = int(len(rows)**0.5),
                prune   = False
              )
  root_node = launchWhere2(d, settings, rows, verbose)
  truncated_rows = []
  all_leaves = []
  for l, level in leaves(root_node):
    all_leaves.append(l)

  if len(all_leaves) > 1:
    for l in all_leaves:
      if l.variance > settings.par_var_ftr*l._up.variance \
              or l.variance > settings.max_var_ftr*d.max_variance:
        truncated_rows += l.val
      
  new_rows = [row for row in root_node.val if row not in truncated_rows]
  if len(new_rows) == 0:
    return root_node
  return launchWhere2(d, settings, new_rows, verbose)
  

if __name__ == "__main__":
  launchTeak()