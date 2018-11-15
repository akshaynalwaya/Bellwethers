"""

# Place to store settings.

## Usual Header

"""
import  sys
sys.dont_write_bytecode = True
"""

## Anonymous Containers

"""
class o:
  def __init__(i,**d): i.has().update(**d)
  def has(i): return i.__dict__
  def update(i,**d) : i.has().update(d); return i
  def __repr__(i)   :
    show=[':%s %s' % (k,i.has()[k])
      for k in sorted(i.has().keys() )
      if k[0] is not "#"]
    txt = ' '.join(show)
    if len(txt) > 60:
      show=map(lambda x: '\t'+x+'\n',show)
    return '{'+' '.join(show)+'}'
  def __getitem__(i, item):
    return i.has().get(item)


class E:
  def __init__(i,txt):
    i.txt   = txt
    i._f    = None
  def __call__(i,*lst,**d):
    return i.f()(*lst,**d)
  def f(i):

    if not i._f: i._f=globals()[i.txt]
    return i._f
  def __repr__(i):
    return i.txt+'()'

def defaults(**d):
  return o(_logo="""
            |    |    |
            |    |    |
            |    |    |
            \    |    /
             |   |   |
             |   |   |
             |   |   |
             \   |   /
              |  |  |
              |  |  |
              |  |  |
              |  |  |
            """,
      what=o(
            b4      = '|.. ', # indent string
            verbose = False,  # show trace info?
      ),
      seed    = 1,
      cache   = o(size=128)
  ).update(**d)

def configs(**d):
  return o(
    minSize   = 8,      # min leaf size
    depthMin  = 2,      # no pruning till this depth
    depthMax  = 10,     # max tree depth
    wriggle   = 0.2,    # min difference of 'better'
    prune     = True,   # If pruning should be performed
    neighbors = 2
  ).update(**d)


def peekSettings():
    return o(
        params   = ["minSize", "depthMin", "depthMax", "wriggle", "prune", "neighbors"],
        defaults = o(
          minSize   = 8,      # min leaf size
          depthMin  = 2,      # no pruning till this depth
          depthMax  = 10,     # max tree depth
          wriggle   = 0.2,    # min difference of 'better'
          prune     = True,   # If pruning should be performed
          neighbors = 2
        ),
        max      = [30,        6,          30,          0.5,       True,     6],
        min      = [2,         2,          4,          0.01,       False,    2]
    )

def teakSettings():
  return o(
        params   = ["minSize", "depthMin", "depthMax", "wriggle", "prune",
                    "neighbors", "par_var_ftr", "max_var_ftr"],
        defaults = o(
          minSize   = 8,      # min leaf size
          depthMin  = 2,      # no pruning till this depth
          depthMax  = 10,     # max tree depth
          wriggle   = 0.2,    # min difference of 'better'
          prune     = True,   # If pruning should be performed
          neighbors = 2,      # Number of neighbors to consider while clustering
          par_var_ftr=1.25,   # Factor to multiply parent variance
          max_var_ftr=0.75,   # Factor to multiply max variance
        ),
        max      = [30,        6,          30,          0.5,       True,     6,     2.0,       2.0],
        min      = [2,         2,          4,          0.01,       False,    2,     0.5,       0.5]
    )

def cartSettings():
  return o(
        params   = ["max_features", "max_depth", "min_samples_split", "min_samples_leaf"],
        defaults = o(
          max_features = 1.0,      # max number of features in a leaf
          max_depth  = 50,        # Max depth of tree
          min_samples_split  = 2,   # Minimum Samples required for split
          min_samples_leaf   = 1    # Min Samples Required to be in leaf node
        ),
        max      = [1.0,       50,         20,        20],
        min      = [0.01,      1,          2,         1]
    )

def svmSettings():
  return o(
        params   = ["C", "kernel", "degree", "gamma", "coef0", "probability", "shrinking", "tol"],
        defaults = o(
          C = 1.0,      # Penalty parameter C of the error term
          kernel  = "rbf",        # Specifies the kernel type to be used in the algorithm.
          degree  = 3,   # Degree of the polynomial kernel function (poly)
          gamma   = 0.0,    # Kernel coefficient for rbf, poly and sigmoid
          coef0 = 0.0,     # Independent term in kernel function
          probability = False, # Whether to enable probability estimates
          shrinking = True,   # Whether to use the shrinking heuristic
          tol = 0.001,        # Tolerance for stopping criterion.
        ),
        max      = [1.5,  ["linear", "poly", "rbf", "sigmoid"], 5,  1.0,  1.0, True,  True,  0.01],
        min      = [0.5,  ["linear", "poly", "rbf", "sigmoid"], 2,  0.0,  0.0, False, False, 0.0001]
  )

def knnSettings():
  return o(
        params = ["distance", "adaption", "k"],
        defaults = o(
          distance = "euclid",
          adaption = "median",
          k = 3
        ),
        max = [["euclid", "weighted", "maximal"], ["median", "mean", "average_weight"], 1],
        min = [["euclid", "weighted", "maximal"], ["median", "mean", "average_weight"], 10]
  )

The=None
