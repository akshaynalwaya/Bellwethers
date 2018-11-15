"""

# Place to store settings.

## Usual Header

"""
from __future__ import division,print_function
import  sys

sys.dont_write_bytecode = True
from lib import *
from settings import *
from Models import nasa93
MODEL = nasa93

def DE_settings(**d):
    # return o(
    #    max     = 100,  # number of repeats
    #    np      = 50,  # number of candidates
    #    f       = 0.5, # extrapolate amount
    #    cf      = 0.75,  # prob of cross-over
    #    lives   = 5
    # ).update(**d)
    return o(
       max     = 1000,  # number of repeats
       np      = 10,  # number of candidates
       f       = 0.75, # extrapolate amount
       cf      = 0.3,  # prob of cross-over
       lives   = 5
    ).update(**d)


def split_data(rows, cross_val):
  random.seed(1)
  splits = []
  for _ in range(cross_val):
    random.shuffle(rows)
    size = len(rows)
            #Train          #Tune                   #Test
    splits.append((rows[:size//3], rows[size//3:-size//3], rows[-size//3:]))
  return splits


def trim(one, low, high):
  return max(low, min(one, high))

def between(low, high, prec=6):
  if isinstance(low, float):
    return round(random.uniform(low, high), prec)
  elif isinstance(low, bool):
    return bool(random.getrandbits(1))
  elif isinstance(low, int):
    return int(round(random.uniform(low, high)))
  elif isinstance(low, list):
    return random.choice(low)
  return None


def MRE(model, inp, classifier, predictor):
  mre = N()
  for row in inp:
    desired = effort(model, row)
    pred = predictor(model, classifier, row)
    mre += abs(desired - pred)/desired
  return mre


class Candidate(o):
  id = 0
  def __init__(i, de, cand= None, default=False):
    if cand:
      i.id = cand.id
      i.decisions = cand.decisions
      i.objectives = None
    else :
      i.id = Candidate.id =  Candidate.id+1
      i.generate(de.settings, default)
      i.evaluate(de)

  def generate(i, settings, default=False):
    if default :
      i.decisions = settings.defaults
    else :
      dec = o()
      for index, key in enumerate(settings.params):
        dec.has()[key] = between(settings.min[index], settings.max[index])
      i.decisions = dec

  def evaluate(i, de):
    classifier = de.builder(de.model, i.decisions, de.train)
    results = MRE(de.model, de.tune, classifier, de.predictor).cache.has()
    i.objectives = [results.median, results.iqr]

  def __lt__(i, j):
    for index in range(0, len(i.objectives)):
      if i.objectives[index] > j.objectives[index]:
        return False
    return True

class DE(o):
  "DE"
  id = 0
  def __init__(i, model, builder, predictor, settings, inp):
    random.seed(1)
    i.id = DE.id = DE.id+1
    i.model = model
    i.builder = builder
    i.predictor = predictor
    i.settings = settings
    i.train, i.tune, i.test = inp
    i.config = DE_settings()
    i.frontier = i.build()

  """
  Build Frontier
  """
  def build(i):
    canditates = [ Candidate(i) for _ in range(i.config.np - 1)]
    canditates.append(Candidate(i, default=True))
    return  canditates

  def best(i):
    return sorted(i.frontier)[0]

  def run(i):
    lives = i.config.lives
    evals = 0
    for _ in range(i.config.max):
      if lives == 0: break
      updated = i.update()
      evals += len(i.frontier)
      if not updated: lives -= 1
    return i.best(),evals

  def update(i):
    nextGen = []
    oldBest = i.best()
    for point in i.frontier:
      mutated = i.mutate(point)
      if mutated > point:
        nextGen.append(mutated)
      else:
        nextGen.append(point)
    i.frontier= nextGen
    newBest = i.best()
    if newBest > oldBest:
      return True
    return False


  def mutate(i, point, prec=3):
    mutated = Candidate(i, cand = point)
    two, three, four = i.threeMore(point)
    obj2, obj3, obj4 = two.decisions, three.decisions, four.decisions
    cf,f = i.config.cf, i.config.f
    for _, (key, low, high) in enumerate(zip(i.settings.params,i.settings.min, i.settings.max)):
      if random.random() > cf :
        if isinstance(low, list):
          extrapolated = random.choice(low)
        elif isinstance(low, float):
          extrapolated = trim(round(obj2[key] + f*(obj3[key] - obj4[key]), prec), low, high)
        elif isinstance(low, bool):
          extrapolated  = bool(random.getrandbits(1))
        elif isinstance(low, int):
          extrapolated =  trim(int(round(random.uniform(low, high))), low, high)
        mutated.decisions.has()[key] = extrapolated
    mutated.evaluate(i)
    return mutated

  def threeMore(i, one):
    seen = [one.id]
    def oneMore(seen):
      while True:
        point = random.choice(i.frontier)
        if point.id not in seen:
          seen += [point.id]
          return point
    return oneMore(seen), oneMore(seen), oneMore(seen)