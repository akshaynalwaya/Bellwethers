from Models import nasa93
from Technix.de import *
from where2 import *
from Technix.TEAK import *
from Technix.CART import *
from Technix.SVM import *
from Technix.KNN import *
from Technix.sk import rdivDemo
from numpy import mean
import time



def PEEKING_DE(model=MODEL, inp=None):
  mdl = model()
  if inp is None:
    inp = split_data(mdl._rows)[0]
  train, tune, test = inp
  de = DE(model(), launchWhere2, predictPEEKING, peekSettings(), inp)
  best, runs = de.run()
  #Tuned
  classifier = de.builder(de.model, best.decisions, train)
  mre = MRE(de.model, test, classifier, de.predictor)
  tuned = mre.cache.has().median
  #Untuned
  classifier = de.builder(de.model, settings=None, rows=train)
  mre = MRE(de.model, test, classifier, de.predictor)
  untuned = mre.cache.has().median
  return tuned, untuned, runs

def TEAK_DE(model=MODEL, inp=None):
  mdl = model()
  if inp is None:
    inp = split_data(mdl._rows)[0]
  train, tune, test = inp
  de = DE(model(), launchTeak, predictTeak, teakSettings(), inp)
  best, runs = de.run()
  #Tuned
  classifier = de.builder(de.model, best.decisions, train)
  mre = MRE(de.model, test, classifier, de.predictor)
  tuned = mre.cache.has().median
  #Untuned
  classifier = de.builder(de.model, settings=None, rows=train)
  mre = MRE(de.model, test, classifier, de.predictor)
  untuned = mre.cache.has().median
  return tuned, untuned, runs

def CART_DE(model=MODEL, inp=None):
  mdl=model()
  if inp is None:
    inp = split_data(mdl._rows)[0]
  train, tune, test = inp
  de = DE(model(), launchCART, predictCART, cartSettings(), inp)
  best, runs = de.run()
  #Tuned
  classifier = de.builder(de.model, best.decisions, train)
  mre = MRE(de.model, test, classifier, de.predictor)
  tuned = mre.cache.has().median
  #Untuned
  classifier = de.builder(de.model, settings=None, rows=train)
  mre = MRE(de.model, test, classifier, de.predictor)
  untuned = mre.cache.has().median
  return tuned, untuned, runs

def SVM_DE(model=MODEL, inp=None):
  mdl=model()
  if inp is None:
    inp = split_data(mdl._rows)[0]
  train, tune, test = inp
  de = DE(model(), launchSVM, predictSVM, svmSettings(), inp)
  best, runs = de.run()
  #Tuned
  classifier = de.builder(de.model, best.decisions, train)
  mre = MRE(de.model, test, classifier, de.predictor)
  tuned = mre.cache.has().median
  #Untuned
  classifier = de.builder(de.model, settings=None, rows=train)
  mre = MRE(de.model, test, classifier, de.predictor)
  untuned = mre.cache.has().median
  return tuned, untuned, runs

def KNN_DE(model=MODEL, inp=None):
  mdl=model()
  if inp is None:
    inp = split_data(mdl._rows)[0]
  train, tune, test = inp
  de = DE(model(), launchKNN, predictKNN, knnSettings(), inp)
  best, runs = de.run()
  #Tuned
  classifier = de.builder(de.model, best.decisions, train)
  mre = MRE(de.model, test, classifier, de.predictor)
  tuned = mre.cache.has().median
  #Untuned
  classifier = de.builder(de.model, settings=None, rows=train)
  mre = MRE(de.model, test, classifier, de.predictor)
  untuned = mre.cache.has().median
  return tuned, untuned, runs



def run_model(model=MODEL, cross_val=3):
  errors = {
    "Peek" : N(),
    "t_Peek" : N(),
    "TEAK" : N(),
    "t_TEAK" : N(),
    "CART" : N(),
    "t_CART"  : N(),
    "SVM" : N(),
    "t_SVM" : N(),
    "knn" : N(),
    "t_knn" : N()
  }
  runs = {
    "Peek" : N(),
    "TEAK" : N(),
    "SVM" : N(),
    "CART" : N(),
    "knn" : N()
  }
  mdl=model()
  print('###'+model.__name__.upper())
  print('####'+str(len(mdl._rows)) + " data points,  " + str(len(mdl.indep)) + " attributes")
  all_rows = mdl._rows
  print("```")
  for inp in split_data(all_rows, cross_val):
    say(".")
    t_err, err, evals = TEAK_DE(model, inp)
    errors["TEAK"] += err; errors["t_TEAK"] += t_err; runs["TEAK"] += evals
    t_err, err, evals = PEEKING_DE(model, inp)
    errors["Peek"] += err; errors["t_Peek"] += t_err; runs["Peek"] += evals
    t_err, err, evals = CART_DE(model, inp)
    errors["CART"] += err; errors["t_CART"] += t_err; runs["CART"] += evals
    t_err, err, evals = SVM_DE(model, inp)
    errors["SVM"] += err; errors["t_SVM"] += t_err; runs["SVM"] += evals
    t_err, err, evals = KNN_DE(model, inp)
    errors["knn"] += err; errors["t_knn"] += t_err; runs["knn"] += evals
  skData=[]
  for key, n in errors.items():
    skData.append([key]+n.cache.all)
  rdivDemo(skData,"cliffs")
  print("```");print("")
  for key, n in runs.items():
    print("#### Average evals for "+key + " " + str(mean(n.cache.all)))

def run_all(cross_val):
  models = [albrecht.albrecht, kemerer.kemerer, maxwell.maxwell,
           telecom.telecom, cosmic.cosmic, isbsg10.isbsg10]
  for mdl in models:
    run_model(mdl,21)

def testRunner(model=MODEL, cross_val=21):
  errors = {
    "knn" : N(),
    "t_knn" : N(),
  }
  mdl=model()
  print('###'+model.__name__.upper())
  print('####'+str(len(mdl._rows)) + " data points,  " + str(len(mdl.indep)) + " attributes")
  print("```")
  all_rows = mdl._rows
  for inp in split_data(all_rows, cross_val):
    say(".")
    train,tune,test = inp
    t_err, err = SVM_DE(model, inp)
    errors["knn"] += err; errors["t_knn"] += t_err
  skData=[]
  for key, n in errors.items():
    skData.append([key]+n.cache.all)
  rdivDemo(skData,"cliffs")
  print("```");print("")

def untuned_runner(model=MODEL, cross_val=21):
  errors = {
    "Peek" : N(),
    "TEAK" : N(),
    "CART" : N(),
  }
  mdl=model()
  print('###'+model.__name__.upper())
  print('####'+str(len(mdl._rows)) + " data points,  " + str(len(mdl.indep)) + " attributes")
  all_rows = mdl._rows
  print("```")
  for inp in split_data(all_rows, cross_val):
    say(".")
    train,tune,test = inp

    de = DE(model(), launchWhere2, predictPEEKING, peekSettings(), inp)
    classifier = de.builder(de.model, settings=None, rows=train)
    mre = MRE(de.model, test, classifier, de.predictor)
    errors["Peek"] += mre.cache.has().median

    de = DE(model(), launchTeak, predictTeak, teakSettings(), inp)
    classifier = de.builder(de.model, settings=None, rows=train)
    mre = MRE(de.model, test, classifier, de.predictor)
    errors["TEAK"] += mre.cache.has().median

    de = DE(model(), launchCART, predictCART, cartSettings(), inp)
    classifier = de.builder(de.model, settings=None, rows=train)
    mre = MRE(de.model, test, classifier, de.predictor)
    errors["CART"] += mre.cache.has().median
  skData=[]
  for key, n in errors.items():
    skData.append([key]+n.cache.all)
  rdivDemo(skData,"cliffs")
  print("```");print("")

if __name__=="__main__":
  start = time.time()
  run_all(21)
  #testRunner(kemerer.kemerer, 21)
  #run_model(kemerer.kemerer, 3)
  print(time.time() - start)
