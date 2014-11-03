"""
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Chris Riederer
# Google, Inc
# 2014-08-26

import magnetak_plot # TODO(cjr): Should probably decouple this

import numpy as np
import scipy
import scipy.optimize

def OptimizeDetector(detector, runDataList, startArgs):
  def helpOptimize(detector, runDataList, args):
    detector.setParameters(args)
    cost = 0
    for runData in runDataList:
      if len(runData['labels']):
        cost += detector.evaluateCost(runData, True)
      else:
        cost += detector.evaluateCost(runData, False)
    return cost

  fun = lambda args : helpOptimize(detector, runDataList, args)
  optResult = scipy.optimize.fmin_powell(fun, startArgs)
  print optResult

def evaluateDetector(detector, runData, msWindow=600, optPlotData=False):
  """Outputs TP, FP and FN for a detector on a certain set of data.

  Args:
    runData: dictionary containing data about a run, in the format mentioned
      previously
    detector: class inheriting from Detector, which can run 'detect' on a
      runData object, returning times at which it thinks a button-press has occurred
    msWindow: the size of the window, in nanoseconds, that determines if a
      detection counts or not
  Returns:
    A tuple of (number true positives, number false positives, number false negatives)
  """

  labels = runData["labels"]

  # TODO: This only handles binary labels right now. Should handle multi-class.
  output = detector.detect(runData)
  if optPlotData:
    magnetak_plot.PlotData(runData, inputLabels=output)

  nsWindow = msWindow * 1000000 # use nanoseconds for everything

  truePos, falseNeg = [], []
  for labelTime, label in labels:
    # TODO: improve efficiency here
    inRange = [t for t in output if abs(labelTime - t) < nsWindow]
    if len(inRange) > 0:
      truePos.append(inRange[0])
      output.remove(inRange[0])
    else:
      falseNeg.append([labelTime, label])
  falsePos = output
  return (truePos, falsePos, falseNeg)

def testDetector(detector, runDataList, optPlotData=False, printFileData=True,
                  printSummary=True):
  """Runs the detector on each dataset in the runDataList"""
  row_format ="{:<40}{:>15}{:>5}{:>5}{:>5}"
  if printFileData:
    print row_format.format("Filename", "Phone Model", "TP", "FP", "FN")
  phoneTabulatedResults = {}
  for runData in runDataList:
    results = evaluateDetector(detector, runData, optPlotData=optPlotData)
    phoneModel = runData["systemInfo"]["Build.MODEL"]
    if printFileData:
      print row_format.format(runData["filename"],
                            phoneModel,
                            len(results[0]), len(results[1]), len(results[2]))
    if phoneModel not in phoneTabulatedResults:
      phoneTabulatedResults[phoneModel] = np.zeros(3)
    phoneTabulatedResults[phoneModel] += np.array([len(results[0]), len(results[1]), len(results[2])])

  tps = sum([v[0] for v in phoneTabulatedResults.values()])
  fps = sum([v[1] for v in phoneTabulatedResults.values()])
  fns = sum([v[2] for v in phoneTabulatedResults.values()])

  if printFileData:
    print
  if printSummary:
    print "Phone Totals"
    row_format = "{:>15}{:>8}{:>8}{:>8}"
    print row_format.format("Phone Model", "TP", "FP", "FN")
    for model, data in phoneTabulatedResults.iteritems():
      print row_format.format(model, *data)
    print "---------------------------------------"
    print row_format.format("total:", tps, fps, fns)

  magnetak_plot.pl.show() # shows all the plots from above
  return phoneTabulatedResults
