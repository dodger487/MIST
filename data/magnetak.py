#!/usr/bin/env python

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

"""TODO: EXPLAIN, DOCUMENT"""

import magnetak_detectors
import magnetak_evaluate
import magnetak_label
import magnetak_ml
import magnetak_plot
import magnetak_util

import argparse
import dill
import json
import ntpath

"""
Current bugs:
--
make dir if it doesn't exist for snips
--justplot fails
explain how to get snips to work
strange glob import error
make the difference between TP labels and detectors guesses more distinct
Physics Toolbox data isn't plotted
Feature histogram isn't labeled
"""

def preprocessRunData(runDataList):
  """Can put code for cleaning up samples here."""
  newList = [rd for rd in runDataList if magnetak_util.TimeLengthMS(rd) > 400]
  return newList

def addToList(fname, runDataList):
  with open(fname) as f:
    try:
      runData = json.load(f)
    except ValueError as e:
      print "Skipping file %s: %s" % (fname, e)
      return runDataList
  runData["filename"] = ntpath.basename(fname) # strip out directory
  runDataList.append(runData)
  return runDataList

def GetRunDataFromArgs(args):
  runDataList = []
  for arg in args:
    if arg.startswith('--'): # this is an option, skip it
      continue
    elif arg.endswith('.json'): # this is a file to use, add it to datalist
      addToList(arg, runDataList)
    else: # check if it's a directory, and add all JSON files in it
      for fname in glob.glob(os.path.join(arg, '*.json')):
        addToList(fname, runDataList)
  return runDataList

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Read JSON formatted recordings of ' + \
    'magnetometer data, and show results of various algorithms labeling them.')
  parser.add_argument('filelist', metavar='filename', type=str, nargs='+',
    help='list of files to process') # TODO(cjr): make this required
  parser.add_argument('--justplot', dest='justplot', action='store_true',
    default=False, help='Plot the specified files without running labeling algorithms.')
  parser.add_argument('--plot-features', dest='plotFeatures', action='store_true',
    default=False, help="Show positive and negative data on the same plot.")
  parser.add_argument('--plot', dest='display', action='store_true',
    default=False, help='Plot the results of the detector on the given files')
  parser.add_argument('--train-ml', dest='clfOutputName', type=str, nargs=1,
    help="Train an algorithm on the given data and save the classifier with the given filename.")
  parser.add_argument('--run-ml', dest='clfIntputName', type=str, nargs=1,
    help="Load a model from the specified file and run the detector")
  parser.add_argument('--optimize', dest='optimize', action='store_true',
    default=False, help='Train the detector on the input files')
  parser.add_argument('--relabel', dest='relabelDir', type=str, nargs=1,
    help="Relabel the given runData files and put output in given directory")
  parser.add_argument('--snip', dest='snipDir', type=str, nargs=1,
    help="Turn the given runData files into snippets and save them in given directory")
  parser.add_argument('--snip-manual', dest='manualSnipDir', type=str, nargs=1,
    help="Turn the given runData files into snippets and save them in given directory,"+
          " using user input")
  parser.add_argument('--plot-feature-hist', dest='plotFeatureHist',
    action='store_true', default=False,
    help="Plot a histogram of feature distributions")
  parser.add_argument('--plot-thresholds', dest='PlotThresholds',
    action='store_true', default=False,
    help="Plot information useful for the original detector.") # TODO better description
  parser.add_argument('--convert-physics-toolbox', dest='convert',
    action='store_true', default=False,
    help="Read the output from Physics Toolbox and print a runData JSON object.")
  args = parser.parse_args()


  # Get the data we'll use
  if not args.convert:
    runDataList = GetRunDataFromArgs(args.filelist)

    # Preprocess
    runDataList = preprocessRunData(runDataList)

  # Set the detector we'll use
  # detector = magnetak_detectors.OriginalDetector()
  detector = magnetak_detectors.TimeWindowDetector()
  # detector = magnetak_detectors.VectorChangeDetector()
  # detector.setParameters([-3, 10, 1])
  # featurizer = magnetak_ml.WindowFeaturizer()

  if args.justplot: # only plot the data, don't run any detectors
    magnetak_plot.PlotList(runDataList, optPlotData=True)

  elif args.plotFeatures: # plot snippets in a fancy way
    # PlotInterpolatedSnips(runDataList)
    # pl.show()
    magnetak_plot.PlotFeatures(runDataList)
    # PlotSnips(runDataList)

  elif args.clfOutputName: # save machine learning classifier
    # featurizer = magnetak_ml.WindowFeaturizer()
    positives = magnetak_util.FilterPositiveRunData(runDataList)
    templates = magnetak_util.CreateTemplates(positives)
    featurizer = magnetak_ml.KitchenSync(templates)

    mldetector = magnetak_ml.TrainDetectorOnData(runDataList, featurizer)
    with open(args.clfOutputName[0], 'w') as f:
      dill.dump(mldetector, f)

  elif args.clfIntputName: # load and run machine learning classifier
    with open(args.clfIntputName[0]) as f:
      mldetector = dill.load(f)
    if args.display:
      magnetak_evaluate.testDetector(mldetector, runDataList, optPlotData=True)
    else:
      magnetak_evaluate.testDetector(mldetector, runDataList, optPlotData=False)

  elif args.optimize:
    print "Running optimization on", len(runDataList), "runData samples"
    magnetak_evaluate.OptimizeDetector(detector, runDataList, detector.args)

  elif args.display: # run detector, print results, and plot
    magnetak_evaluate.testDetector(detector, runDataList, optPlotData=True)

  elif args.relabelDir: # run detector, print results, and plot
    relabelDir = args.relabelDir[0]
    for rd in runDataList:
      magnetak_label.RelabelAndWriteRunData(rd, outputDir=relabelDir)

  elif args.snipDir: # run detector, print results, and plot
    snipDir = args.snipDir[0]
    for rd in runDataList:
      magnetak_label.WriteSnippets(rd, outputDir=snipDir)

  elif args.manualSnipDir:
    snipDir = args.manualSnipDir[0]
    magnetak_label.createSnippetsFromRunDataList(runDataList, outputDir=snipDir)

  elif args.plotFeatureHist:
    featurizer = magnetak_ml.WindowFeaturizer()
    magnetak_plot.PlotFeatureHistograms(runDataList, featurizer)

  elif args.PlotThresholds:
    for rd in runDataList:
      magnetak_plot.PlotThresholds(rd)
    magnetak_plot.pl.show()

  elif args.convert:
    for fname in args.filelist:
      rd = magnetak_util.ConvertPhysicsToolboxFile(fname)
      print json.dumps(rd)

  else: # run detector and print results
    magnetak_evaluate.testDetector(detector, runDataList)
