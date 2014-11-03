#!/usr/bin/env python

# Chris Riederer
# Google, Inc
# 2014-07-25

import test_detect

import numpy as np
import os
import json
import random
import sys

def makeNegativeSnippets(runData, number, snipPrefixTime=100000000, snipPostfixTime=500000000):
  return makeSnippets(runData, True, numberNegative=number, snipPrefixTime=snipPrefixTime, snipPostfixTime=snipPostfixTime)

def makePositiveSnippets(runData, snipPrefixTime=100000000, snipPostfixTime=500000000):
  return makeSnippets(runData, False, snipPrefixTime=snipPrefixTime, snipPostfixTime=snipPostfixTime)

def makeSnippets(runData, isNegative, numberNegative=None, snipPrefixTime=10000000, snipPostfixTime=100000000):
  """Given a runData file, makes smaller snippets of positive examples for training

  runData: the JSON object representation of a recording
  snipPrefixTime: the time, in NANOSECONDS, preceding the label time that we're
    putting in the snippet
  snipPrefixTime: the time, in NANOSECONDS, after the label time that we're
    putting in the snippet
  """

  data = np.array(runData["magnetometer"])
  data = data[data[:, 2:].any(1)]
  domain = data[:,0]

  if isNegative and len(runData['labels']) != 0:
    raise Exception("Length of labels should be 0 when generating negative examples")
  elif not isNegative and len(runData['labels']) == 0:
    raise Exception("Length of labels cannot be 0 when generating positive examples")
  elif isNegative:
    # generate start point for snippets, and ensure snippet is entirely in recorded data
    possibleStartPoints = domain[domain < domain[-1] - snipPostfixTime - snipPostfixTime]
    labels = [[labelTime, 1] for labelTime in random.sample(possibleStartPoints, numberNegative)]
  else:
    labels = runData['labels']

  snippets = []
  for index, (labelTime, label) in enumerate(labels):
    snippet = runData.copy()
    if isNegative:
      snippet['labels'] = []
    else:
      snippet['labels'] = [[labelTime, label]]
    snippet['filename'] = "%s-%02d.json" % (runData['filename'].rsplit('.')[0], index)
    snippetIndices = (domain >= labelTime-snipPrefixTime) & (domain < labelTime+snipPostfixTime)
    snippet['magnetometer'] = list(map(list, data[snippetIndices, :])) # convert back to python list, so JSON can serialize
    snippets.append(snippet)

  return snippets

def makeSnippet(runData, snipId, startTime, snipLength=600000000):
  data = np.array(runData["magnetometer"])
  data = data[data[:, 2:].any(1)]
  domain = data[:,0]
  snippet = runData.copy()
  labels = [[labelTime, label] for labelTime, label in runData['labels'] if startTime < labelTime < startTime+snipLength]
  snippet['labels'] = labels
  # todo: filename
  snippet['filename'] = "%s-hn-%02d.json" % (runData['filename'].rsplit('.')[0], snipId)
  snippetIndices = (domain >= startTime) & (domain < startTime+snipLength)
  snippet['magnetometer'] = list(map(list, data[snippetIndices, :])) # convert back to python list, so JSON can serialize
  return snippet

def findHardNegatives(runData, snipLength=600000000):
  """Find portions of a signal that are difficult for our detector to realize are negative"""
  # TODO: initially writing this just for negative runData files... should make it work with everything

  detector = test_detect.OriginalDetector()
  snippet = runData.copy()

  data = np.array(runData["magnetometer"])
  data = data[data[:, 2:].any(1)]
  domain = data[:,0]

  min_cost = float('inf')
  for startTime in domain[(domain < domain[-1] - snipLength)]:
    snippetIndices = (domain >= startTime) & (domain < startTime+snipLength)
    snippet['magnetometer'] = list(map(list, data[snippetIndices, :])) # convert back to python list, so JSON can serialize
    snippet['labels'] = []
    cost = detector.evaluateCost(snippet, True)
    if cost < min_cost:
      min_cost = cost
      worst_snip = snippet.copy()

  return worst_snip

def createSnippetsFromRunDataList(runDataList):
  runDataList = test_detect.GetRunDataFromArgs(sys.argv[1:])
  for runData in runDataList:
    snips = createSnippetsFromPlot(runData)
    for snip in snips:
      newFilename = os.path.join('relabeled', snip['filename'])
      with open(newFilename, 'w') as f:
        print newFilename
        json.dump(snip, f)

def createSnippetsFromPlot(runData, inputLabels=[], snipLength=600000000):
  """This creates a plot from runData. When the user clicks on the plot, a snippet
  of length snipLength nanoseconds is created and plotted. The user can repeat
  this process as many times as he or she likes. When the user closes the
  original plot, the list of the created snippets is returned.
  """
  snippets = []

  def onclick(event):
    startTime = event.xdata
    print "Start time of snippet: %16d" % int(startTime)
    snipId = len(snippets)
    snip = makeSnippet(runData, snipId, startTime, snipLength=snipLength)
    snippets.append(snip) # add to snippets

    test_detect.PlotData(snip) # plot new snip
    test_detect.pl.show()

  test_detect.PlotData(runData, inputLabels=inputLabels)
  fig = test_detect.pl.gcf()
  cid = fig.canvas.mpl_connect('button_press_event', onclick)
  test_detect.pl.show()
  return snippets

if __name__ == '__main__':
  runDataList = test_detect.GetRunDataFromArgs(sys.argv[1:])
  createSnippetsFromRunDataList(runDataList)
  # print sum([len(runData['labels']) for runData in runDataList])
