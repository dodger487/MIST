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
# 2014-07-09

"""
A program for labeling magnet button pulls. Takes the output from magneto as
input, and prints a JSON-formatted dictionary representing the data.
"""

import magnetak_plot

import json
import numpy as np
import os
import pylab as pl
import random
import sys

RELABELED_DIR_NAME = 'relabeled'
NUMBER_NEGATIVE_SNIPS = 10 # number of negative snippets to make from a run

class LabelDrawer(object):
  windowSize = 100
  lineColor = 'red'
  moveAmount = 1

  def __init__(self, ax, domain, startLabelPosition):
    self.ax = ax
    self.domain = np.array(domain)
    # self.labelPosition = startLabelPosition
    # find the position of the closest time to the label in the domain
    self.xCoord = len(domain[domain <= startLabelPosition])
    self.labelPosition = domain[self.xCoord]
    self.lineHandle = None
    self.erased = False # keeps track of if we're removing this label

  def draw(self):
    self.erased = False
    if self.lineHandle in self.ax.lines: # delete previous lines
      lineIndex = self.ax.lines.index(self.lineHandle)
      del self.ax.lines[lineIndex]
    self.lineHandle = self.ax.axvline(self.labelPosition, color=self.lineColor)
    # set the x limits, so the window we're looking at doesn't get strange dimensions
    domain_limits = [self.domain[self.xCoord - self.windowSize/2],
                      self.domain[self.xCoord + self.windowSize/2]]
    self.ax.set_xlim(domain_limits)
    lines = self.ax.get_lines()
    # X, Y, Z, mag = [l.get_data()[1] for l in lines[:4]]
    thisSnip = lambda a : np.array(a)[(self.domain > domain_limits[0]) & (self.domain <= domain_limits[1])]
    X, Y, Z, mag = map(thisSnip, [l.get_data()[1] for l in lines[:4]])
    minY = min([np.min(X), np.min(Y), np.min(Z), np.min(mag),])
    maxY = max([np.max(X), np.max(Y), np.max(Z), np.max(mag),])
    self.ax.set_ylim([minY, maxY])

  def erase(self):
    if self.lineHandle in self.ax.lines: # delete previous lines
      lineIndex = self.ax.lines.index(self.lineHandle)
      del self.ax.lines[lineIndex]
    self.lineHandle = None
    self.erased = True

  def move(self, amount):
    self.xCoord += amount
    self.labelPosition = self.domain[self.xCoord]

  def moveRight(self):
    self.move(self.moveAmount)

  def moveLeft(self):
    self.move(-self.moveAmount)

def LeftRightPlot(fig, ax, domain, startLabel):
  labelDrawer = LabelDrawer(ax, domain, startLabel)
  labelDrawer.draw()
  ax.axvline(startLabel, color='green') # draw original label

  def onkey(event):
    if event.key in {'right', 'left'}:
      labelDrawer.moveRight() if event.key == 'right' else labelDrawer.moveLeft()
      labelDrawer.draw()
      pl.draw()
    elif event.key == 'down':
      labelDrawer.erase()
      pl.draw()
    elif event.key == 'enter':
      pl.close()

  cid = fig.canvas.mpl_connect('key_press_event', onkey)
  pl.show()
  if labelDrawer.erased:
    return 0
  else:
    return int(labelDrawer.labelPosition)

def LabelAndPrintData():
  output = LabelData()
  print json.dumps(output, sort_keys=True, indent=4, separators=(',', ': '))

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

def WriteSnippets(runData, outputDir='.'):
  if len(runData['labels']) > 0:
    snips = makePositiveSnippets(runData)
  else:
    snips = makeNegativeSnippets(runData, NUMBER_NEGATIVE_SNIPS)
  for snip in snips:
    snipFilename = os.path.join(outputDir, snip['filename'])
    with open(snipFilename, 'w') as f:
      json.dump(snip, f)


def PlotLabelData(runData, inputLabels=[]):
  """Plots the data from a run"""

  fig = pl.figure(figsize=(8, 10))
  ax = fig.add_subplot(111)
  pl.title(runData['systemInfo']['Build.MODEL'])

  magData = np.array(runData['magnetometer'])
  accuracyChanges = np.array(runData['onAccuracyChangedData'])

  magDomain = magData[:,0] # first index is time, second is accuracy
  accuracyData = magData[:,1]
  X = magData[:,2]
  Y = magData[:,3]
  Z = magData[:,4]
  mag = np.sqrt(X**2 + Y**2 + Z**2)

  ax.plot(magDomain, X, color='red')
  ax.plot(magDomain, Y, color='blue')
  ax.plot(magDomain, Z, color='green')
  ax.plot(magDomain, mag, color='black')

  for t, l in runData['labels']:
    ax.axvline(t, color='blue')

  for t, l in inputLabels:
    ax.axvline(t, color='red')

  return (fig, ax)

def RelabelAndWriteRunData(runData, outputDir='.'):
  if not os.path.exists(outputDir):
    os.makedirs(outputDir)
  newRunData = RelabelRunData(runData)
  with open(os.path.join(outputDir, newRunData['filename']), 'w') as f:
    json.dump(newRunData, f) # TODO(cjr): better pretty printing
    #, sort_keys=True, indent=2, separators=(',', ': '))

def RelabelRunData(runData):
  """Given a runData object, provides a UI for "cleaning up" the labels

      When labeling data, the labels may be a little messy in time, coming
      before or after the signal. This method creates a plot and lets users move
      the labels so that they're in the same relative location. A runData object
      with the new labels is returned.
  """
  print "File:", runData['filename']
  domain = np.array(runData['magnetometer'])[:,0]
  newlabels = []
  for label in runData['labels']:
    fig, ax = PlotLabelData(runData)
    newlabel = LeftRightPlot(fig, ax, domain, startLabel=label[0])
    if newlabel != 0:
      newlabels.append([newlabel, label[1]])
  PlotLabelData(runData, inputLabels=newlabels) # show both labels, for comparison
  pl.show()
  runData['labels'] = newlabels
  return runData

def RelabelAllRunData(runDataList, outputDir='.'):
  for runData in runDataList:
    # First, clean up the location in time of true positive labels using a UI
    RelabelAndWriteRunData(runData, outputDir=outputDir)

    # Next, create tiny "snippet" files of 600ms
    WriteSnippets(runData, outputDir=outputDir)

def createSnippetsFromRunDataList(runDataList, outputDir=','):
  for runData in runDataList:
    snips = createSnippetsFromPlot(runData)
    for snip in snips:
      newFilename = os.path.join(outputDir, snip['filename'])
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

    magnetak_plot.PlotData(snip) # plot new snip
    magnetak_plot.pl.show()

  magnetak_plot.PlotData(runData, inputLabels=inputLabels)
  fig = magnetak_plot.pl.gcf()
  cid = fig.canvas.mpl_connect('button_press_event', onclick)
  magnetak_plot.pl.show()
  return snippets
