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


"""Plotting module for MagnetAK, the Magnetometer Android toolKit"""

import magnetak_util

import pylab as pl
import numpy as np

TRUE_COLOR = 'green'
INPUT_COLOR = 'red'

def PlotData(runData, optPlotData=False, inputLabels=[]):
  """Plots the data from a run"""
  pl.figure()
  pl.title(runData['systemInfo']['Build.MODEL'] + " " + runData['filename'])

  magData = np.array(runData['magnetometer'])

  magDomain = magData[:,0] # first index is time, second is accuracy
  accuracyData = magData[:,1]
  X = magData[:,2]
  Y = magData[:,3]
  Z = magData[:,4]
  mag = np.sqrt(X**2 + Y**2 + Z**2)

  # pl.scatter(magDomain, X, color='red')
  # pl.scatter(magDomain, Y, color='blue')
  # pl.scatter(magDomain, Z, color='green')
  # pl.scatter(magDomain, mag, color='black')

  pl.plot(magDomain, X, color='red')
  pl.plot(magDomain, Y, color='blue')
  pl.plot(magDomain, Z, color='green')
  pl.plot(magDomain, mag, color='black')

  pl.xlabel("Time (ns)") # show axes labels
  pl.ylabel("Magnetometer Data ($\mu$T)")
  pl.legend(["X","Y","Z","Magnitude"], loc="lower left")

  accuracyColors = ['red','blue','green','black']
  if optPlotData:
    for index in xrange(1,len(accuracyData)-1):
      if accuracyData[index] != accuracyData[index-1]:
        pl.scatter(magDomain[index], 0,
                    color=accuracyColors[int(accuracyData[index])])

  if 'labels' in runData.keys() and len(runData['labels']):
    labelTime = np.array(runData['labels'])[:,0]
    for t in labelTime:
      pl.axvline(t, color=TRUE_COLOR)

  for inputLabel in inputLabels:
    pl.axvline(inputLabel, color=INPUT_COLOR)

  def format_coord(x, y): # let us see the full time coordinate in the display
    return 'x=%16f, y=%16f' % (x / 1e6, y)
  ax = pl.gca()
  ax.format_coord = format_coord

def PlotList(runDataList, optPlotData=True):
  """In separate figures, plot the data for each run"""
  for runData in runDataList:
    PlotData(runData, optPlotData=optPlotData)
  pl.show() # shows all the plots from above

def PlotFeatures(runDataList):
  """Plot X,Y,Z and magnitude of snippet in separate plots"""
  f, axarr = pl.subplots(2, 4, sharex=True)

  for runData in runDataList:
    SubPlotFeature(runData, axarr)

  positives = [rd for rd in runDataList if len(rd['labels']) > 0]
  negatives = [rd for rd in runDataList if len(rd['labels']) == 0]

  xp, yp, zp, mp = magnetak_util.CreateTemplates(positives)
  newT = range(0,450000000,1000000)
  axarr[0, 0].plot(newT, [xp(t) for t in newT], color='red')
  axarr[0, 1].plot(newT, [yp(t) for t in newT], color='red')
  axarr[0, 2].plot(newT, [zp(t) for t in newT], color='red')
  axarr[0, 3].plot(newT, [mp(t) for t in newT], color='red')

  xp, yp, zp, mp = magnetak_util.CreateTemplates(negatives)
  newT = range(0,450000000,1000000)
  axarr[1, 0].plot(newT, [xp(t) for t in newT], color='red')
  axarr[1, 1].plot(newT, [yp(t) for t in newT], color='red')
  axarr[1, 2].plot(newT, [zp(t) for t in newT], color='red')
  axarr[1, 3].plot(newT, [mp(t) for t in newT], color='red')

  pl.show()

def SubPlotFeature(runData, axarr):
  magData = np.array(runData['magnetometer'])
  magData = magData - magData[0,:] # normalize based on the first row
  # magData = magData - magData[-1,:] # normalize based on the last value
  magDomain = magData[:,0] # first index is time, second is accuracy
  X = magData[:,2]
  Y = magData[:,3]
  Z = magData[:,4]
  mag = np.sqrt(X**2 + Y**2 + Z**2)
  magDomain = magDomain - magDomain[0] # put in same timescale

  X = magnetak_util.scale(X)
  Y = magnetak_util.scale(Y)
  Z = magnetak_util.scale(Z)
  mag = magnetak_util.scale(mag)

  row = 0 if len(runData['labels']) > 0 else 1
  axarr[row, 0].plot(magDomain, X, alpha=0.2)
  axarr[row, 1].plot(magDomain, Y, alpha=0.2)
  axarr[row, 2].plot(magDomain, Z, alpha=0.2)
  axarr[row, 3].plot(magDomain, mag, alpha=0.2)

  if row == 0:
    axarr[row, 0].set_ylabel('True Positive')
    axarr[row, 0].set_title('X')
    axarr[row, 1].set_title('Y')
    axarr[row, 2].set_title('Z')
    axarr[row, 3].set_title('Magnitude')
  else:
    axarr[row, 0].set_ylabel('True Negative')
    axarr[row, 0].set_ylim(axarr[0, 0].get_ylim())
    axarr[row, 1].set_ylim(axarr[0, 1].get_ylim())
    axarr[row, 2].set_ylim(axarr[0, 2].get_ylim())
    axarr[row, 3].set_ylim(axarr[0, 3].get_ylim())

def PlotSnip(runData):
  """Plot magnitude of snippet in the same plot,
      red if positive, blue otherwise
  """
  magData = np.array(runData['magnetometer'])
  magData = magData - magData[0,:] # normalize data based on first row
  magDomain = magData[:,0] # first index is time, second is accuracy
  X = magData[:,2]
  Y = magData[:,3]
  Z = magData[:,4]
  mag = np.sqrt(X**2 + Y**2 + Z**2)

  magDomain = magDomain - magDomain[0] # put in same timemagnetak_util.scale

  color = 'blue' if len(runData['labels']) > 0 else 'red'
  pl.plot(magDomain, mag, color=color, alpha=0.1)

def PlotSnips(runDataList):
  pl.figure()
  pl.title("Snips")
  for s in runDataList:
    PlotSnip(s)
  pl.show()

def PlotInterpolatedSnips(runDataList):
  fcns = []
  for runData in runDataList:
    if len(runData['labels']) == 0:
      continue

    magData = np.array(runData['magnetometer'])
    magData = magData - magData[0,:] # normalize data based on first row
    # magData = magData - magData[-1,:] # normalize data based on last row
    magDomain = magData[:,0] # first index is time, second is accuracy
    X = magData[:,2]
    Y = magData[:,3]
    Z = magData[:,4]
    mag = np.sqrt(X**2 + Y**2 + Z**2)
    magDomain = magDomain - magDomain[0] # put in same timescale

    mag = magnetak_util.scale(mag)

    fcns.append(scipy.interpolate.interp1d(magDomain, mag, kind='cubic'))
    pl.plot(magDomain, mag, alpha=0.2)

  numFcns = float(len(fcns))
  BigF = lambda x: sum([f(x) for f in fcns]) / numFcns
  newX = range(0,450000000,1000000)
  newY = [BigF(x) for x in newX]
  pl.plot(newX, newY, color='red')

def PlotFeatureHistograms(snipList, featurizer, featureIndex=0, samePlot=True):
  """Plots two histograms of features, one for positive examples and one for
    negative examples. This is used to help engineer good features."""

  positives = [rd for rd in snipList if len(rd['labels']) > 0]
  negatives = [rd for rd in snipList if len(rd['labels']) == 0]

  pos_features = np.array([featurizer.featurize(rd['magnetometer']) for rd in positives])
  neg_features = np.array([featurizer.featurize(rd['magnetometer']) for rd in negatives])

  if samePlot:
    n, bins, patches = pl.hist(pos_features[:,featureIndex], color='red', alpha=0.4)
    pl.hist(neg_features[:,featureIndex], color='blue', bins=bins, alpha=0.4)
    pl.show()
  else:
    pl.figure()
    pl.title("Positive examples feature distribution")
    pl.hist(pos_features[:,featureIndex], color='red')
    pl.figure()
    pl.title("Negative examples feature distribution")
    pl.hist(neg_features[:,featureIndex], color='blue')
    pl.show()

def PlotThresholds(runData, T1=30, T2=130, segment_size=200):
  pl.figure()
  pl.title(runData['systemInfo']['Build.MODEL'] + " " + runData['filename'] + " Thresholds")

  data = np.array(runData['magnetometer'])

  domain = data[:,0] # first index is time, second is accuracy
  # domain = domain * 1e9

  min_seg1 = []
  max_seg2 = []

  segment_time_ns = segment_size * 1e6
  window_size = segment_time_ns * 2
  newDomain = domain[domain > domain[0] + window_size]
  newDomain = map(long, newDomain)

  for sensorTime in newDomain:
    segment1 = data[(domain > sensorTime - window_size) & (domain <= sensorTime - segment_time_ns)]
    segment2 = data[(domain > sensorTime - segment_time_ns) & (domain <= sensorTime)]

    # For each window, calculate the baseline.
    # Get the baseline S0, the last value before we start the segmentation.
    S0 = segment2[-1, 2:5]
    offsets1 = segment1[:, 2:5] - S0
    offsets2 = segment2[:, 2:5] - S0
    norms1 = [np.linalg.norm(row) for row in offsets1]
    norms2 = [np.linalg.norm(row) for row in offsets2]
    min_seg1.append(min(norms1))
    max_seg2.append(max(norms2))

  # Plot the thresholds.
  pl.plot(newDomain, min_seg1, color='red')
  pl.plot(newDomain, max_seg2, color='blue')
  pl.plot(newDomain, np.ones(len(newDomain)) * T1, color='#aadddd') # Minimum must be lower
  pl.plot(newDomain, np.ones(len(newDomain)) * T2, color='#ddaadd') # Maximum must be higher
  pl.show()
