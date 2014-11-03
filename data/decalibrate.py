#!/usr/bin/env python

# Chris Riederer
# Google, Inc
# 2014-07-10

import numpy as np
import pylab as pl
import json
import sys

TRUE_COLOR = 'green'
INPUT_COLOR = 'red'

def PlotDecalibrated(jDict, showOnAccuracyChanged=False, inputLabels=[]):
  """Given calibrated magnetometer data, transforms it to decalibrated data"""

  magData = np.array(jDict['magnetometer'])
  accuracyData = np.array(jDict['onAccuracyChangedData'])

  magDomain = magData[:,0] # first index is time, second is accuracy
  X = magData[:,2]
  Y = magData[:,3]
  Z = magData[:,4]

  previousAccuracy = magData[0,1] # accuracy of first datapoint
  calibrationTimes = []
  for a in accuracyData:
    t, accuracy = a[0], a[1]
    print t, accuracy
    if accuracy > previousAccuracy:
      calibrationTimes.append(t)

  for cTime in calibrationTimes:
    addMask = (cTime < magDomain) # indices for everything after calibration time

    beforeVect = magData[magData[:,0] < cTime][-1] # last datum before calibration
    afterVect = magData[magData[:,0] > cTime][0] # first datum after calibration
    difference = beforeVect - afterVect

    X[addMask] = X[addMask] + difference[1]
    Y[addMask] = Y[addMask] + difference[2]
    Z[addMask] = Z[addMask] + difference[3]

  mag = np.sqrt(X**2 + Y**2 + Z**2)

  pl.plot(magDomain, X, color='red')
  pl.plot(magDomain, Y, color='blue')
  pl.plot(magDomain, Z, color='green')
  pl.plot(magDomain, mag, color='black')
  pl.title('Decalibrated Data')

  accuracyColors = ['red','blue','green','black']
  if showOnAccuracyChanged:
    for a in accuracyData: # plot accuracy changed events
      t, value = a[0], int(a[1])
      pl.axvline(t, color=accuracyColors[value])

  if 'labels' in jDict.keys() and len(jDict['labels']):
    labelTime = np.array(jDict['labels'])[:,0]
    for t in labelTime:
      pl.axvline(t, color=TRUE_COLOR)

  print inputLabels
  for inputLabel in inputLabels:
    print inputLabel
    pl.axvline(inputLabel, color=INPUT_COLOR)

  pl.show()

def PlotDecalibratedOld():
  """Given calibrated magnetometer data, transforms it to decalibrated data

  Note: This works on the old tabular file format. Use the new method to work
  with the JSON formatted data. Also, this uses onAccuracyChanged, not the
  accuracy reading from each sensor event
  """
  data = np.loadtxt(sys.argv[1], delimiter=',')

  magData = data[data[:, -1] == 0]
  accuracyData = data[data[:, -1] == 2]

  magDomain = magData[:,0]
  X = magData[:,1]
  Y = magData[:,2]
  Z = magData[:,3]

  previousAccuracy = 0
  calibrationTimes = []
  for a in accuracyData:
    t, accuracy, label = a[0], int(a[1]), a[-1]
    if accuracy > previousAccuracy:
      calibrationTimes.append(t)


  for cTime in calibrationTimes:
    addMask = (cTime < magDomain) # indices for everything after calibration time

    beforeVect = magData[magData[:,0] < cTime][-1] # last datum before calibration
    afterVect = magData[magData[:,0] > cTime][0] # first datum after calibration
    difference = beforeVect - afterVect

    X[addMask] = X[addMask] + difference[1]
    Y[addMask] = Y[addMask] + difference[2]
    Z[addMask] = Z[addMask] + difference[3]

  mag = np.sqrt(X**2 + Y**2 + Z**2)

  pl.plot(magDomain, X, color='red')
  pl.plot(magDomain, Y, color='blue')
  pl.plot(magDomain, Z, color='green')
  pl.plot(magDomain, mag, color='black')
  pl.title('Decalibrated Data')

  accuracyColors = ['red','blue','green','black']
  for a in accuracyData: # plot accuracy changed events
    t, value, label = a[0], int(a[1]), a[-1]
    pl.axvline(t, color=accuracyColors[value])

  pl.show()

if __name__ == '__main__':
  with open(sys.argv[1]) as f:
    jDict = json.load(f)

  if "--old" in sys.argv:
    PlotDecalibratedOld()
  else:
    PlotDecalibrated(jDict)
