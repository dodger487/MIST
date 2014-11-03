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

"""Utilities for magneto toolkit"""

import numpy as np
import scipy
import scipy.interpolate

def TimeLengthMS(runData):
  """Returns how long the runData is in milliseconds"""
  firsttime = runData['magnetometer'][0][0]
  lasttime = runData['magnetometer'][-1][0]
  return (lasttime - firsttime) / 1e6 # convert to ms from ns

def scale(vec, lowerPercentile=2, upperPercentile=98):
  lo = np.percentile(vec, lowerPercentile)
  hi = np.percentile(vec, upperPercentile)
  if hi == lo:
    hi = 1
    lo = 0
  return np.abs(vec / (hi - lo))

def ConvertPhysicsToolboxFile(fname):
  """Convert files generated from this Android app:
    https://play.google.com/store/apps/details?id=com.chrystianvieyra.android.physicstoolboxmagnetometer&hl=en
    into our format
  """

  runData = MakeDummyRunData()
  with open(fname) as f:
    magdata = []
    f.readline()
    f.readline()
    for line in f:
      empty, time, x, y, z = line.strip().split(',')
      magdata.append(map(float, [time, 0, x, y, z]))

  runData['magnetometer'] = magdata
  return runData

def MakeDummyRunData():
  """Makes a runData object with no data populated.

    This could be useful for dealing with data from other sources, making it
    compatible with this toolkit's code.
  """

  dummyData = {
    'magnetometer': {},
    'filename': 'dummy.txt',
    "version": "1.0.0",
    "calibrated": True,
    "labels": [],
    "onAccuracyChangedData": [],
    'systemInfo': {
      "Build.MODEL": "DUMMY_MODEL",
      "Build.VERSION.SDK_INT": 0,
      "Build.DEVICE": "DUMMY_DEVICE",
      "screenResolution.X": 1196,
      "screenResolution.Y": 720,
      "Build.PRODUCT": "dummy_product"
    },
  }
  return dummyData

def FilterPositiveRunData(runDataList):
    return [rd for rd in runDataList if len(rd['labels']) > 0]

def FilterNegativeRunData(runDataList):
    return [rd for rd in runDataList if len(rd['labels']) == 0]

def CreateTemplate(domains, axes, templateLen=450e6):
  """Given two lists of domains and the axis values, creates a template of
    length 450e6 ns (450ms). The template is a function f : time -> microteslas.
    It is the interpolation of provided examples. Any given domain of time less
    than 450ms is not used.
  """
  templateLen = int(templateLen)+int(1e6)

  fcns = []
  for domain, axis in zip(domains, axes):
    domain = np.array(domain)
    domain = domain - domain[0]
    if max(domain) < templateLen: # don't include segments that are too short
      continue
    axis = scale(axis)
    fcns.append(scipy.interpolate.interp1d(domain, axis, kind='cubic'))
  numFcns = len(fcns)
  if numFcns == 0:
    return lambda x : 0
  avgF = lambda x: sum([f(x) for f in fcns]) / float(numFcns)
  newDomain = np.array([d for d in xrange(0, templateLen, int(1e6))]) # 1ms increments up to limit of time snippets
  template = scipy.interpolate.interp1d(newDomain, avgF(newDomain), kind='cubic')
  return template

def CreateTemplates(runDataList, templateLen=450e6):
  """Given a list of snippets, returns an average template of the samples.
    This template is represented as a function f : time -> microteslas. It is
    the average of interpolations of all provided examples.
  """
  domains, Xs, Ys, Zs, mags = [], [], [], [], []
  for runData in runDataList:
    magData = np.array(runData['magnetometer'])

    magData = magData - magData[0,:]

    domain = magData[:,0] # first index is time, second is accuracy
    X = magData[:,2]
    Y = magData[:,3]
    Z = magData[:,4]
    mag = np.sqrt(X**2 + Y**2 + Z**2)

    domains.append(domain)
    Xs.append(X)
    Ys.append(Y)
    Zs.append(Z)
    mags.append(mag)

  return [CreateTemplate(domains, Xs, templateLen=templateLen),
          CreateTemplate(domains, Ys, templateLen=templateLen),
          CreateTemplate(domains, Zs, templateLen=templateLen),
          CreateTemplate(domains, mags, templateLen=templateLen)]
