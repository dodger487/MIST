#!/usr/bin/env python

# Chris Riederer
# Google, Inc
# 2014-07-09

"""
A testing framework for evaluating magnet-button-press algorithms
"""

import argparse
import bisect
import copy
import dill # pickle.dump fails if dill isn't imported!?
import glob
import json
import math
import ntpath
import numpy as np
import os
import pickle
import pylab as pl
import scipy.optimize
import scipy.interpolate
import scipy.spatial.distance
import sklearn
import sklearn.cross_validation
import sklearn.svm
import sklearn.linear_model
import sys
import time

TRUE_COLOR = 'green'
INPUT_COLOR = 'red'

RESULTS_FNAME = "RESULTS.csv"

"""
Data format
  magnetometer data
  labels (when an event occurred)
  device info (optional)
  calibrated or not (optional)
  rotation data (optional)
  onAccuracyChanged data (optional)

{
  "version":"1.0.0",
  "magnetometer": [
    [1023929383847183, 0, -432.12, 34.21, 220.4], # time, accuracy, X, Y, Z
    [1023929399482938, 0, -429.34, 33.45, 221.23],
    ...
    [1023929430291923, 0, -431.32, -345.55, 32.23]
  ]
  "labels": [[1029389838425443, 1], [10293938472453452, 1], ... ],
  "deviceInfo": {"name": "s4", ...},
  "onAccuracyChanged": [[102938983844356432, 0], [1029393847234234, 1], ...],
  "calibrated": False
}
"""

class Detector(object):
  """Abstract base class of magnet-button-press detectors"""

  def detect(self, runData):
    """This method should take in a runData dictionary and output times"""
    raise NotImplementedError("Please implement this method")

  def setParameters(self, args):
    """This method should take in a list of parameters and set them appropriately"""
    raise NotImplementedError("Please implement this method")

  def evaluateCost(self, runData, isPositive):
    """This method should return a cost for how far away the detector was from being correct"""
    raise NotImplementedError("Please implement this method")

class SimpleMagnitudeDetector(Detector):
  """A simple detector which detects a button press if magnet vector magnitude is
  above a certain threshold"""

  def __init__(self):
    self.threshold = 582

  def setParameters(self, args):
    self.threshold = args[0]

  def detect(self, runData):
    output = []
    magData = np.array(runData["magnetometer"])
    magTimes = magData[:,0] # first index is time, second is accuracy
    X = magData[:,2]
    Y = magData[:,3]
    Z = magData[:,4]
    magnitudes = np.sqrt(X**2 + Y**2 + Z**2)

    lastTime = 0
    for i in xrange(len(magTimes)):
      if i - lastTime < 30:
        continue
      if magnitudes[i] > self.threshold:
        output.append(magTimes[i])
        lastTime = i
    return output

class MLDetector(Detector):
  """A simple detector which detects a button press if magnet vector magnitude is
  above a certain threshold"""

  def __init__(self):
    self.clf = None
    # fcn that takes magnetomer data and converts it to a feature vector
    self.MagnetToVectorObj = None
    self.lookBehindTime = 400 #ms
    self.waitTime = 350 # ms

  def detect(self, runData):
    lookBehindTime = self.lookBehindTime * 1e6 # convert to nanoseconds

    waitTime = self.waitTime *1e6
    detections = []

    data = np.array(runData["magnetometer"])
    data = data[data[:, 2:].any(1)]
    domain = data[:,0] # times
    lastFiring = 0 # keep track of last time button was pulled
    for sensorTime in domain[domain > domain[0]+lookBehindTime]:
      # wait for a full window before looking again
      if sensorTime - lastFiring < waitTime:
        continue

      window = data[(domain > sensorTime - lookBehindTime) & (domain <= sensorTime)]

      # wait to fire if we don't have any sensor events
      if len(window) == 0:
        continue

      X = window[:,2]
      Y = window[:,3]
      Z = window[:,4]
      magnitudes = np.sqrt(X**2 + Y**2 + Z**2)

      # some basic thresholds, put in sequence for easy commenting-out!
      if abs(magnitudes[0] - magnitudes[-1]) > 500:
        continue
      # if min(magnitudes) > 1400:
      #   continue
      if max(magnitudes) - min(magnitudes) < 30:
        continue

      featureVector = self.MagnetToVectorObj.featurize(window)
      if self.clf.predict(featureVector)[0]:
        detections.append(sensorTime)
        lastFiring = sensorTime

    return detections

class MagnitudeTemplateDetector(Detector):
  """Fires if the magnitude is close to a given template"""

  def __init__(self):
    self.threshold = 1
    self.template = None # needs to be set externally

  def setParameters(self, args):
    self.threshold = args[0]

  def difference(self, domain, X, Y, Z):
    magnitudes = np.sqrt(X**2 + Y**2 + Z**2)
    magnitudes = scale(magnitudes)
    domain = domain - domain[0]

    return sum([abs(mag - self.template(t)) for mag, t in zip(magnitudes, domain) if t < 4000000])

  def detect(self, runData):
    output = []
    magData = np.array(runData["magnetometer"])
    domain = magData[:,0] # first index is time, second is accuracy
    X = magData[:,2]
    Y = magData[:,3]
    Z = magData[:,4]
    # magnitudes = np.sqrt(X**2 + Y**2 + Z**2)

    diff = self.difference(domain, X, Y, Z)
    print runData['filename'], diff

    if diff < self.threshold:
      return [domain[0]]
    else:
      return []

class TimeWindowDetector(Detector):
  """The original detector for paperscope modified to use time instead of samples"""

  def __init__(self):
    self.T1 = 30
    self.T2 = 130
    self.segment_time = 200 # ms

    # self.T1 = 9.8
    # self.T2 = 405
    # self.segment_time = 204 # ms

    self.waitTime = 350 # ms

    # self.T1 = 8.56265499e+01
    # self.T2 = 3.29885044e+01
    # self.segment_time = 2.00000056e+08
    # self.segment_time = 200000000

    self.args = [self.T1, self.T2, self.segment_time]

  def setParameters(self, args):
    self.T1 = args[0]
    self.T2 = args[1]
    self.segment_time = args[2]

  def detect(self, runData):
    segment_time_ns = self.segment_time * 1e6 # convert to nanoseconds
    window_size = segment_time_ns * 2
    waitTime = self.waitTime * 1e6
    history = []
    detections = []

    data = np.array(runData["magnetometer"])
    data = data[data[:, 2:].any(1)]
    domain = data[:,0] # times
    lastFiring = 0 # keep track of last time button was pulled
    for sensorTime in domain[domain > domain[0]+window_size]:
      segment1 = data[(domain > sensorTime - window_size) & (domain <= sensorTime - segment_time_ns)]
      segment2 = data[(domain > sensorTime - segment_time_ns) & (domain <= sensorTime)]
      window = data[(domain > sensorTime - window_size) & (domain <= sensorTime)]

      # wait to fire if we don't have any sensor events
      if len(segment1) == 0 or len(segment2) == 0:
        continue

      if sensorTime - lastFiring < waitTime: # wait for a full window before looking again
        continue

      # For each window, calculate the baseline.
      # Get the baseline S0, the last value before we start the segmentation.
      S0 = segment2[-1, 2:5]

      # A place for the means and maximums.
      means = []
      maximums = []
      minimums = []

      for segment in [segment1, segment2]:
        # Calculate the offset for each of the samples in the segment.
        samples = segment[:, 2:5]
        offsets = samples - S0
        norms = [np.linalg.norm(row) for row in offsets]

        # Calculate the metrics for each segment.
        # print segment
        # print offsets
        # print norms
        means.append(np.mean(norms))
        maximums.append(np.max(norms))
        minimums.append(np.min(norms))

      # Apply the thresholds to the computed statistics.
      min_1 = minimums[0]
      max_2 = maximums[1]

      if min_1 < self.T1 and max_2 > self.T2:
        detections.append(sensorTime)
        lastFiring = sensorTime
    return detections

  def evaluateCost(self, runData, isPositive):
    segment_time_ns = self.segment_time * 1e6 # convert to nanoseconds
    window_size = segment_time_ns * 2
    history = []
    detections = []

    data = np.array(runData["magnetometer"])
    data = data[data[:, 2:].any(1)]
    domain = data[:,0] # times
    lastFiring = 0 # keep track of last time button was pulled
    for sensorTime in domain[domain > domain[0]+window_size]:
      segment1 = data[(domain > sensorTime - window_size) & (domain <= sensorTime - segment_time_ns)]
      segment2 = data[(domain > sensorTime - segment_time_ns) & (domain <= sensorTime)]
      window = data[(domain > sensorTime - window_size) & (domain <= sensorTime)]

      # TODO: deal with extra firings
      if sensorTime - lastFiring < segment_time_ns: # wait for a full window before looking again
        continue

      # For each window, calculate the baseline.
      # Get the baseline S0, the last value before we start the segmentation.
      S0 = segment2[-1, 2:5]

      # A place for the means and maximums.
      means = []
      maximums = []
      minimums = []

      for segment in [segment1, segment2]:
        # Calculate the offset for each of the samples in the segment.
        samples = segment[:, 2:5]
        offsets = samples - S0
        norms = [np.linalg.norm(row) for row in offsets]

        # Calculate the metrics for each segment.
        means.append(np.mean(norms))
        maximums.append(np.max(norms))
        minimums.append(np.min(norms))

      # Apply the thresholds to the computed statistics.
      min_1 = minimums[0]
      max_2 = maximums[1]

      # Store cost for each window. Cost=0 if it fires and is positive.
      if isPositive:
        cost = max(0, min_1 - self.T1) + max(0, self.T2 - max_2)
      else:
        cost = max(0, self.T1 - min_1) + max(0, max_2 - self.T2)
      history.append(cost)
    return min(history) if isPositive else max(history)


class OriginalDetector(Detector):
  """The original detector for paperscope"""

  def __init__(self):
    self.T1 = 30
    self.T2 = 130

    # self.T1 = 90 # trying to make difficult false positives
    # self.T2 = 90

    # self.T1 = 31.5 # found by optimization
    # self.T2 = 123.5

  def setParameters(self, args):
    self.T1 = args[0]
    self.T2 = args[1]

  def detect(self, runData):
    segment_size = 20
    # segment_size = 10
    window_size = segment_size * 2
    history = []
    detections = []

    data = np.array(runData["magnetometer"])
    data = data[data[:, 2:].any(1)]
    lastFiring = 0 # keep track of last time button was pulled
    for window in np.arange(len(data) - window_size):
      window_end = window + window_size
      if window_end - lastFiring < window_size: # wait for a full window before looking again
        continue

      # For each window, calculate the baseline.
      # Get the baseline S0, the last value before we start the segmentation.
      S0 = data[window_end, 2:5]

      # Also, split the whole window into segments.
      # TODO: Calculate the segment size in samples based on time.
      segments = np.arange(window, window_end, segment_size)

      # A place for the means and maximums.
      means = []
      maximums = []
      minimums = []

      for segment in segments:
        # Calculate the offset for each of the samples in the segment.
        samples = data[segment:segment + segment_size, 2:5]
        offsets = samples - S0
        norms = [np.linalg.norm(row) for row in offsets]

        # Calculate the metrics for each segment.
        means.append(np.mean(norms))
        maximums.append(np.max(norms))
        minimums.append(np.min(norms))

      # Apply the thresholds to the computed statistics.
      min_1 = minimums[0]
      max_2 = maximums[1]

      # Store I_1, M_2 and I_3 for each window.
      #history.append([window_end, U1, M2, U3, np.linalg.norm(S0)])
      history.append([window_end, min_1, max_2, np.linalg.norm(S0), np.linalg.norm(S0)])
      # print [window_end, min_1, max_2, np.linalg.norm(S0), np.linalg.norm(S0)]

      #print 'Window %d: I_1=%f, M_2=%f, I_3=%f' % (window, U1, M2, U3)

      #if U1 < MAX_U1 and U3 < MAX_U3 and M2 > MIN_M2:
      # print min_1, max_2
      if min_1 < self.T1 and max_2 > self.T2:
        detections.append(data[window_end, 0])
        lastFiring = window_end
    return detections

  def evaluateCost(self, runData, isPositive):
    segment_size = 10
    window_size = segment_size * 2
    history = []
    detections = []

    data = np.array(runData["magnetometer"])
    data = data[data[:, 2:].any(1)]
    lastFiring = 0 # keep track of last time button was pulled
    for window in np.arange(len(data) - window_size):
      window_end = window + window_size
      if window_end - lastFiring < window_size: # wait for a full window before looking again
        continue

      # For each window, calculate the baseline.
      # Get the baseline S0, the last value before we start the segmentation.
      S0 = data[window_end, 2:5]

      # Also, split the whole window into segments.
      # TODO: Calculate the segment size in samples based on time.
      segments = np.arange(window, window_end, segment_size)

      # A place for the means and maximums.
      means = []
      maximums = []
      minimums = []

      for segment in segments:
        # Calculate the offset for each of the samples in the segment.
        samples = data[segment:segment + segment_size, 2:5]
        offsets = samples - S0
        norms = [np.linalg.norm(row) for row in offsets]

        # Calculate the metrics for each segment.
        means.append(np.mean(norms))
        maximums.append(np.max(norms))
        minimums.append(np.min(norms))

      # Apply the thresholds to the computed statistics.
      min_1 = minimums[0]
      max_2 = maximums[1]

      # Store cost for each window. Cost=0 if it fires and is positive.
      if isPositive:
        cost = max(0, min_1 - self.T1) + max(0, self.T2 - max_2)
      else:
        cost = max(0, self.T1 - min_1) + max(0, max_2 - self.T2)
      history.append(cost)
    return min(history) if isPositive else max(history)


class ScaledTimeWindowDetector(Detector):
  """The original detector for paperscope modified to use time instead of samples"""

  def __init__(self):
    self.T1 = 0.7
    self.T2 = 0.7
    self.T3 = 0.1

    self.segment_time = 200 # ms
    self.waitTime = 350 # ms
    self.args = [self.T1, self.T2, self.segment_time]

  def setParameters(self, args):
    self.T1 = args[0]
    self.T2 = args[1]
    self.T3 = args[2]
    self.segment_time = args[2]

  def detect(self, runData):
    segment_time_ns = self.segment_time * 1e6 # convert to nanoseconds
    window_size = segment_time_ns * 2
    waitTime = self.waitTime *1e6
    history = []
    detections = []

    data = np.array(runData["magnetometer"])
    data = data[data[:, 2:].any(1)]
    domain = data[:,0] # times
    lastFiring = 0 # keep track of last time button was pulled
    for sensorTime in domain[domain > domain[0]+window_size]:
      if sensorTime - lastFiring < waitTime: # wait for a full window before looking again
        continue

      window = data[(domain > sensorTime - window_size) & (domain <= sensorTime)]
      window = window - window[0,:]
      X = window[:,2]
      Y = window[:,3]
      Z = window[:,4]
      raw_mag = np.sqrt(X**2 + Y**2 + Z**2)

      if abs(raw_mag[0] - raw_mag[-1]) > 500:
        continue
      # if min(magnitudes) > 1400:
      #   continue
      if max(raw_mag) - min(raw_mag) < 30:
        continue
      scaled_mag = scale(raw_mag)

      segment1 = np.array(raw_mag[(window[:,0] < segment_time_ns)])
      segment2 = np.array(raw_mag[(window[:,0] > segment_time_ns) & (window[:,0] <= window_size)])
      scaled_segment1 = np.array(scaled_mag[(window[:,0] < segment_time_ns)])
      scaled_segment2 = np.array(scaled_mag[(window[:,0] > segment_time_ns) & (window[:,0] <= window_size)])

      # wait to fire if we don't have any sensor events
      if len(segment1) == 0 or len(segment2) == 0:
        continue

      # if min(segment1) < 50 and min(scaled_segment1<0.3) and


# np.array([
#             min(norms1),
#             max(norms1),
#             np.mean(norms1),
#             min(norms2),
#             max(norms2),
#             np.mean(norms2),

#             min(scaled_segment1),
#             max(scaled_segment1),
#             np.mean(scaled_segment1),
#             min(scaled_segment2),
#             max(scaled_segment2),
#             np.mean(scaled_segment2),
#             ])

#       # A place for the means and maximums.
#       means = []
#       maximums = []
#       minimums = []
#       raw_minimums = []

#       for segment in [segment1, segment2]:
#         # Calculate the metrics for each segment.
#         raw_minimums.append(np.min(segment))
#         means.append(np.mean(segment))
#         maximums.append(np.max(segment))
#         minimums.append(np.min(segment))

#       # Apply the thresholds to the computed statistics.
#       raw_minimums
#       min_1 = minimums[0]
#       max_2 = maximums[1]
#       min_2 = minimums[1]
#       # print [min_1, max_2, min_2, ]

      if (max(scaled_segment1) < self.T1
          and max(scaled_segment2) > self.T2
          and min(scaled_segment2) < self.T3
          and min(segment2) < 50
          # and max(segment2) > 80
          ):
        detections.append(sensorTime)
        # print [min_1, max_2, min_2, ]
        lastFiring = sensorTime

    return detections


class ScaledThreeWindowDetector(Detector):
  """The original detector for paperscope modified to use time instead of samples"""

  def __init__(self):
    self.T1 = 0.7
    self.T2 = 0.7
    self.T3 = 0.1

    self.segment_time_1 = 150 # ms
    self.segment_time_2 = 200 # ms
    self.segment_time_3 = 50 # ms
    self.waitTime = 350 # ms
    # self.args = [self.T1, self.T2, self.segment_time]

  def setParameters(self, args):
    raise NotImplementedError()

  def detect(self, runData):
    segment_time_1 = self.segment_time_1 * 1e6 # convert to nanoseconds
    segment_time_2 = self.segment_time_2 * 1e6 # convert to nanoseconds
    segment_time_3 = self.segment_time_3 * 1e6 # convert to nanoseconds
    window_size = (self.segment_time_1 + self.segment_time_2 + self.segment_time_3) * 1e6
    waitTime = self.waitTime *1e6
    history = []
    detections = []

    data = np.array(runData["magnetometer"])
    data = data[data[:, 2:].any(1)]
    domain = data[:,0] # times
    lastFiring = 0 # keep track of last time button was pulled
    for sensorTime in domain[domain > domain[0]+window_size]:
      if sensorTime - lastFiring < waitTime: # wait for a full window before looking again
        continue

      window = data[(domain > sensorTime - window_size) & (domain <= sensorTime)]
      window = window - window[0,:]
      X = window[:,2]
      Y = window[:,3]
      Z = window[:,4]
      raw_mag = np.sqrt(X**2 + Y**2 + Z**2)

      if abs(raw_mag[0] - raw_mag[-1]) > 500:
        continue
      # if min(magnitudes) > 1400:
      #   continue
      if max(raw_mag) - min(raw_mag) < 30:
        continue
      scaled_mag = scale(raw_mag)

      segment1 = np.array(raw_mag[(window[:,0] <= segment_time_1)])
      segment2 = np.array(raw_mag[(window[:,0] > segment_time_1) & (window[:,0] <= segment_time_1 +segment_time_2)])
      segment3 = np.array(raw_mag[(window[:,0] > segment_time_1 +segment_time_2) & (window[:,0] <= window_size)])
      scaled_segment1 = np.array(scaled_mag[(window[:,0] <= segment_time_1)])
      scaled_segment2 = np.array(scaled_mag[(window[:,0] > segment_time_1) & (window[:,0] <= segment_time_1 +segment_time_2)])
      scaled_segment3 = np.array(scaled_mag[(window[:,0] > segment_time_1 +segment_time_2) & (window[:,0] <= window_size)])
      # scaled_segment1 = np.array(scaled_mag[(window[:,0] < segment_time_ns)])
      # scaled_segment2 = np.array(scaled_mag[(window[:,0] > segment_time_ns) & (window[:,0] <= window_size)])

      # wait to fire if we don't have any sensor events
      if len(segment1) == 0 or len(segment2) == 0 or len(segment3) == 0:
      # if len(segment1) == 0 or len(segment2) == 0:
        continue

      # if min(segment1) < 50 and min(scaled_segment1<0.3) and


# np.array([
#             min(norms1),
#             max(norms1),
#             np.mean(norms1),
#             min(norms2),
#             max(norms2),
#             np.mean(norms2),

#             min(scaled_segment1),
#             max(scaled_segment1),
#             np.mean(scaled_segment1),
#             min(scaled_segment2),
#             max(scaled_segment2),
#             np.mean(scaled_segment2),
#             ])

#       # A place for the means and maximums.
#       means = []
#       maximums = []
#       minimums = []
#       raw_minimums = []

#       for segment in [segment1, segment2]:
#         # Calculate the metrics for each segment.
#         raw_minimums.append(np.min(segment))
#         means.append(np.mean(segment))
#         maximums.append(np.max(segment))
#         minimums.append(np.min(segment))

#       # Apply the thresholds to the computed statistics.
#       raw_minimums
#       min_1 = minimums[0]
#       max_2 = maximums[1]
#       min_2 = minimums[1]
#       # print [min_1, max_2, min_2, ]

      if (
          # np.mean(scaled_segment2) > 0.8
          max(scaled_segment2) > 0.6
          and max(scaled_segment3) < 0.3
          # max(scaled_segment1) < self.T1
          # and max(scaled_segment2) > self.T2
          # and min(scaled_segment2) < self.T3
          # max(segment1) < 30
          # and max(segment2) > 100
          and max(segment3) < 60
          # and min(segment2) < 50
          # and max(segment2) > 80
          ):
        detections.append(sensorTime)
        # print [min_1, max_2, min_2, ]
        lastFiring = sensorTime

    return detections

class MagGradientThresholdDetector(Detector):
  """A simple threshold based on the gradient of the magnitude"""
  def __init__(self):
    self.Thi = 3
    self.Tlo = -3
    self.window_size = 100 # time in milliseconds we wait
    self.args = [self.Thi, self.Tlo, self.window_size]

  def setParameters(self, args):
    self.Thi = args[0]
    self.Tlo = args[1]
    self.window_size = args[2]
    self.args = args

  def detect(self, runData):
    window_size = self.window_size * 1000000 # convert to nanoseconds
    data = np.array(runData["magnetometer"])
    data = data[data[:, 2:].any(1)]
    domain = data[:,0]
    X = data[:,2]
    Y = data[:,3]
    Z = data[:,4]
    magnitudes = np.sqrt(X**2 + Y**2 + Z**2)
    gradM = np.gradient(magnitudes)
    detections = []

    lastTime = 0
    for index, magStart in enumerate(domain[domain < domain[-1]-window_size]):
      # if index - lastTime < 30:
        # continue
      nextIndex = len([d for d in domain if d < magStart + window_size])
      magEnd = gradM[nextIndex]
      if magStart > self.Thi and magEnd < self.Tlo:
        t = domain[nextIndex]
        detections.append(t)
        lastTime = index

    return detections

  def evaluateCost(self, runData, isPositive):
    pass

class ThresholdDetector(Detector):
  """A simple threshold based on the gradient of the magnitude"""
  def __init__(self):
    self.magT = 3

  def setParameters(self, args):
    self.T = args[0]

  def detect(self, runData):
    data = np.array(runData["magnetometer"])
    data = data[data[:, 2:].any(1)]
    domain = data[:,0]
    X = data[:,2]
    Y = data[:,3]
    Z = data[:,4]
    magnitudes = np.sqrt(X**2 + Y**2 + Z**2)
    gradM = np.gradient(magnitudes)
    detections = []

    lastTime = 0
    for index, t in enumerate(domain):
      if index - lastTime < 30:
        continue
      mag = gradM[index]
      if mag > self.T:
        detections.append(t)
        lastTime = index

    return detections

class VectorChangeDetector(Detector):
  """TODO"""
  def __init__(self):
    self.window_size = 100 # window size in milliseconds
    self.Xlo = -3
    self.Yhi = 18
    self.Zhi = 6

    # self.window_size, self.Xlo, self.Yhi, self.Zhi = [-0.38549015, -2.03822928,  4.02000427,  2.27985856]
    # self.Xlo, self.Yhi, self.Zhi = [-0.23956289, -0.23956307 , 0.1800537 ]
    # self.args = [self.window_size, self.Xlo, self.Yhi, self.Zhi]
    self.args = [self.Xlo, self.Yhi, self.Zhi]

  def setParameters(self, args):
    self.Xlo, self.Yhi, self.Zhi = args

    # self.window_size = abs(args[0])
    # self.Xlo = args[1]
    # self.Yhi = args[2]
    # self.Zhi = args[3]
    self.args = args

  def detectAndEvaluate(self, runData, isPositive=1):
    history = []
    window_size = self.window_size * 1e6 # convert to nanoseconds
    data = np.array(runData["magnetometer"])
    data = data[data[:, 2:].any(1)]
    domain = data[:,0]
    X = data[:,2]
    Y = data[:,3]
    Z = data[:,4]
    detections = []

    lastTime = 0
    for index, timeStart in enumerate(domain[domain < domain[-1]-window_size]):
      nextIndex = len(domain[domain < timeStart + window_size])
      currentTime = domain[nextIndex]
      if currentTime - lastTime < 350000000:
        continue
      oldValues = data[index, 2:5]
      currentValues = data[nextIndex, 2:5]
      difference = currentValues - oldValues
      # print difference
      if difference[0] < self.Xlo and difference[1] > self.Yhi and difference[2] > self.Zhi:
        t = domain[nextIndex]
        detections.append(t)
        lastTime = currentTime

      # Store cost for each window. Cost=0 if it fires and is positive.
      if isPositive:
        cost = max(0, difference[0] - self.Xlo) + max(0, self.Yhi - difference[1]) + max(0, self.Zhi - difference[2])
        # cost = cost**2
      else:
        cost = max(0, -difference[0] + self.Xlo) + max(0, -self.Yhi + difference[1]) + max(0, -self.Zhi + difference[2])
        # cost = cost**2
      history.append(cost)
    out_cost = min(history) if isPositive else max(history)
    return (detections, out_cost)

  def detect(self, runData):
    detections, out_cost = self.detectAndEvaluate(runData)
    return detections

  def evaluateCost(self, runData, isPositive):
    detections, out_cost = self.detectAndEvaluate(runData, isPositive)
    return out_cost

class MagnetometerToFeatureVector(object):
  def featurize(self, magnetometer):
    """This method should take in magnetometer data and output a feature vector"""
    raise NotImplementedError("Please implement this method")

class MagnitudeTemplateSumOfDifferencesMagToVec(MagnetometerToFeatureVector):
  def __init__(self, templates):
    self.templates = templates
    self.window_size = 400000000

  def SumOfDifferences(self, domain, axis, template):
    domain = np.array(domain)
    domain = domain - domain[0]
    axis = scale(axis)

    distances = [abs(data - template(t)) for data, t in zip(axis, domain) if t < self.window_size]
    return sum(distances) / len(distances)

  def featurize(self, magData):
    """This method should take in magnetometer data and output a feature vector"""
    magData = np.array(magData)
    domain = magData[:,0] # first index is time, second is accuracy
    X = magData[:,2]
    Y = magData[:,3]
    Z = magData[:,4]
    magnitudes = np.sqrt(X**2 + Y**2 + Z**2)
    return [self.SumOfDifferences(domain, magnitudes, self.templates[3])]

class AllAxesTemplateSumOfDifferencesMagToVec(MagnetometerToFeatureVector):
  def __init__(self, templates):
    self.templates = templates
    self.window_size = 400 * 1e6

  def SumOfDifferences(self, domain, axis, template):
    domain = np.array(domain)
    domain = domain - domain[0]
    axis = scale(axis)
    distances = [abs(data - template(t)) for data, t in zip(axis, domain) if t < self.window_size]
    return sum(distances) / len(distances)

  def featurize(self, magData):
    """This method should take in magnetometer data and output a feature vector"""
    magData = np.array(magData)
    domain = magData[:,0] # first index is time, second is accuracy
    X = magData[:,2]
    Y = magData[:,3]
    Z = magData[:,4]
    magnitudes = np.sqrt(X**2 + Y**2 + Z**2)
    return [self.SumOfDifferences(domain, X, self.templates[0]),
            self.SumOfDifferences(domain, Y, self.templates[1]),
            self.SumOfDifferences(domain, Z, self.templates[2]),
            self.SumOfDifferences(domain, magnitudes, self.templates[3]),
            ]

class ManyFeaturesSumOfDifferencesMagToVec(MagnetometerToFeatureVector):
  def __init__(self, templates):
    self.templates = templates
    # self.window_size = 500 * 1e6
    self.window_size = 450 * 1e6

  def SumOfDifferences(self, domain, axis, template):
    domain = np.array(domain)
    domain = domain - domain[0]
    # axis = axis - axis[0]
    axis = scale(axis)

    distances = [abs(data - template(t))**2 for data, t in zip(axis, domain) if t < self.window_size]
    return sum(distances) / len(distances)

  def featurize(self, magData):
    """This method should take in magnetometer data and output a feature vector"""
    magData = np.array(magData)
    domain = magData[:,0] # first index is time, second is accuracy
    magData = magData[ domain < domain[0] + self.window_size ]
    magData = magData - magData[0,:]

    domain = magData[:,0] # first index is time, second is accuracy
    X = magData[:,2]
    Y = magData[:,3]
    Z = magData[:,4]
    magnitudes = np.sqrt(X**2 + Y**2 + Z**2)
    return [self.SumOfDifferences(domain, X, self.templates[0]),
            self.SumOfDifferences(domain, Y, self.templates[1]),
            self.SumOfDifferences(domain, Z, self.templates[2]),
            self.SumOfDifferences(domain, magnitudes, self.templates[3]),
            magnitudes[0] - magnitudes[-1],
            max(magnitudes),
            min(magnitudes),
            max(magnitudes) - min(magnitudes),
            ]

class RawMagnitudeManyFeaturesMagToVec(MagnetometerToFeatureVector):
  def __init__(self, templates):
    self.templates = templates
    # self.window_size = 500 * 1e6
    self.window_size = 450 * 1e6

  def SumOfDifferences(self, domain, axis, template):
    domain = np.array(domain)
    domain = domain - domain[0]
    # axis = axis - axis[0]
    axis = scale(axis)

    distances = [abs(data - template(t))**2 for data, t in zip(axis, domain) if t < self.window_size]
    return sum(distances) / len(distances)

  def featurize(self, magData):
    """This method should take in magnetometer data and output a feature vector"""
    magData = np.array(magData)
    domain = magData[:,0] # first index is time, second is accuracy
    magData = magData[ domain < domain[0] + self.window_size ]

    X = magData[:,2]
    Y = magData[:,3]
    Z = magData[:,4]
    raw_magnitudes = np.sqrt(X**2 + Y**2 + Z**2)
    gradM = np.gradient(raw_magnitudes)

    magData = magData - magData[0,:]

    domain = magData[:,0] # first index is time, second is accuracy
    X = magData[:,2]
    Y = magData[:,3]
    Z = magData[:,4]
    magnitudes = np.sqrt(X**2 + Y**2 + Z**2)
    return [self.SumOfDifferences(domain, X, self.templates[0]),
            self.SumOfDifferences(domain, Y, self.templates[1]),
            self.SumOfDifferences(domain, Z, self.templates[2]),
            self.SumOfDifferences(domain, magnitudes, self.templates[3]),
            raw_magnitudes[0] - raw_magnitudes[-1],
            raw_magnitudes[-1] - raw_magnitudes[0],
            abs(raw_magnitudes[0] - raw_magnitudes[-1]),
            max(raw_magnitudes),
            min(raw_magnitudes),
            max(raw_magnitudes) - min(raw_magnitudes),
            max(gradM)
            ]

class NegAndPosTemplatesMagToVec(MagnetometerToFeatureVector):
  def __init__(self, posTemplates, negTemplates):
    self.posTemplates = posTemplates
    self.negTemplates = negTemplates
    self.window_size = 450 * 1e6

    myFunc = lambda x : float(x) / self.window_size
    self.negTemplates = [myFunc] * 4

  def SumOfDifferences(self, domain, axis, template):
    domain = np.array(domain)
    domain = domain - domain[0]
    # axis = axis - axis[0]
    axis = scale(axis)

    distances = [abs(data - template(t))**2 for data, t in zip(axis, domain) if t < self.window_size]
    return sum(distances) / len(distances)

  def CosineSimilarity(self, domain, axis, template):
    domain = np.array(domain)
    domain = domain - domain[0]
    axis = scale(axis)
    otherVect = [template(t) for t in domain if t < self.window_size]
    distance = scipy.spatial.distance.cosine(axis, otherVect)
    # features = [f if not np.isnan(f) else 0 for f in features]
    # return features
    return distance if not np.isnan(distance) else 0

  def featurize(self, magData):
    """This method should take in magnetometer data and output a feature vector"""
    magData = np.array(magData)
    domain = magData[:,0] # first index is time, second is accuracy
    magData = magData[ domain < domain[0] + self.window_size ]

    X = magData[:,2]
    Y = magData[:,3]
    Z = magData[:,4]
    raw_magnitudes = np.sqrt(X**2 + Y**2 + Z**2)
    gradM = np.gradient(raw_magnitudes)

    magData = magData - magData[0,:]

    domain = magData[:,0] # first index is time, second is accuracy
    X = magData[:,2]
    Y = magData[:,3]
    Z = magData[:,4]
    magnitudes = np.sqrt(X**2 + Y**2 + Z**2)
    return [self.CosineSimilarity(domain, X, self.posTemplates[0]),
            self.CosineSimilarity(domain, Y, self.posTemplates[1]),
            self.CosineSimilarity(domain, Z, self.posTemplates[2]),
            self.CosineSimilarity(domain, magnitudes, self.posTemplates[3]),
            self.CosineSimilarity(domain, X, self.negTemplates[0]),
            self.CosineSimilarity(domain, Y, self.negTemplates[1]),
            self.CosineSimilarity(domain, Z, self.negTemplates[2]),
            self.CosineSimilarity(domain, magnitudes, self.negTemplates[3]),
            raw_magnitudes[0] - raw_magnitudes[-1],
            raw_magnitudes[-1] - raw_magnitudes[0],
            abs(raw_magnitudes[0] - raw_magnitudes[-1]),
            max(raw_magnitudes),
            min(raw_magnitudes),
            max(raw_magnitudes) - min(raw_magnitudes),
            max(gradM)
            ]

class KitchenSync(MagnetometerToFeatureVector):
  def __init__(self, posTemplates):
    self.posTemplates = posTemplates
    self.window_size = 400 * 1e6
    myFunc = lambda x : float(x) / self.window_size
    self.negTemplates = [myFunc] * 4

  def CosineSimilarity(self, domain, axis, template):
    domain = np.array(domain)
    domain = domain - domain[0]
    axis = scale(axis)
    otherVect = [template(t) for t in domain if t < self.window_size]
    distance = scipy.spatial.distance.cosine(axis, otherVect)
    return distance if not np.isnan(distance) else 0

  def featurize(self, magData):
    """This method should take in magnetometer data and output a feature vector"""
    magData = np.array(magData)
    domain = magData[:,0] # first index is time, second is accuracy
    magData = magData[ domain < domain[0] + self.window_size ]

    X = magData[:,2]
    Y = magData[:,3]
    Z = magData[:,4]
    raw_magnitudes = np.sqrt(X**2 + Y**2 + Z**2)
    gradM = np.gradient(raw_magnitudes)

    magData = magData - magData[0,:]

    domain = magData[:,0] # first index is time, second is accuracy
    X = magData[:,2]
    Y = magData[:,3]
    Z = magData[:,4]
    magnitudes = np.sqrt(X**2 + Y**2 + Z**2)
    return [self.CosineSimilarity(domain, X, self.posTemplates[0]),
            self.CosineSimilarity(domain, Y, self.posTemplates[1]),
            self.CosineSimilarity(domain, Z, self.posTemplates[2]),
            self.CosineSimilarity(domain, magnitudes, self.posTemplates[3]),
            self.CosineSimilarity(domain, X, self.negTemplates[0]),
            self.CosineSimilarity(domain, Y, self.negTemplates[1]),
            self.CosineSimilarity(domain, Z, self.negTemplates[2]),
            self.CosineSimilarity(domain, magnitudes, self.negTemplates[3]),
            raw_magnitudes[0] - raw_magnitudes[-1],
            raw_magnitudes[-1] - raw_magnitudes[0],
            abs(raw_magnitudes[0] - raw_magnitudes[-1]),
            max(raw_magnitudes),
            min(raw_magnitudes),
            max(raw_magnitudes) - min(raw_magnitudes),
            max(gradM)
            ]

class MagnitudeFeaturesDataToVec(MagnetometerToFeatureVector):
  def __init__(self):
    self.window_size = 450 * 1e6

  def featurize(self, magData):
    """This method should take in magnetometer data and output a feature vector"""
    magData = np.array(magData)
    domain = magData[:,0] # first index is time, second is accuracy
    magData = magData[ domain < domain[0] + self.window_size ]

    X = magData[:,2]
    Y = magData[:,3]
    Z = magData[:,4]
    raw_magnitudes = np.sqrt(X**2 + Y**2 + Z**2)
    gradM = np.gradient(raw_magnitudes)

    magData = magData - magData[0,:]

    domain = magData[:,0] # first index is time, second is accuracy
    X = magData[:,2]
    Y = magData[:,3]
    Z = magData[:,4]
    magnitudes = np.sqrt(X**2 + Y**2 + Z**2)
    return [
            raw_magnitudes[0] - raw_magnitudes[-1],
            raw_magnitudes[-1] - raw_magnitudes[0],
            abs(raw_magnitudes[0] - raw_magnitudes[-1]),
            max(raw_magnitudes),
            min(raw_magnitudes),
            max(raw_magnitudes) - min(raw_magnitudes),
            max(gradM)
            ]

class TestTemplateDifferencesMagToVec(MagnetometerToFeatureVector):
  def __init__(self, posTemplates):
    self.posTemplates = posTemplates
    self.window_size = 450 * 1e6

    myFunc = lambda x : float(x) / self.window_size
    self.negTemplates = [myFunc] * 4

  def SumOfDifferences(self, domain, axis, template):
    domain = np.array(domain)
    domain = domain - domain[0]
    axis = scale(axis)
    distances = [abs(data - template(t)) for data, t in zip(axis, domain) if t < self.window_size]
    return sum(distances) / len(distances)

  def SquareSumOfDifferences(self, domain, axis, template):
    domain = np.array(domain)
    domain = domain - domain[0]
    axis = scale(axis)
    distances = [abs(data - template(t))**2 for data, t in zip(axis, domain) if t < self.window_size]
    return sum(distances) / len(distances)

  def CosineSimilarity(self, domain, axis, template):
    domain = np.array(domain)
    domain = domain - domain[0]
    axis = scale(axis)
    otherVect = [template(t) for t in domain if t < self.window_size]
    distance = scipy.spatial.distance.cosine(axis, otherVect)
    return distance

  def featurize(self, magData):
    """This method should take in magnetometer data and output a feature vector"""
    magData = np.array(magData)
    domain = magData[:,0] # first index is time, second is accuracy
    magData = magData[ domain < domain[0] + self.window_size ]

    X = magData[:,2]
    Y = magData[:,3]
    Z = magData[:,4]
    raw_magnitudes = np.sqrt(X**2 + Y**2 + Z**2)
    gradM = np.gradient(raw_magnitudes)

    magData = magData - magData[0,:]

    domain = magData[:,0] # first index is time, second is accuracy
    X = magData[:,2]
    Y = magData[:,3]
    Z = magData[:,4]
    magnitudes = np.sqrt(X**2 + Y**2 + Z**2)
    features = [
            self.SumOfDifferences(domain, X, self.posTemplates[0]),
            self.SumOfDifferences(domain, Y, self.posTemplates[1]),
            self.SumOfDifferences(domain, Z, self.posTemplates[2]),
            self.SumOfDifferences(domain, magnitudes, self.posTemplates[3]),
            self.SumOfDifferences(domain, X, self.negTemplates[0]),
            self.SumOfDifferences(domain, Y, self.negTemplates[1]),
            self.SumOfDifferences(domain, Z, self.negTemplates[2]),
            self.SumOfDifferences(domain, magnitudes, self.negTemplates[3]),

            self.SquareSumOfDifferences(domain, X, self.posTemplates[0]),
            self.SquareSumOfDifferences(domain, Y, self.posTemplates[1]),
            self.SquareSumOfDifferences(domain, Z, self.posTemplates[2]),
            self.SquareSumOfDifferences(domain, magnitudes, self.posTemplates[3]),
            self.SquareSumOfDifferences(domain, X, self.negTemplates[0]),
            self.SquareSumOfDifferences(domain, Y, self.negTemplates[1]),
            self.SquareSumOfDifferences(domain, Z, self.negTemplates[2]),
            self.SquareSumOfDifferences(domain, magnitudes, self.negTemplates[3]),

            self.CosineSimilarity(domain, X, self.posTemplates[0]),
            self.CosineSimilarity(domain, Y, self.posTemplates[1]),
            self.CosineSimilarity(domain, Z, self.posTemplates[2]),
            self.CosineSimilarity(domain, magnitudes, self.posTemplates[3]),
            self.CosineSimilarity(domain, X, self.negTemplates[0]),
            self.CosineSimilarity(domain, Y, self.negTemplates[1]),
            self.CosineSimilarity(domain, Z, self.negTemplates[2]),
            self.CosineSimilarity(domain, magnitudes, self.negTemplates[3]),
            ]
    features = [f if not np.isnan(f) else 0 for f in features]
    return features

class CloseToOriginal(MagnetometerToFeatureVector):
  def __init__(self, T1=30, T2=130):
    self.T1 = 30
    self.T2 = 130
    self.segment_time = 200 # ms

  def featurize(self, data):
    """This method should take in magnetometer data and output a feature vector"""
    segment_time_ns = self.segment_time * 1e6 # convert to nanoseconds
    window_size = segment_time_ns * 2

    data = np.array(data)
    domain = data[:,0] # first index is time, second is accuracy
    # magData = magData[ domain < domain[0] + self.window_size ]

    segment1 = data[(domain <= domain[0] + segment_time_ns)]
    segment2 = data[(domain > domain[0] + segment_time_ns) & (domain <= domain[0] + window_size)]
    # window = data[(domain > sensorTime - window_size) & (domain <= sensorTime)]
    if len(segment1) == 0 or len(segment2) == 0:
      return [0,0]

    S0 = segment2[-1, 2:5]
    offsets1 = segment1[:, 2:5] - S0
    offsets2 = segment2[:, 2:5] - S0
    norms1 = [np.linalg.norm(row) for row in offsets1]
    norms2 = [np.linalg.norm(row) for row in offsets2]

    return [min(norms1), max(norms2)]

class ThreePartFeaturizer(MagnetometerToFeatureVector):
  def __init__(self, T1=30, T2=130):
    self.segment1_time = 100
    self.segment2_time = 200 # ms
    self.segment3_time = 100 # ms

  def featurize(self, data):
    """This method should take in magnetometer data and output a feature vector"""
    segment_time1_ns = self.segment1_time * 1e6 # convert to nanoseconds
    segment_time2_ns = self.segment2_time * 1e6 # convert to nanoseconds
    segment_time3_ns = self.segment3_time * 1e6 # convert to nanoseconds

    data = np.array(data)
    domain = data[:,0] # first index is time, second is accuracy

    segment1 = data[(domain <= domain[0] + segment_time1_ns)]
    segment2 = data[(domain > domain[0] + segment_time1_ns) &
                    (domain <= domain[0] + segment_time1_ns + segment_time2_ns)]
    segment3 = data[(domain > domain[0] + segment_time1_ns + segment_time2_ns) &
                  (domain <= domain[0] + segment_time1_ns + segment_time2_ns + segment_time3_ns)]

    if len(segment1) == 0 or len(segment2) == 0 or len(segment3) == 0:
      return [0,0,0]

    S0 = segment2[-1, 2:5]
    offsets1 = segment1[:, 2:5] - S0
    offsets2 = segment2[:, 2:5] - S0
    offsets3 = segment3[:, 2:5] - S0
    norms1 = [np.linalg.norm(row) for row in offsets1]
    norms2 = [np.linalg.norm(row) for row in offsets2]
    norms3 = [np.linalg.norm(row) for row in offsets3]

    return [max(norms1), max(norms2), max(norms3)]

class WindowFeaturizer(MagnetometerToFeatureVector):
  def __init__(self, T1=30, T2=130):
    self.segment_time = 200 # ms

  def featurize(self, data):
    """This method should take in magnetometer data and output a feature vector"""
    segment_time_ns = self.segment_time * 1e6 # convert to nanoseconds
    window_size = segment_time_ns * 2

    data = np.array(data)
    domain = data[:,0] # first index is time, second is accuracy
    # magData = magData[ domain < domain[0] + self.window_size ]

    segment1 = data[(domain <= domain[0] + segment_time_ns)]
    segment2 = data[(domain > domain[0] + segment_time_ns) & (domain <= domain[0] + window_size)]
    if len(segment1) == 0 or len(segment2) == 0:
      return np.array([0,0,0,0,0,0,0,0,0,0,0,0])

    S0 = segment2[-1, 2:5]
    offsets1 = segment1[:, 2:5] - S0
    offsets2 = segment2[:, 2:5] - S0
    norms1 = [np.linalg.norm(row) for row in offsets1]
    norms2 = [np.linalg.norm(row) for row in offsets2]

    window = data[(domain <= domain[0] + window_size)]
    window = window - window[0,:]
    norms_scaled = [np.linalg.norm(row[2:5]) for row in window]
    # X = window[:,2]
    # Y = window[:,3]
    # Z = window[:,4]
    # magnitudes = np.sqrt(X**2 + Y**2 + Z**2)
    scaled_magnitudes = np.array(scale(norms_scaled))
    scaled_segment1 = np.array(scaled_magnitudes[(window[:,0] < segment_time_ns)])
    scaled_segment2 = np.array(scaled_magnitudes[(window[:,0] > segment_time_ns) & (window[:,0] <= window_size)])

    # print len(norms1), len(norms2)
    # print len(scaled_segment1), len(scaled_segment2)

    return np.array([
            min(norms1),
            max(norms1),
            np.mean(norms1),
            min(norms2),
            max(norms2),
            np.mean(norms2),

            min(scaled_segment1),
            max(scaled_segment1),
            np.mean(scaled_segment1),
            min(scaled_segment2),
            max(scaled_segment2),
            np.mean(scaled_segment2),
            ])


def GenerateData(runDataList, DataToVectorObj):
  X, Y = [], []
  for runData in runDataList:
    # print runData['filename']
    features = DataToVectorObj.featurize(runData['magnetometer'])

    if float('NaN') in features or float('inf') in features or float('-inf') in features:
      print runData['filename']
    if len(filter(np.isnan, features)) > 1:
      print runData['filename']
    # print np.array(features)
    X.append(features)
    if len(runData['labels']) > 0:
      Y.append(1.)
    else:
      Y.append(0)
  return np.array(X), np.array(Y)

def TrainDetectorOnData(runDataList):
  # train, test = sklearn.cross_validation.train_test_split(runDataList)
  positives = [rd for rd in runDataList if len(rd['labels']) > 0]
  posTemplates = CreateTemplates(positives)
  negatives = [rd for rd in runDataList if len(rd['labels']) == 0]
  negTemplates = CreateTemplates(negatives)

  # DataToVector = ManyFeaturesSumOfDifferencesMagToVec(templates)
  # DataToVector = RawMagnitudeManyFeaturesMagToVec(templates)
  # DataToVector = NegAndPosTemplatesMagToVec(posTemplates, negTemplates)
  # DataToVector = CloseToOriginal()
  # DataToVector = ThreePartFeaturizer()
  DataToVector = WindowFeaturizer()
  # DataToVector = TestTemplateDifferencesMagToVec(posTemplates)

  trainX, trainY = GenerateData(runDataList, DataToVector)

  # print trainX

  # clf = sklearn.svm.LinearSVC()
  clf = sklearn.svm.SVC(kernel='linear')
  # clf = sklearn.linear_model.LogisticRegression()
  clf.fit(trainX, trainY)
  print clf.coef_

  detector = MLDetector()
  detector.clf = clf
  detector.MagnetToVectorObj = DataToVector

  return detector

def PlotThresholds(runData, T1, T2, segment_size=200):
  pl.figure()
  pl.title(runData['systemInfo']['Build.MODEL'] + " " + runData['filename'] + " Thresholds")

  data = np.array(runData['magnetometer'])
  # accuracyChanges = np.array(runData['onAccuracyChangedData'])

  domain = data[:,0] # first index is time, second is accuracy
  domain = domain * 1e9
  # accuracyData = data[:,1]
  # X = data[:,2]
  # Y = data[:,3]
  # Z = data[:,4]
  # mag = np.sqrt(X**2 + Y**2 + Z**2)

  min_seg1 = []
  max_seg2 = []

  segment_time_ns = segment_size * 1e6
  window_size = segment_time_ns * 2
  newDomain = domain[domain > domain[0]+window_size]
  print len(newDomain)

  for sensorTime in newDomain:
    segment1 = data[(domain > sensorTime - window_size) & (domain <= sensorTime - segment_time_ns)]
    segment2 = data[(domain > sensorTime - segment_time_ns) & (domain <= sensorTime)]
    # window = data[(domain > sensorTime - window_size) & (domain <= sensorTime)]
    # segment1 = data[(domain < sensorTime)][-40:-20]
    # segment2 = data[(domain < sensorTime)][-20:]

    # # wait to fire if we don't have any sensor events
    # if len(segment1) == 0 or len(segment2) == 0:
    #   continue

    # if sensorTime - lastFiring < waitTime: # wait for a full window before looking again
    #   continue

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

def PlotData(runData, optPlotData=False, inputLabels=[]):
  """Plots the data from a run"""

  pl.figure()
  pl.title(runData['systemInfo']['Build.MODEL'] + " " + runData['filename'])

  magData = np.array(runData['magnetometer'])
  # accuracyChanges = np.array(runData['onAccuracyChangedData'])

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
        pl.scatter(magDomain[index], 0, color=accuracyColors[int(accuracyData[index])])
    # for t, value in accuracyData: # plot accuracy changed events
    #   pl.scatter([t],[0], color=accuracyColors[value])
      # pl.axvline(t, color=accuracyColors[value])

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

  xp, yp, zp, mp = CreateTemplates(positives)
  newT = range(0,450000000,1000000)
  axarr[0, 0].plot(newT, [xp(t) for t in newT], color='red')
  axarr[0, 1].plot(newT, [yp(t) for t in newT], color='red')
  axarr[0, 2].plot(newT, [zp(t) for t in newT], color='red')
  axarr[0, 3].plot(newT, [mp(t) for t in newT], color='red')

  xp, yp, zp, mp = CreateTemplates(negatives)
  newT = range(0,450000000,1000000)
  axarr[1, 0].plot(newT, [xp(t) for t in newT], color='red')
  axarr[1, 1].plot(newT, [yp(t) for t in newT], color='red')
  axarr[1, 2].plot(newT, [zp(t) for t in newT], color='red')
  axarr[1, 3].plot(newT, [mp(t) for t in newT], color='red')

  pl.show()

def scale(vec, lowerPercentile=2, upperPercentile=98):
  lo = np.percentile(vec, lowerPercentile)
  hi = np.percentile(vec, upperPercentile)
  if hi == lo:
    hi = 1
    lo = 0
  return np.abs(vec / (hi - lo))

def SubPlotFeature(runData, axarr):
  magData = np.array(runData['magnetometer'])
  magData = magData - magData[0,:] # normalize everything based on the first value
  # magData = magData - magData[-1,:] # normalize everything based on the last value
  magDomain = magData[:,0] # first index is time, second is accuracy
  X = magData[:,2]
  Y = magData[:,3]
  Z = magData[:,4]
  mag = np.sqrt(X**2 + Y**2 + Z**2)
  magDomain = magDomain - magDomain[0] # put in same timescale

  X = scale(X)
  Y = scale(Y)
  Z = scale(Z)
  mag = scale(mag)

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
  """Plot magnitude of snippet in the same plot, red if positive, blue otherwise"""
  magData = np.array(runData['magnetometer'])
  magData = magData - magData[0,:] # normalize everything based on the first value
  magDomain = magData[:,0] # first index is time, second is accuracy
  X = magData[:,2]
  Y = magData[:,3]
  Z = magData[:,4]
  mag = np.sqrt(X**2 + Y**2 + Z**2)

  magDomain = magDomain - magDomain[0] # put in same timescale

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
    magData = magData - magData[0,:] # normalize everything based on the first value
    # magData = magData - magData[-1,:] # normalize everything based on the last value
    magDomain = magData[:,0] # first index is time, second is accuracy
    X = magData[:,2]
    Y = magData[:,3]
    Z = magData[:,4]
    mag = np.sqrt(X**2 + Y**2 + Z**2)
    magDomain = magDomain - magDomain[0] # put in same timescale

    mag = scale(mag)

    fcns.append(scipy.interpolate.interp1d(magDomain, mag, kind='cubic'))
    pl.plot(magDomain, mag, alpha=0.2)

  numFcns = float(len(fcns))
  BigF = lambda x: sum([f(x) for f in fcns]) / numFcns
  newX = range(0,450000000,1000000)
  newY = [BigF(x) for x in newX]
  pl.plot(newX, newY, color='red')

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

def decalibrate(runData):
  """Given calibrated magnetometer data, returns an uncalibrated version"""

  magData = np.array(runData['magnetometer'])
  accuracyChanges = np.array(runData['onAccuracyChangedData'])

  magDomain = magData[:,0] # first index is time, second is accuracy
  accuracyData = magData[:,1] # second is accuracy
  X = magData[:,2]
  Y = magData[:,3]
  Z = magData[:,4]

  calibrationTimes = []
  for index in xrange(1,len(accuracyData)-1):
    if accuracyData[index] > accuracyData[index-1]:
      # print int(magDomain[index]), int(magDomain[index-1])
      calibrationTimes.append((magDomain[index] + magDomain[index-1])/2.)

  for cTime in calibrationTimes:
    addMask = (cTime < magDomain) # indices for everything after calibration time

    beforeVect = magData[magData[:,0] < cTime][-1] # last datum before calibration
    afterVect = magData[magData[:,0] > cTime][0] # first datum after calibration
    difference = beforeVect - afterVect

    X[addMask] = X[addMask] + difference[2]
    Y[addMask] = Y[addMask] + difference[3]
    Z[addMask] = Z[addMask] + difference[4]

  newMagnetometerData = np.column_stack((magDomain, accuracyData, X, Y, Z))

  runData["magnetometer"] = newMagnetometerData
  return runData

def evaluateDetector(runData, detector, msWindow=600, optPlotData=False):
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

  output = detector.detect(runData)
  if optPlotData:
    PlotData(runData, inputLabels=output)

  nsWindow = msWindow * 1000000 # use nanoseconds for everything
  # truePos, falsePos, falseNeg = 0, 0, len(labels)
  # for labelTime, label in labels:
  #   i = bisect.bisect_left(output, labelTime) # leftmost value of output < labelTime
  #   if i < len(output) and output[i] - labelTime < nsWindow:
  #     truePos += 1
  #     falseNeg -= 1
  #     continue
  #   if i != 0 and labelTime - output[i-1] < nsWindow:
  #     truePos += 1
  #     falseNeg -= 1
  #     continue

  #   falsePos += 1
    # output[i] # closest label greater than labelTime
    # output[i-1] # closest label smaller than labelTime

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
    results = evaluateDetector(runData, detector, optPlotData=optPlotData)
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

  pl.show() # shows all the plots from above
  return phoneTabulatedResults

def OptimizeDetector(runDataList, detector, startArgs):
  """TODO(cjr): document"""
  fun = lambda args : helpOptimize(runDataList, detector, args)
  optResult = scipy.optimize.fmin_powell(fun, startArgs)
  print optResult

def helpOptimize(runDataList, detector, args):
  detector.setParameters(args)
  errors = 0
  for runData in runDataList:
    tp, fp, fn = evaluateDetector(runData, detector, optPlotData=False)
    errors += len(fp) + len(fn)
  return errors

def OptimizeDetector2(runDataList, detector, startArgs):
  def helpOptimize2(runDataList, detector, args):
    detector.setParameters(args)
    cost = 0
    for runData in runDataList:
      if len(runData['labels']):
        cost += detector.evaluateCost(runData, True)
      else:
        cost += detector.evaluateCost(runData, False)
    return cost

  fun = lambda args : helpOptimize2(runDataList, detector, args)
  optResult = scipy.optimize.fmin_powell(fun, startArgs)
  print optResult

def helpOptimize2(runDataList, detector, args):
  detector.setParameters(args)
  cost = 0
  for runData in runDataList:
    if len(runData['labels']):
      cost += detector.evaluateCost(runData, True)
    else:
      cost += detector.evaluateCost(runData, False)
  return cost

def sampleMagnetometer(runData):
  runData['magnetometer'] = runData['magnetometer'][::4] # sample every other
  return runData

def MakeDummyRunData():
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

def preprocessRunData(runDataList):
  # newList = [sampleMagnetometer(decalibrate(runData)) for runData in runDataList]
  newList = [decalibrate(runData) for runData in runDataList]
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

def WriteToSpreadsheet(message, phoneTabulatedResults):
  """Write our latest results to a spreadsheet

  The header for this spreadsheet is:
  time,description,TP,FP,FN,Nexus4 TP,Nexus4 FP,Nexus4 FN,HTC1 TP,HTC1 FP,HTC1 FN,s4 TP,s4 FP,s4 FN
  The file is kept the same, with the filename stored in the constant RESULTS.
  """
  t = time.strftime("%Y-%m-%d %H:%M:%S")
  description = message
  tps = sum([v[0] for v in phoneTabulatedResults.values()])
  fps = sum([v[1] for v in phoneTabulatedResults.values()])
  fns = sum([v[2] for v in phoneTabulatedResults.values()])
  nexus4_tp, nexus4_fp, nexus4_fn = phoneTabulatedResults["Nexus 4"]
  htc1_tp, htc1_fp, htc1_fn = phoneTabulatedResults["HTC One"]
  s4_tp, s4_fp, s4_fn = phoneTabulatedResults["SCH-I545"]
  out_str = ','.join(map(str, [t, description, tps, fps, fns, nexus4_tp, nexus4_fp,
    nexus4_fn, htc1_tp, htc1_fp, htc1_fn, s4_tp, s4_fp, s4_fn]))
  with open(RESULTS_FNAME, 'a') as f:
    f.write(out_str + '\n')

def WriteDataList(runDataList):
  """Writes a list of runData objects to files"""
  for rd in runDataList:
    with open(rd['filename'], 'w') as f:
      json.dump(rd, f)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Read JSON formatted recordings of ' + \
    'magnetometer data, and show results of various algorithms labeling them.')
  parser.add_argument('filelist', metavar='filename', type=str, nargs='+',
    help='list of files to process')
  parser.add_argument('--justplot', dest='justplot', action='store_true',
    default=False, help='Plot the specified files without running labeling algorithms.')
  parser.add_argument('--optimize', dest='optimize', action='store_true',
    default=False, help='Train the detector on the input files')
  parser.add_argument('--plot', dest='display', action='store_true',
    default=False, help='Plot the results of the detector on the given files')
  parser.add_argument('--no-preprocess', dest='nodecalibrate', action='store_true',
    default=False, help="Don't run decalibration on the data.")
  parser.add_argument('--plot-features', dest='plotFeatures', action='store_true',
    default=False, help="Show positive and negative data on the same plot.")
  parser.add_argument('--save-train', dest='clfOutputName', type=str, nargs=1,
    help="Train an algorithm on the given data and save the classifier with the given filename.")
  parser.add_argument('--run-ml', dest='clfIntputName', type=str, nargs=1,
    help="Load a model from the specified file and run the detector")
  parser.add_argument('--train', dest='trainingset', type=str, nargs='+', #TODO: kill
    help="Train algorithm on this list of files and run the result.", default=[])
  parser.add_argument('--record', dest='record', nargs=1, metavar='MESSAGE',
    help='Record the results in ' + RESULTS_FNAME + '. Takes a description string as input') # TODO: kill
  args = parser.parse_args()

  # give options and files or directories containing files
  if len(sys.argv) < 2:
    print "Please enter a JSON file, a directory, or list of files"
    exit()

  # Get the data we'll use
  runDataList = GetRunDataFromArgs(args.filelist)

  if len(runDataList) == 0:
    print "Please enter a JSON file, a directory, or a list of files"
    exit()

  # detector = OriginalDetector()
  detector = TimeWindowDetector()
  # detector.T1 = 40
  # detector.T2 = 30
  # detector.T1 = 9.8
  # detector.T2 = 405
  # self.segment_time = 204 # ms

  detector.T1 = 8.56265499e+01
  detector.T2 = 3.29885044e+01

  # [   2.55166172   16.98332041  199.9999998 ]
  # detector = ScaledTimeWindowDetector()
  # detector = ScaledThreeWindowDetector()
  # detector = TrainDetectorOnData(runDataList)
  # detector = MagGradientThresholdDetector()
  # detector = VectorChangeDetector()
  # detector.setParameters([ 2.00387233,  -2.06634197,  11.02856022])
  # detector = SimpleMagnitudeDetector()
  # detector = MagnitudeTemplateDetector()
  # detector.template = CreateTemplates(runDataList)[-1]

  if not args.nodecalibrate:
    print "Decalibrating"
    runDataList = preprocessRunData(runDataList)

  if args.justplot:
    PlotList(runDataList, optPlotData=True)
  elif args.optimize:
    OptimizeDetector2(runDataList, detector, detector.args)
    # testDetector(detector, runDataList) # show results of optimization
  elif args.record:
    phoneTabulatedResults = testDetector(detector, runDataList)
    WriteToSpreadsheet(args.record[0], phoneTabulatedResults)
  elif args.plotFeatures:
    # PlotInterpolatedSnips(runDataList)
    # pl.show()
    PlotFeatures(runDataList)
    # PlotSnips(runDataList)
  elif args.clfOutputName:
    mldetector = TrainDetectorOnData(runDataList)
    with open(args.clfOutputName[0], 'w') as f:
      dill.dump(mldetector, f)
  elif args.clfIntputName:
    with open(args.clfIntputName[0]) as f:
      mldetector = dill.load(f)
    if args.display:
      testDetector(mldetector, runDataList, optPlotData=True)
    else:
      testDetector(mldetector, runDataList, optPlotData=False)
  elif len(args.trainingset) > 0:
    trainRunData = GetRunDataFromArgs(args.trainingset)
    print "Training on %d samples." % len(trainRunData)
    mldetector = TrainDetectorOnData(trainRunData)
    if args.display:
      testDetector(mldetector, runDataList, optPlotData=True)
    else:
      testDetector(mldetector, runDataList, optPlotData=False)
  elif args.display:
    testDetector(detector, runDataList, optPlotData=True)
  else: # this just prints the results
    testDetector(detector, runDataList)
