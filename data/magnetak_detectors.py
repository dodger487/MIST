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


"""Contains the detectors used in MagnetAK"""

import numpy as np

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


class TimeWindowDetector(Detector):
  """The original detector for paperscope modified to use time instead of samples"""

  def __init__(self):
    self.T1 = 30
    self.T2 = 130
    self.segment_time = 200 # ms

    self.waitTime = 350 # ms

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
    # print runData['filename']
    for sensorTime in domain[domain > domain[0]+window_size]:
      segment1 = data[(domain > sensorTime - window_size) & (domain <= sensorTime - segment_time_ns)]
      segment2 = data[(domain > sensorTime - segment_time_ns) & (domain <= sensorTime)]
      window = data[(domain > sensorTime - window_size) & (domain <= sensorTime)]

      # wait to fire if we don't have any sensor events
      if len(segment1) == 0 or len(segment2) == 0:
        continue

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

class VectorDistanceDetector(Detector):
  """TODO"""
  def __init__(self):
    self.window_size = 100 # window size in milliseconds
    self.X = 0
    self.Y = 0
    self.Z = 0
    self.threshold = 0.1

    self.args = [self.X, self.Y, self.Z, self.threshold, self.window_size]

  def setParameters(self, args):
    self.X, self.Y, self.Z, self.threshold, self.window_size = args
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
      distance = np.sqrt(sum(difference**2))
      # print difference
      if distance < self.threshold:
        t = domain[nextIndex]
        detections.append(t)
        lastTime = currentTime

      # Store cost for each window. Cost=0 if it fires and is positive.
      if isPositive:
        cost = max(0, self.threshold - distance)
      else:
        cost = max(0, distance - self.threshold)
      history.append(cost)
    out_cost = min(history) if isPositive else max(history)
    return (detections, out_cost)

  def detect(self, runData):
    detections, out_cost = self.detectAndEvaluate(runData)
    return detections

  def evaluateCost(self, runData, isPositive):
    detections, out_cost = self.detectAndEvaluate(runData, isPositive)
    return out_cost
