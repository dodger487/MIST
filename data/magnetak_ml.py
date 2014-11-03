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

"""Contains everything related to machine learning in magnetak"""

import magnetak_detectors
import magnetak_util

import numpy as np
import scipy
import scipy.spatial
import scipy.spatial.distance
import sklearn
import sklearn.cross_validation
import sklearn.svm
import sklearn.linear_model

class MLDetector(magnetak_detectors.Detector):
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
    axis = magnetak_util.scale(axis)

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
    axis = magnetak_util.scale(axis)
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
    axis = magnetak_util.scale(axis)

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
    axis = magnetak_util.scale(axis)

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
    axis = magnetak_util.scale(axis)

    distances = [abs(data - template(t))**2 for data, t in zip(axis, domain) if t < self.window_size]
    return sum(distances) / len(distances)

  def CosineSimilarity(self, domain, axis, template):
    domain = np.array(domain)
    domain = domain - domain[0]
    axis = magnetak_util.scale(axis)
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
    axis = magnetak_util.scale(axis)
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
    axis = magnetak_util.scale(axis)
    distances = [abs(data - template(t)) for data, t in zip(axis, domain) if t < self.window_size]
    return sum(distances) / len(distances)

  def SquareSumOfDifferences(self, domain, axis, template):
    domain = np.array(domain)
    domain = domain - domain[0]
    axis = magnetak_util.scale(axis)
    distances = [abs(data - template(t))**2 for data, t in zip(axis, domain) if t < self.window_size]
    return sum(distances) / len(distances)

  def CosineSimilarity(self, domain, axis, template):
    domain = np.array(domain)
    domain = domain - domain[0]
    axis = magnetak_util.scale(axis)
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
    scaled_magnitudes = np.array(magnetak_util.scale(norms_scaled))
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
  """Given a list of runData objects, and a Featurizer object, returns a list
      of feature vectors and labels"""
  X, Y = [], []
  for runData in runDataList:
    # print runData['filename']
    features = DataToVectorObj.featurize(runData['magnetometer'])

    # if float('NaN') in features or float('inf') in features or float('-inf') in features:
    #   print runData['filename']
    # if len(filter(np.isnan, features)) > 1:
    #   print runData['filename']
    X.append(features)
    if len(runData['labels']) > 0:
      label = runData['labels'][0][1] # label of first labeled item
      Y.append(label)
    else:
      Y.append(0)
  return np.array(X), np.array(Y)

def TrainDetectorOnData(runDataList, featurizer):
  """Given a list of runData objects and a Featurizer, creates training data
      and trains an algorithm. Returns a trained MLDetector object.
  """
  # TODO(cjr): make options for using other algorithms

  # train, test = sklearn.cross_validation.train_test_split(runDataList)
  positives = [rd for rd in runDataList if len(rd['labels']) > 0]
  posTemplates = magnetak_util.CreateTemplates(positives)
  negatives = [rd for rd in runDataList if len(rd['labels']) == 0]
  negTemplates = magnetak_util.CreateTemplates(negatives)

  trainX, trainY = GenerateData(runDataList, featurizer)

  # clf = sklearn.svm.LinearSVC()
  # clf = sklearn.svm.SVC(kernel='linear')
  clf = sklearn.linear_model.LogisticRegression()
  clf.fit(trainX, trainY)
  print clf.coef_

  detector = MLDetector()
  detector.clf = clf
  detector.MagnetToVectorObj = featurizer

  return detector
