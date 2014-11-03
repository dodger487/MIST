#!/usr/bin/env python

# Chris Riederer
# Google, Inc
# 2014-08-19

import test_detect as t

import glob
import numpy as np
import scipy.interpolate

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

    mag = t.scale(mag)

    fcns.append(scipy.interpolate.interp1d(magDomain, mag, kind='cubic'))
    t.pl.plot(magDomain, mag, alpha=0.2)

  numFcns = float(len(fcns))
  BigF = lambda x: sum([f(x) for f in fcns]) / numFcns
  newX = range(0,500000000,1000000)
  newY = [BigF(x) for x in newX]
  t.pl.plot(newX, newY, color='red')


# things to plot:
#   snips w/ template for each axis
#   distances of snips from template


# TODO: NOT JUST MMX
# decalSnipList = t.GetRunDataFromArgs(glob.glob('snips/!(htc*|s4*|nexus*|moto*)'))
decalSnipList = t.GetRunDataFromArgs(glob.glob('snips/[!h]*.json')) # not htc
# decalSnipList = t.GetRunDataFromArgs(glob.glob('snips/mmx*.json')) # not htc
# decalSnipList = t.GetRunDataFromArgs(glob.glob('snips/motox-negative-cjr-4*.json')) # not htc
# decalSnipList = t.GetRunDataFromArgs(glob.glob('snips/[!hs]*.json')) # not htc
# decalSnipList = t.GetRunDataFromArgs(glob.glob('snips/mmx*.json'))
decalSnipList = t.preprocessRunData(decalSnipList)
print len(decalSnipList)

# break into positives and negatives
positives = [rd for rd in decalSnipList if len(rd['labels']) > 0]
negatives = [rd for rd in decalSnipList if len(rd['labels']) == 0]


# xt, yt, zt, magt = t.CreateTemplates(decalSnipList)
# newX = range(0,500000000,1000000)
# newY = [magt(x) for x in newX]
# t.pl.plot(newX, newY)

# templates = t.CreateTemplates(decalSnipList)
# featurizer = t.ManyFeaturesSumOfDifferencesMagToVec(templates)
# featurizer = t.MagnitudeFeaturesDataToVec()
featurizer = t.WindowFeaturizer()
pos_distances = np.array([featurizer.featurize(rd['magnetometer']) for rd in positives])
neg_distances = np.array([featurizer.featurize(rd['magnetometer']) for rd in negatives])

# for ind, x in enumerate(neg_distances):
#   print len(x), type(x), x, negatives[ind]['filename']

# print neg_distances[:,0]

# overlay histogram
t.pl.figure()
n, bins, patches = t.pl.hist(pos_distances[:,-3], color='red', alpha=0.4)
t.pl.hist(neg_distances[:,-3], color='blue', bins=bins, alpha=0.4)
# n, bins, patches = t.pl.hist([p for p in pos_distances[:,-3] if p < 100], color='red', alpha=0.4)
# t.pl.hist([n for n in neg_distances[:,-3] if n < 100], color='blue', alpha=0.4)
# t.pl.hist(, color='blue')

# side-by-side histogram
# t.pl.figure()
# t.pl.hist([pos_distances[:,2], neg_distances[:,2]], histtype='bar', color=['red','blue'])
# t.pl.figure()
# t.pl.hist([pos_distances[:,3], neg_distances[:,3]], histtype='bar', color=['red','blue'])
# t.pl.figure()
# t.pl.hist([pos_distances[:,4], neg_distances[:,4]], histtype='bar', color=['red','blue'])
# t.pl.figure()
# t.pl.hist([pos_distances[:,5], neg_distances[:,5]], histtype='bar', color=['red','blue'])
# t.pl.figure()
# t.pl.hist([pos_distances[:,6], neg_distances[:,6]], histtype='bar', color=['red','blue'])

# t.pl.boxplot([pos_distances[:,-1], neg_distances[:,-1]], sym='')
# t.pl.figure()
# t.pl.boxplot([pos_distances[:,-2], neg_distances[:,-2]], sym='')
# t.pl.figure()
# t.pl.boxplot([pos_distances[:,2], neg_distances[:,2]], sym='')
# t.pl.figure()
# t.pl.boxplot([pos_distances[:,3], neg_distances[:,3]], sym='')

# t.pl.figure()
# PlotInterpolatedSnips(decalSnipList)

# t.pl.plot(xt)
t.pl.show()
