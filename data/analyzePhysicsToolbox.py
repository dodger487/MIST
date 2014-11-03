#!/usr/bin/env python

# Chris Riederer
# 2014-08-13

"""This script is to look at the data coming from Physics Toolbox Magnetometer.
  It's to help debug the magnet button on Google Cardboard.
"""

import test_detect as t

import sys

if len(sys.argv) < 2:
  print "Please provide the name of the file you'd like to analyze."

runData = t.MakeDummyRunData()

with open(sys.argv[1]) as f:
  magdata = []
  f.readline()
  f.readline()
  for line in f:
    empty, time, x, y, z = line.strip().split(',')
    magdata.append(map(float, [time, 0, x, y, z]))

runData['magnetometer'] = magdata

runDataList = [runData]
runDataList = t.preprocessRunData(runDataList)

detector = t.TimeWindowDetector()
detector.segment_time = 175
# detector = t.OriginalDetector()

# t.PlotData(runData)
# t.pl.show()

# t.testDetector(detector, runDataList, optPlotData=True)
t.PlotThresholds(runData, 30, 130, segment_size=400)
