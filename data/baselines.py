# Chris Riederer
# Google, Inc
# 2014-08-15

"""Record baselines and show improvements for working on magnet"""

import test_detect as t

import glob

cleanedList = t.GetRunDataFromArgs(glob.glob('cleaned/*.json'))
snipList = t.GetRunDataFromArgs(glob.glob('snips/*.json'))

decalCleanedList = t.GetRunDataFromArgs(glob.glob('cleaned/*.json'))
decalCleanedList = t.preprocessRunData(decalCleanedList)
decalSnipList = t.GetRunDataFromArgs(glob.glob('snips/*.json'))
decalSnipList = t.preprocessRunData(decalSnipList)

def tryDetector(detector, detectorLabel):
  print detectorLabel
  print "Running on snippets"
  t.testDetector(detector, snipList, printFileData=False)
  print

  print detectorLabel
  print "Running on full files"
  t.testDetector(detector, cleanedList, printFileData=False)
  print

  print detectorLabel
  print "Running on snippets, decalibrated"
  t.testDetector(detector, decalSnipList, printFileData=False)
  print

  print detectorLabel
  print "Running on full files, decalibrated"
  t.testDetector(detector, decalCleanedList, printFileData=False)
  print


def runAll():
  """Run every detector and print out the results"""
  detector = t.OriginalDetector()
  tryDetector(detector, "ORIGINAL DETECTOR")

  detector = t.TimeWindowDetector()
  tryDetector(detector, "TIME WINDOW DETECTOR")

  detector = t.VectorChangeDetector()
  tryDetector(detector, "VECTOR CHANGE DETECTOR")


if __name__ == '__main__':
  runAll()
