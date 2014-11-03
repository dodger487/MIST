#!/usr/bin/env python

# Chris Riederer
# Google, Inc
# 2014-08-11

import test_detect as t

import sys

def patentPlot(rd, title):
  rd['labels'] = [] # We don't want to plot labels, so just remove them
  t.PlotData(rd)
  t.pl.title(title)
  t.pl.xlabel("Time (ns)") # show axes labels
  t.pl.ylabel("Magnetometer Data ($\mu$T)")
  t.pl.legend(["X","Y","Z","Magnitude"], loc="lower left")

runDataList = []
fnames = ["cleaned/nexus4-negative-cjr-3.json",
          "cleaned/nexus4-positive-cjr-1.json",
          "cleaned/nexus4-positive-cjr-4.json",
          "cleaned/nexus4-positive-gavin-2.json",
          ]
for fname in fnames:
  runDataList = t.addToList(fname, runDataList)

for rd in runDataList:
  patentPlot(rd, "Magnetometer Data, Before Reverse Calibration")
  ylims = t.pl.gca().get_ylim()
  rd = t.decalibrate(rd)
  patentPlot(rd, "Magnetometer Data, After Reverse Calibration")
  t.pl.gca().set_ylim(ylims)

t.pl.show()
