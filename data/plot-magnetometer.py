#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import sys
import re

if len(sys.argv) != 2:
  print 'data.txt required'
  sys.exit(1)

path = sys.argv[1]
f = open(path, 'r')
x = []
y = []
z = []
avg = []
magnitudes = []

for line in f.readlines():
  if line == '\n':
    continue
  axes = re.split('[, ]', line)
  axes = [_ for _ in axes if _ != '']
  print axes
  axes = np.array(map(float, axes))
  x.append(float(axes[1]))
  y.append(float(axes[2]))
  z.append(float(axes[3]))
  avg.append(sum(axes[1:]) / 3.)
  mag = np.sqrt(sum(axes[1:] ** 2))
  magnitudes.append(mag)

domain = range(len(x))

plt.plot(domain, x, color='red')
plt.plot(domain, y, color='green')
plt.plot(domain, z, color='blue')
plt.plot(domain, avg, color='black')
plt.plot(domain, magnitudes, color='orange')
plt.show()
