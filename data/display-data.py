#!/usr/bin/env python
#

import sys
import pylab as pl
import numpy as np

def ShowRotationData():
  data = np.loadtxt(sys.argv[1], delimiter=',')

  magData = data[data[:, -1] == 0]
  gyroData = data[data[:, -1] == 1]

  magDomain = magData[:,0]
  X = magData[:,1]
  Y = magData[:,2]
  Z = magData[:,3]
  mag = np.sqrt(X**2 + Y**2 + Z**2)

  pl.plot(magDomain, X, color='red')
  pl.plot(magDomain, Y, color='blue')
  pl.plot(magDomain, Z, color='green')
  # pl.plot(magDomain, mag, color='black')


  gyroDomain = gyroData[:,0]
  X = gyroData[:,1]
  Y = gyroData[:,2]
  Z = gyroData[:,3]
  mag = np.sqrt(X**2 + Y**2 + Z**2)

  # pl.plot(gyroDomain, X, color='red')
  # pl.plot(gyroDomain, Y, color='blue')
  # pl.plot(gyroDomain, Z, color='green')
  # pl.plot(gyroDomain, mag, color='black')

  pl.show()

def ShowRotatedData():
  data = np.loadtxt(sys.argv[1], delimiter=',')

  magData = data[data[:, -1] == 0]
  rotateData = data[data[:, -1] == 1]

  # lenData = min([len(magData), len(gyroData)])

  rotations = {}
  rotationTimes = []
  for rotateDatum in rotateData:
    t = rotateDatum[0]
    rotationTimes.append(t)
    thisQ = rotateDatum[1:5]
    thisQ[3] = GetWFromQuaternion(thisQ[:3])
    rotations[t] = thisQ

  domain, X, Y, Z = [], [], [], []
  for magDatum in magData:
    magTime = magDatum[0]
    closestTime = max([t for t in rotationTimes if t < magTime])
    thisMag = magDatum[1:4]
    thisQ = rotations[closestTime]

    print "%16d, %16d, , %16d" % (magTime, closestTime, magTime - closestTime)
    print thisQ
    print thisMag
    rotatedVec = RotateByQuaternion(thisMag, thisQ)
    print rotatedVec
    print

    domain.append(magTime)
    X.append(rotatedVec[0])
    Y.append(rotatedVec[1])
    Z.append(rotatedVec[2])

  pl.plot(domain, X, color='red')
  pl.plot(domain, Y, color='blue')
  pl.plot(domain, Z, color='green')
  pl.show()


def ShowData():
  data = np.loadtxt(sys.argv[1], delimiter=',')

  X = data[:,1]
  Y = data[:,2]
  Z = data[:,3]
  mag = np.sqrt(X**2 + Y**2 + Z**2)

  domain = np.arange(len(X))

  pl.plot(domain, X, color='red')
  pl.plot(domain, Y, color='blue')
  pl.plot(domain, Z, color='green')
  pl.plot(domain, mag, color='black')

  pl.show()

def RotateByQuaternion(vect, quaternion):
  """Rotates a vector by a rotation represented as a quaternion.
    The vector should be a represented as a list of length 3, the quaternion as
    a list of length 4, of the format x, y, z, w.
  """

  q_x, q_y, q_z, q_w = quaternion
  x, y, z = vect

  new_x = x*(1- 2*(q_y**2) - 2*(q_z**2)) + y*(2*q_x*q_y - 2*q_w*q_z) + z*(2*q_x*q_z+2*q_w*q_y)
  new_y = x*(2*q_x*q_y + 2*q_w*q_z) + y*(1- 2*(q_x**2) - 2*(q_z**2)) + z*(2*q_y*q_z+2*q_w*q_x)
  new_z = x*(2*q_x*q_z - 2*q_w*q_y) + y*(2*q_y*q_z-2*q_w*q_x) + z*(1- 2*(q_x**2) - 2*(q_y**2))

  return [new_x, new_y, new_z]

def GetWFromQuaternion(quaternion):
  """Reconstruct w from the first three values of a quaternion"""
  return np.sqrt(1 - sum(quaternion**2))

def PlotAccuracyChangedEvents():
  data = np.loadtxt(sys.argv[1], delimiter=',')

  magData = data[data[:, -1] == 0]
  accuracyData = data[data[:, -1] == 2]

  magDomain = magData[:,0]
  X = magData[:,1]
  Y = magData[:,2]
  Z = magData[:,3]
  mag = np.sqrt(X**2 + Y**2 + Z**2)

  pl.plot(magDomain, X, color='red')
  pl.plot(magDomain, Y, color='blue')
  pl.plot(magDomain, Z, color='green')
  pl.plot(magDomain, mag, color='black')

  accuracyColors = ['red','blue','green','black']
  for a in accuracyData:
    t, value, label = a[0], int(a[1]), a[-1]
    pl.axvline(t, color=accuracyColors[value])

  pl.show()


""" The main method """
if __name__ == '__main__':
  if len(sys.argv) != 2:
    print 'Please specify path.'
    sys.exit(1)
  PlotAccuracyChangedEvents()
  # ShowRotationData()
  # ShowRotatedData()
  # ShowData()
