#!/usr/bin/env python
#
# Sketch of the approach:
# - Load in the raw data for x,y,z axes.
# - Look at a window of samples.
# - For window, calculate:
#       Baseline S0 (3-vector) based on the last sample.
# - Split window into 2 segments of fixed time (eg. 300ms).
# - For each of the three segments, calculate:
#       Offset from baseline for each value: O_i = |S - S0|
#       Mean for each segment: U = mean(O_i)
#       Max for each segment: M = max{O_i}
#       Min for each segment: N = min{O_i}
# - Then, see if thresholds match up.
#       MIN1 < T1 && MAX2 > T2

import math
import numpy as np
import sys

BASELINE_TIME = 400
SEGMENT_TIME = 300

T1 = 30
T2 = 130
# T2 = 140

# Store Ui and Mi for all windows.
history = []
detections = []

cjr_history = []
cjr_detections = []

cjr_data = []

def Process():
  global history, cjr_data
  data = np.loadtxt(sys.argv[1], delimiter=',')

  # data = data[::2] # sample every other

  # Find and remove (0,0,0) data points.
  data = data[data[:, 1:].any(1)]

   # = np.gradient([np.linalrg.norm()])
  cjr_data = np.gradient(np.array([np.linalg.norm(row[1:4]) for row in data]))


  # TODO: Calculate window size based on time.
  segment_size = 20
  window_size = segment_size * 2

  # Split the data set into windows.
  for window in np.arange(len(data) - window_size):
    window_end = window + window_size
    # For each window, calculate the baseline.
    #baseline_samples = data[window:window + baseline_size, 1:4]
    #S0 = np.median(baseline_samples, 0)
    #S0 = Mode(baseline_samples)
    # Get the baseline S0, the last value before we start the segmentation.
    S0 = data[window_end, 1:4]
    # S0 = data[window, 1:4]
    # print 'Window %d: S0=%s' % (window, str(S0))

    # Also, split the whole window into segments.
    # TODO: Calculate the segment size in samples based on time.
    segments = np.arange(window, window_end, segment_size)

    # A place for the means and maximums.
    means = []
    maximums = []
    minimums = []

    cjr_maximums = []
    cjr_minimums = []

    for segment in segments:
      # Calculate the offset for each of the samples in the segment.
      samples = data[segment:segment + segment_size, 1:4]
      offsets = samples - S0
      norms = [np.linalg.norm(row) for row in offsets]

      # Calculate the metrics for each segment.
      means.append(np.mean(norms))
      maximums.append(np.max(norms))
      minimums.append(np.min(norms))

      cjr_maximums.append(np.max(cjr_data[segment:segment + segment_size]))
      cjr_minimums.append(np.min(cjr_data[segment:segment + segment_size]))

    # Apply the thresholds to the computed statistics.
    min_1 = minimums[0]
    max_2 = maximums[1]

    # Store I_1, M_2 and I_3 for each window.
    #history.append([window_end, U1, M2, U3, np.linalg.norm(S0)])
    history.append([window_end, min_1, max_2, np.linalg.norm(S0), np.linalg.norm(S0)])

    #print 'Window %d: I_1=%f, M_2=%f, I_3=%f' % (window, U1, M2, U3)

    #if U1 < MAX_U1 and U3 < MAX_U3 and M2 > MIN_M2:
    if min_1 < T1 and max_2 > T2:
      print 'Detection at sample %d' % window_end
      detections.append(window_end)


    cjr_max_1 = cjr_maximums[0]
    cjr_min_2 = cjr_minimums[1]
    if cjr_max_1 > 10 and cjr_min_2 < -10:
      print 'Derivative-based detection at sample %d' % window_end
      cjr_detections.append(window_end)


  history = np.array(history)

  # Write the TSUMU history to a file.
  np.savetxt('/tmp/imi.txt', history, delimiter=',')



# Plot the integral history and maximum history too.
def PlotTSUMU():
  import matplotlib.pyplot as plt
  domain = history[:, 0]

  plot_deriv = True

  if plot_deriv:
    # plot derivative of norm...
    deriv = np.gradient(history[:,3])
    deriv2 = np.gradient(deriv)

    # plt.plot(domain, history[:, 3], color='blue') # Norm of S0
    plt.plot(cjr_data, color='black')
    # plt.plot(domain, deriv2, color='red')
    plt.plot(cjr_detections, np.ones(len(cjr_detections)), marker='o', color='r', ls='')


    # from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
    # rbf = Rbf(domain, deriv)


    # plt.show()
    # return

  plt.plot(domain, history[:, 1], color='red') # Minimum of first window
  plt.plot(domain, history[:, 2], color='green') # Maximum of second window
  plt.plot(domain, history[:, 3], color='blue') # Norm of S0

  # Plot the detections.
  plt.plot(detections, np.ones(len(detections)) * 500, marker='o', color='r', ls='')

  # Plot the thresholds.
  plt.plot(domain, np.ones(len(domain)) * T1, color='#aadddd') # Minimum must be lower
  plt.plot(domain, np.ones(len(domain)) * T2, color='#ddaadd') # Maximum must be higher
  plt.show()

def Mode(vec_array):
  modes = []
  for col in vec_array.transpose():
    u, indices = np.unique(col.round(1), return_inverse=True)
    col_mode = u[np.argmax(np.bincount(indices))]
    modes.append(col_mode)
  return np.array(modes)


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print 'Please specify path.'
    sys.exit(1)

  Process()
  PlotTSUMU()
