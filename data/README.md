Before getting started, run the following command:

    pip install -r requirements.txt

All imports:

import argparse
import dill
import json
import magnetak_detectors
import magnetak_evaluate
import magnetak_label
import magnetak_ml
import magnetak_plot
import magnetak_plot # TODO(cjr): Should probably decouple this
import magnetak_util
import ntpath
import numpy as np
import os
import pylab as pl
import random
import scipy
import scipy.interpolate
import scipy.optimize
import scipy.spatial
import scipy.spatial.distance
import sklearn
import sklearn.cross_validation
import sklearn.linear_model
import sklearn.svm
import sys
