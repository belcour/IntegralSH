#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

directory = '.'

from glob import glob
from os   import path
files = glob(path.join(directory, '*.gnuplot'))

for f in files:
    matrix = np.loadtxt(f)

    X = matrix[:, [0]]
    Y = matrix[:, [2]]
    Z = matrix[:, [5]]
    plt.plot(X, Y)
    plt.plot(X, Z)
    plt.savefig(f.replace('gnuplot', 'png'))
    plt.clf()
