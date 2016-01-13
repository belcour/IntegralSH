#!/usr/bin/python

import sys
directory = sys.argv[1]

from glob import glob
from os   import path
files = glob(path.join(directory, '*.gnuplot'))

import numpy as np
import matplotlib.pyplot as plt
for f in files:
    print '>> Process data file ' + f
    matrix = np.loadtxt(f)

    X = matrix[:, [0]]
    Y = matrix[:, [2]]
    Z = matrix[:, [5]]
    plt.plot(X, Y)
    plt.plot(X, Z)
    outfile = f.replace('gnuplot', 'png')
    print '>> Saving to ' + outfile + '\n'
    plt.savefig(outfile)
    plt.clf()
