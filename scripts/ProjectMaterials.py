#!/usr/bin/python

import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-o', default=15, help='Maximum band for the SH expansion')
parser.add_argument('-n', default=500, help='Number of elements for the spherical quadrature')
parser.add_argument('dir', metavar='dir', type=str, help='Ddirectory containing MERL database objects')
args = parser.parse_args()
order = str(args.o)
N     = str(args.n)
dir   = args.dir

from glob import glob
from os   import path
files = glob(path.join(dir, '*.binary'))

script = path.split(path.realpath(sys.argv[0]))[0]
build  = path.join(script, '../build/')

bin = ''
if path.isfile(path.join(build, 'Merl2Sh')):
    bin = path.join(build, 'Merl2Sh')
else:
    print '>> Error: could not find \'Merl2Sh\''
    exit(1)

from subprocess import call
for f in files:

    output = path.splitext(f)[0] + '.mats'
    if path.exists(output):
        continue;

    print '>> Process data file ' + f
    args = [path.normpath(bin), '-o', order, '-n', N, f]
    call(args)
