#!/bin/env python


import pynbody,sys,os
import pynbody.profile as profile
import numpy as np


filename = sys.argv[1]
outfile = filename + '.fourier'

s = pynbody.load(filename)
pynbody.analysis.centre(s)

p = profile.Profile(s.s, nbins=50, max = 15, min = 0)

np.savez(outfile, c=p.fourier['c'], amp=p.fourier['amp'], phi=p.fourier['phi'],
         mass=p.mass, den=p.den, bins = p.r)

if int(sys.argv[2]):
    print ['deleting ' + filename]
    os.remove(filename)
