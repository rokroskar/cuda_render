import pynbody as pnb
import os, glob
import numpy as np


list = glob.glob("[1-9]/*.00???")
list.extend(glob.glob("10/*.01000"))

for file in sorted(list, key = lambda file: file[-5:]) :
    s = pnb.load(file)
    print 'processing ' + file
    pnb.analysis.halo.centre(s)
    p = pnb.profile.Profile(s.s,nbins=50,max=15,min=0)

    np.savez(file+".fourier", c=p.fourier['c'], amp=p.fourier['amp'], phi=p.fourier['phi'],
             mass=p.mass, den=p.den, bins = p.r)
