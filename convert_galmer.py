import pynbody
import glob
import pyfits
import numpy as np


def to_tipsy(output) : 
    im  = pyfits.getdata(output)
    hdr = pyfits.getheader(output,1)

    # field 12 is particle type (0,1,2 is gas,star,dark)

    g = np.where(im.field(12) == 0)[0]
    s = np.where(im.field(12) == 1)[0]
    d = np.where(im.field(12) == 2)[0]

    newsnap = pynbody.snapshot._new(gas=len(g), dm=len(d), star = len(s))

    for field, name in enumerate(['x','y','z','vx','vy','vz','mass']) : 
        newsnap.g[name] = im.field(field)[g]
        newsnap.d[name] = im.field(field)[d]
        newsnap.s[name] = im.field(field)[s]


    newsnap.properties['a'] = hdr['T_NOW']

    newsnap.write(fmt=pynbody.tipsy.TipsySnap, filename=output+'.std')
