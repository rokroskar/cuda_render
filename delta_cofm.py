#!/usr/local/Python/2.6.2/bin/python

import pynbody
import pynbody.analysis.halo as halo
import glob
import numpy as np

f = open("cofm.dat", "w")
list = glob.glob('?/*.0???[0,5]')
list.extend(glob.glob('10/*.01000'))

for file in sorted(list, key = lambda name: name[-5:]) :
    s = pynbody.load(file)

    print 'processing ' + file

    cen_xyz = halo.potential_minimum(s)
    cen, cen_vxyz = halo.centre_of_mass(s)

    f.write(np.str(s.properties['a']) + "\t" +
            np.str(cen_xyz[0]) + "\t" +
            np.str(cen_xyz[1]) + "\t" +
            np.str(cen_xyz[2]) + "\t" +
            np.str(cen_vxyz[0]) + "\t" +
            np.str(cen_vxyz[1]) + "\t" +
            np.str(cen_vxyz[2]) + "\n")

    f.flush()

f.close()

    
