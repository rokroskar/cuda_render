import ebf
import pynbody
import os
import numpy as np


def make_galaxia_input(sim, run_enbid=False) :

    # load and align the snapshot
    
    s = pynbody.load(sim)
    pynbody.analysis.angmom.faceon(s)

    filename = s.filename

    # make the pos array

    pos = np.array([s.s['x'],s.s['y'],s.s['z'],s.s['vx'],s.s['vy'],s.s['vz']]).T
    
    # set low metalicities to some reasonable value

    s.s['mets'] = s.s['feh']
    s.s['alpha'] = 0.0
    bad = np.where(s.s['feh'] < -5)[0]
    s.s[bad]['mets'] = -5.0

    # make the enbid file
    
    ebf.write(filename+'_galaxia.ebf', '/pos', pos, 'w')
    ebf.write(filename+'_galaxia.ebf', '/pos3', s.s['pos'], 'a')
    ebf.write(filename+'_galaxia.ebf', '/vel3', s.s['vel'], 'a')
    ebf.write(filename+'_galaxia.ebf', '/Mass', s.s['mass'].in_units('Msol'), 'a')
    ebf.write(filename+'_galaxia.ebf', '/feh', s.s['mets'], 'a')
    ebf.write(filename+'_galaxia.ebf', '/alpha', s.s['alpha'], 'a')
    ebf.write(filename+'_galaxia.ebf', '/age', s.s['age'], 'a')
    ebf.write(filename+'_galaxia.ebf', '/id', s.s['iord'], 'a')
    
    


    if run_enbid:

        # remove 's' to save space

        del(s)

        # run enbid

        os.system('~/bin/enbid -dmc --dim=3 --ngb=64 --dsuffix=_d3n64 %s_galaxia.ebf'%filename)
        os.system('~/bin/enbid -dmc --dim=6 --ngb=64 --dsuffix=_d6n64 %s_galaxia.ebf'%filename)




    
