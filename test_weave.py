import numpy as np
from pynbody.units import takes_arg_in_units

support_code = """
#include <stdio.h>
#include <math.h>
"""

code = file('/home/itp/roskar/homegrown/com.c').read()
    
from scipy import weave
from scipy.weave import converters  
import pynbody

@takes_arg_in_units(("r", "kpc"))
def weave_ssc(sim, r=None, shrink_factor = 0.7, min_particles = 100, verbose=False) :
    """
    
    Return the center according to the shrinking-sphere method of
    Power et al (2003)
    
    """
    x = sim

    if r is None :
        # use rough estimate for a maximum radius
        # results will be insensitive to the exact value chosen
        r = (sim["x"].max()-sim["x"].min())/2
    
    com=np.array([0.0,0.0,0.0],dtype='double')

    if verbose: verbose = 1
    else: verbose = 0

    print r

    with sim.immediate_mode : 
        rs = np.sqrt(np.sum(sim['pos']**2,axis=1))
        ind = np.where(rs < r)[0]
        mass = np.array(sim['mass'][ind],dtype='double')
        pos = np.array(sim['pos'][ind],dtype='double')
        rs = rs[ind]

        npart = len(ind)
        
        vars = ['pos','com','mass','min_particles','npart','r','verbose']

        weave.inline(code,vars,support_code=support_code,compiler='gcc')
            
    return pynbody.array.SimArray(com,sim['pos'].units)
    
def shrink_sphere_center(sim, r=None, shrink_factor = 0.7, min_particles = 100, verbose=False) :
    """
    
    Return the center according to the shrinking-sphere method of
    Power et al (2003)
    
    """
    x = sim

    if r is None :
        # use rough estimate for a maximum radius
        # results will be insensitive to the exact value chosen
        r = (sim["x"].max()-sim["x"].min())/2
    com=np.zeros(3)

    with sim.immediate_mode : 
        rs = np.sqrt(np.sum(sim['pos']**2,axis=1))
        ind = np.where(rs < r)[0]
        mass = sim['mass'][ind]
        pos = sim['pos'][ind]
        rs = rs[ind]
        
        while len(ind)>min_particles or com is None :
#            for ri in rs: print ri,com[0] 
            mtot = mass.sum()
            com = np.sum(mass*pos.transpose(),axis=1)/mtot
            
            if verbose:
                print com,r,len(ind),mtot
            r*=shrink_factor
            rs = np.sqrt(np.sum((pos-com)**2,axis=1))

            ind = np.where(rs < r)[0]
            mass = mass[ind]
            pos = pos[ind]
            rs = rs[ind]
            
    return com
