import numpy as np
import pynbody.array as array
import pynbody.units as units




class KeyboardInterruptError(Exception): pass

def direct(pos, mass, ipos, eps= None) :
    try :
        import scipy, scipy.weave
        from scipy.weave import inline
        import os
    except ImportError :
        pass

    nips = len(ipos)
    m_by_r2 = np.zeros((nips,3))
    m_by_r = np.zeros(nips)

    n = len(mass)
    epssq = np.float(eps*eps)
        
    code = file('/home/itp/roskar/nbody/python/src/pynbody/pynbody/gravity/direct.c').read()
    inline(code,['nips','n','pos','mass','ipos','epssq','m_by_r','m_by_r2'])
        
    return -m_by_r, -m_by_r2

def direct_serial(f, ipos, eps=None, **kwargs) : 
    pot, accel = direct(f['pos'].view(np.ndarray), f['mass'].view(np.ndarray), ipos, eps)
    
    pot = array.SimArray(pot, units = f['mass'].units/f['pos'].units)
    accel = array.SimArray(accel, units = f['mass'].units/f['pos'].units**2)

#    pot *= units.G
#    accel *= units.G

    return pot, accel

def direct_wrap(listargs) : 
    try : 
        return direct(*listargs)
    except KeyboardInterrupt : 
        raise KeyboardInterruptError()

def direct_parallel(f, ipos, eps=None, processes = None) : 
    import multiprocessing
    from multiprocessing import Pool
    from itertools import izip, repeat
    
    if processes is None : processes = multiprocessing.cpu_count()
    
    pool = Pool(processes=processes)

    chunk = np.floor(len(f)/processes)
    
    pos_list = []
    mass_list = []

    for i in range(processes) : 
        if i == processes - 1: 
            sl = slice(i*chunk, len(f))
        else : 
            sl = slice(i*chunk, (i+1)*chunk)

        pos_list.append(f['pos'][sl].view(np.ndarray))
        mass_list.append(f['mass'][sl].view(np.ndarray))
    
    try:
        res = pool.map(direct_wrap, izip(pos_list, mass_list, repeat(ipos), repeat(eps)))
        pool.close()
    except KeyboardInterrupt : 
        pool.terminate()
    finally : 
        pool.join()


        # reduce the result and set the units

        pot = array.SimArray(reduce(lambda x, y: x+y,(res[i][0] for i in range(len(res)))), 
                             units = f['mass'].units/f['pos'].units)
        accel = array.SimArray(reduce(lambda x, y: x+y,(res[i][1] for i in range(len(res)))), 
                               units = f['mass'].units/f['pos'].units**2)

        pot *= units.G
        accel *= units.G

    return pot, accel

def midplane_rot_curve(f, rxy_points, eps = None, mode='tree', **kwargs) :
    import direct

    if eps is None :
        try :
            eps = f['eps']
        except KeyError :
            eps = f.properties['eps']
            
    if isinstance(eps, str) :
        eps = units.Unit(eps)

    # u_out = (units.G * f['mass'].units / f['pos'].units)**(1,2)
    
    # Do four samples like Tipsy does
    rs = [pos for r in rxy_points for pos in [(r,0,0), (0,r,0), (-r,0,0), (0,-r,0)]]

    try:
        fn = {'direct': direct_serial,
              'direct_parallel' : direct_parallel,
              'direct_openmp': direct.direct_omp
              }[mode]
    except KeyError :
        fn = mode

    pot, accel = fn(f,np.array(rs), eps=np.min(eps), **kwargs)

    u_out = (accel.units*f['pos'].units)**(1,2)
    
    # accel = array.SimArray(m_by_r2,units.G * f['mass'].units / (f['pos'].units**2) )

    vels = []

    i=0
    for r in rxy_points:
        r_acc_r = []
        for pos in [(r,0,0), (0,r,0), (-r,0,0), (0,-r,0)]:
            r_acc_r.append(np.dot(-accel[i,:],pos))
            i = i+1
        
        vel2 = np.mean(r_acc_r)
        if vel2>0 :
            vel = np.sqrt(vel2)
        else :
            vel = 0

        vels.append(vel)

    x = array.SimArray(vels, units = u_out)
    x.sim = f.ancestor
    return x


