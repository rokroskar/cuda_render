

import pynbody
from pynbody.array import SimArray
from pynbody import filt
import numpy as np
import warnings
import multiprocessing
#try: 
#    from IPython.parallel import Client
#    lv = Client().load_balanced_view()
#except : 
#    warnings.warn("Parallel interface not able to start -- using serial functions", RuntimeWarning)

class KeyboardInterruptError(Exception): pass

def single_file_pos(file, pinds, family, mode) : 
    try: 
        s = pynbody.load(file)
        pynbody.analysis.halo.center(s, mode = mode)
        cen_size = 5
        cen = s.star[filt.Sphere(cen_size)]
        
        if len(cen)<5 :
            # fall-back to DM
            cen = s.dm[filt.Sphere(cen_size)]
        if len(cen)<5 :
            # fall-back to gas
            cen = s.gas[filt.Sphere(cen_size)]
        if len(cen)<5 :
            # very weird snapshot, or mis-centering!
            raise ValueError, "Insufficient particles around center to get velocity"

        vcen = (cen['vel'].transpose()*cen['mass']).sum(axis=1)/cen['mass'].sum()
        vcen.units = cen['vel'].units
        s['vel']-=vcen
        
        if family == 'star' : 
            subs = s.s
        elif family == 'gas' : 
            subs = s.g
        elif family == 'dark' :
            subs = s.d

        if (np.array(pinds).max() < len(subs)) : 
            pos = subs[pinds]['pos']
            vel = subs[pinds]['vel']
        else : 
            pos = np.zeros((len(pinds),3))
            vel = np.zeros((len(pinds),3))
        mass = subs[pinds]['mass']
        phi = subs[pinds]['phi']
        #metals = subs[pinds]['metals']

        good = np.where(pinds < len(subs))[0]
        if len(good) > 0:
            pos[good] = subs[pinds[good]]['pos']
            vel[good] = subs[pinds[good]]['vel']

        return np.array(pos), np.array(vel), np.array(mass), np.array(phi), s.properties['time']

    except KeyboardInterrupt: 
        raise KeyboardInterruptError()


def trace_orbits(filelist, pinds) : 
    pos = SimArray(np.zeros((len(filelist),len(pinds),3)))
    vel = SimArray(np.zeros((len(filelist),len(pinds),3)))
    mass = SimArry(np.zeros((len(filelist),len(pinds),1)))
    phi = SimArry(np.zeros((len(filelist),len(pinds),1)))
    time = SimArray(np.zeros(len(filelist)))
    
    cen_size = 5
    for i,file in enumerate(filelist) : 
        s = pynbody.load(file)
        pynbody.analysis.halo.center(s)
        cen = s.star[filt.Sphere(cen_size)]
        if len(cen)<5 :
            # fall-back to DM
            cen = s.dm[filt.Sphere(cen_size)]
        if len(cen)<5 :
            # fall-back to gas
            cen = s.gas[filt.Sphere(cen_size)]
        if len(cen)<5 :
            # very weird snapshot, or mis-centering!
            raise ValueError, "Insufficient particles around center to get velocity"

        vcen = (cen['vel'].transpose()*cen['mass']).sum(axis=1)/cen['mass'].sum()
        vcen.units = cen['vel'].units
        s['vel']-=vcen

        pos[i], vel[i], mass[i], time[i] = \
            s.s[pinds]['pos'], s.s[pinds]['vel'], s.s[pinds]['mass'], s.s[pinds]['metals'], s.properties['a']

    return pos, vel, mass, time
    


def trace_orbits_wrap(listinds) : 
    return single_file_pos(*listinds)



def trace_orbits_parallel(filelist, pinds, processes = multiprocessing.cpu_count()/4, block=True, test = False, family='star', mode = 'hyb'): 
    
    from multiprocessing import Pool
    import itertools

    pool = Pool(processes=processes)

    pos = np.empty((len(filelist), len(pinds), 3))
    vel = np.empty((len(filelist), len(pinds), 3))
    mass = np.empty((len(filelist), len(pinds)))
    phi = np.empty((len(filelist), len(pinds)))
    time = np.empty(len(filelist))

    if not test: 

        try : 
            res = np.array(pool.map(trace_orbits_wrap, 
                                    itertools.izip(filelist, itertools.repeat(pinds), itertools.repeat(family), itertools.repeat(mode))))
            pool.close()
        except KeyboardInterrupt : 
            pool.terminate()

        finally: 
            pool.join()
    else : 
        res = np.array(map(trace_orbits_wrap, 
                                    itertools.izip(filelist, itertools.repeat(pinds), itertools.repeat(family), itertools.repeat(mode))))

    for i in range(len(res)) : 
        pos[i] = res[i][0]
        vel[i] = res[i][1]
        mass[i] = res[i][2]
        phi[i] = res[i][3]
        time[i] = res[i][4].ratio('Gyr')
        

    return pos, vel, mass, phi, time        
    

def orbit_cwt(x,y,t,pad=True, ax = None, plot_ridges = False) :

    import cwt
    from scipy.special import erf
    import pylab as plt

    if pad:
        xpad = np.append(x,np.zeros(2**np.ceil(np.log(len(t))/np.log(2))-len(t)))
        ypad = np.append(y,np.zeros(2**np.ceil(np.log(len(t))/np.log(2))-len(t)))
    else:
        xpad = x
        ypad = y
        

    wavx = cwt.Morlet(xpad, 1, 300, scaling='log')
    wavy = cwt.Morlet(ypad, 1, 32, scaling='log')
    scalesx = wavx.getscales()*wavx.fourierwl*0.01
    scalesy = wavy.getscales()*wavy.fourierwl*0.01
    freqsx = 2*np.pi/scalesx/2.
    pwrx = wavx.getpower()[:,0:len(t)]
    pwry = np.sqrt(wavy.getpower()[:,0:len(t)])
    

    # normalize the power (from Nener et al. 1999, "infrared physics & technology")
    cn = (4*np.pi**.5/(1+erf(5)))**.5
    for i in range(len(t)):
        pwrx[:,i] *= 2/cn/np.sqrt(wavx.getscales())
        pwry[:,i] *= 2/cn/np.sqrt(wavy.getscales())
        
    # make the figure

    if ax is None : 
        #plt.figure(figsize=(12,6))
        #plt.subplot(2,1,1)
        #plt.contourf(t,scalesx,pwrx,np.linspace(pwrx.min(),pwrx.max(),100))
 #       plt.imshow(np.log10(pwrx),origin='upper',extent=[t[0],t[-1],scalesx[-1],scalesx[0]])
        #plt.xlim(t[0],t[-1])
        #plt.ylim(scalesx[-1],scalesx[0])
        #plt.semilogy()
        
        #plt.subplot(2,1,2)
        plt.contourf(t,scalesx,np.log10(pwrx),100)
        plt.xlim(t[0],t[-1])
        plt.ylim(scalesx[0],scalesx[-1])
        plt.semilogy()

    else : 
        cf = ax.contourf(t,scalesx,np.log10(pwrx),np.linspace(-3,3,50))
        aspect = (t[-1]-t[0])/(scalesx[-1]-scalesx[0])
#        cf = ax.imshow(np.log10(pwrx),origin='lower',extent=[t[0],t[-1],scalesx[-1],scalesx[0]])
        #plt.ylim(scalesx[0],scalesx[-1])
        plt.semilogy()
    
    if plot_ridges : 
        for t0 in range(len(t)):
            p0 = pwrx[:,t0]
            pd = np.gradient(p0)
            maxs = np.where(pd[1:]*pd[:-1] < 0)[0]
            for max0 in maxs:
                if pwrx[max0,t0] > 0.1*pwrx.max() : 
                    plt.plot(t[t0],scalesx[max0],'b.')

    return cf
    

