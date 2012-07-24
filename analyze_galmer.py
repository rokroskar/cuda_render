import pynbody
import matplotlib.pylab as plt
import numpy as np
from matplotlib.font_manager import FontProperties

def sigma_plots(flist) : 

    flist.sort()

    f = plt.figure(figsize=(8,10))
    
    a1 = f.add_subplot(211)
    a2 = f.add_subplot(212)

    for f in flist: 
        s = pynbody.load(f)
        pynbody.analysis.halo.center(s.s,mode='ssc')
        p = pynbody.analysis.profile.Profile(s.s,max=15,nbins=30)
        
        try : 
            time = int(np.ceil(s.properties['a']))
        except AttributeError:
            time = 0

        a1.plot(p['rbins'],p['vr_disp'], label='%d Myr'%time)

        a2.plot(p['rbins'],p['vz_disp'])
        
    a1.set_xlabel('R [kpc]')
    a1.set_ylabel('$\sigma_R$')
    
    a1.legend(prop=FontProperties(size='small'))

    a2.set_xlabel('R [kpc]')
    a2.set_ylabel('$\sigma_z$')
    
def density_plots(flist) : 

    flist.sort()

    f = plt.figure(figsize=(8,8))
    
    a1 = f.add_subplot(111)
    
    for f in flist: 
        s = pynbody.load(f)
        pynbody.analysis.halo.center(s.s,mode='ssc')
        p = pynbody.analysis.profile.Profile(s.s,max=15,nbins=30)
        
        try : 
            time = int(np.ceil(s.properties['a']))
        except AttributeError:
            time = 0

        a1.plot(p['rbins'],p['density'], label='%d Myr'%time)
        
    a1.set_xlabel('R [kpc]')
    a1.set_ylabel('$\Sigma_{\star}$')
    a1.semilogy()

    a1.legend(prop=FontProperties(size='small'))
    
