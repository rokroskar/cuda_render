




import pynbody
import isolated as iso
import numpy as np
import matplotlib.pylab as plt

def get_zrms_grid(s) : 

    s.s['dr'] = s.s['rxy']-s.s['rform']

    hist, xs, ys = pynbody.plot.generic.hist2d(s.s['dr'],s.s['age'],mass=s.s['mass'],
                                               make_plot=False,gridsize=(20,20))

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    zrms = np.zeros((len(xs),len(ys)))
    zrms_i = np.zeros((len(xs),len(ys)))
    for i,x in enumerate(xs) : 
        for j,y in enumerate(ys) : 
            ind = np.where((s.s['dr'] > x - dx/2) & (s.s['dr'] < x + dx/2) & 
                           (s.s['age'] > y - dy/2) & (s.s['age'] < y + dy/2))[0]
            
            print i,j,len(ind)
            if len(ind) > 100 : 
                
                zrms[j,i] = np.sqrt((s.s['z'][ind]**2).sum()/len(ind))
                zrms_i[j,i] = np.sqrt((s.s['zform'][ind]**2).sum()/len(ind))
            
            else : 
                zrms[j,i] = -500
    return hist, zrms, zrms_i, xs, ys

def get_vdisp_grid(s,varx,vary) : 

    s.s['dr'] = s.s['rxy']-s.s['rform']

    hist, xs, ys = pynbody.plot.generic.hist2d(s.s[varx],s.s[vary],mass=s.s['mass'],
                                               make_plot=False,gridsize=(20,20))

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    vdisp_r = np.zeros((len(xs),len(ys)))
    vdisp_z = np.zeros((len(xs),len(ys)))

    vdisp_r_i = np.zeros((len(xs),len(ys)))
    vdisp_z_i = np.zeros((len(xs),len(ys)))

    for i,x in enumerate(xs) : 
        for j,y in enumerate(ys) : 
            ind = np.where((s.s['dr'] > x - dx/2) & (s.s['dr'] < x + dx/2) & 
                           (s.s['age'] > y - dy/2) & (s.s['age'] < y + dy/2))[0]
            
            print i,j,len(ind)
            if len(ind) > 100 : 
                
                vdisp_r[j,i] = np.std(s.s['vr'][ind])
                vdisp_z[j,i] = np.std(s.s['vz'][ind])
                vdisp_z_i[j,i] = np.std(s.s['vzform'][ind])

    return hist, vdisp_r, vdisp_z, vdisp_z_i, xs, ys

def zrms_deltar(s) : 
    
    pynbody.analysis.angmom.faceon(s)

    if 'rform' not in s.s.keys():
        iso.get_rform(s.s)
    
    mindr = pynbody.filt.BandPass('dr', 1,10)
    hist, zrms, zrms_i, xs, ys = get_zrms_grid(s.s[mindr])

    plt.figure()
    plt.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')
    im = plt.imshow(zrms-zrms_i,origin='lower',extent=(min(xs),max(xs),min(ys),max(ys)),
                    aspect='auto',vmin=0,vmax=1.5,interpolation='nearest')
    cb = plt.colorbar(im)
    cb.set_label('$z_{rms}$')
    plt.xlabel('$\Delta R$')
    plt.ylabel('Age')
    return zrms, hist, xs, ys

def vdisp_deltar(s) : 
    
    pynbody.analysis.angmom.faceon(s)

    if 'rform' not in s.s.keys():
        iso.get_rform(s.s)
    
    mindr = pynbody.filt.BandPass('dr', 1,10)

    hist, vdisp_r, vdisp_z, vdisp_z_i, xs, ys = get_vdisp_grid(s.s[mindr],'dr','age')

    fig = plt.figure(figsize=(10,5))

    ax = fig.add_subplot(1,2,1)
    plt.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')
    im = plt.imshow(vdisp_r,origin='lower',extent=(min(xs),max(xs),min(ys),max(ys)),
                    aspect='auto',interpolation='nearest',vmin=0,vmax=100)

    cb = plt.colorbar(im)
    cb.set_label('$\sigma_r$ [km/s]')
    plt.xlabel('$\Delta R$')
    plt.ylabel('Age')

    ax = fig.add_subplot(1,2,2)
    plt.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')
    im = plt.imshow(vdisp_z-vdisp_z_i,origin='lower',extent=(min(xs),max(xs),min(ys),max(ys)),
                    aspect='auto',interpolation='nearest')

    cb = plt.colorbar(im)
    cb.set_label('$\sigma_z$ [km/s]')
    plt.xlabel('$\Delta R$')
    plt.ylabel('Age')

    return vdisp_r, vdisp_z, hist, xs, ys

def zrms_deltar_rform(s):
    
    fig = plt.figure(figsize=(15,15))
    
    pynbody.analysis.angmom.faceon(s)

    if 'rform' not in s.s.keys():
        iso.get_rform(s.s)
    
    for i,rlims in enumerate([[2,4],[4,6],[6,8],[8,10]]) : 
        
        rfilt = pynbody.filt.BandPass('rform',rlims[0],rlims[1])

        hist, zrms, zrms_i, xs, ys = get_zrms_grid(s.s[rfilt])
        
        ax = fig.add_subplot(2,2,i+1)

        plt.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')

        im = plt.imshow(zrms-zrms_i,origin='lower',extent=(min(xs),max(xs),min(ys),max(ys)),
                        aspect='auto',vmin=-.5,vmax=1.5,interpolation='nearest')
        
        cb = plt.colorbar(im)
        cb.set_label('$\Delta z_{rms}$ [kpc]')
        
        plt.xlabel('$\Delta R$ [kpc]')
        plt.ylabel('Age [Gyr]')
        
        plt.title('$%d < R_{form} \\mathrm{ [kpc]} < %d$'%(rlims[0],rlims[1]))

def zrms_deltar_rfinal(s):
    
    fig = plt.figure(figsize=(15,15))
    
    pynbody.analysis.angmom.faceon(s)

    if 'rform' not in s.s.keys():
        iso.get_rform(s.s)
    
    for i,rlims in enumerate([[2,4],[4,6],[6,8],[8,10]]) : 
        
        rfilt = pynbody.filt.BandPass('rxy',rlims[0],rlims[1])

        hist, zrms, zrms_i, xs, ys = get_zrms_grid(s.s[rfilt])
        
        ax = fig.add_subplot(2,2,i+1)

        plt.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')

        im = plt.imshow(zrms-zrms_i,origin='lower',extent=(min(xs),max(xs),min(ys),max(ys)),
                        aspect='auto',vmin=0,vmax=1.,interpolation='nearest')
        
        cb = plt.colorbar(im)
        cb.set_label('$\Delta z_{rms}$ [kpc]')
        
        plt.xlabel('$\Delta R$ [kpc]')
        plt.ylabel('Age [Gyr]')
        
        plt.title('$%d < R_{final} \\mathrm{ [kpc]} < %d$'%(rlims[0],rlims[1]))

def vdisp_deltar_rform(s):
    
    fig = plt.figure(figsize=(15,15))
    
    pynbody.analysis.angmom.faceon(s)

    if 'rform' not in s.s.keys():
        iso.get_rform(s.s)
    
    for i,rlims in enumerate([[2,4],[4,6],[6,8],[8,10]]) : 
        
        rfilt = pynbody.filt.BandPass('rform',rlims[0],rlims[1])
       # mindr = pynbody.filt.BandPass('dr', 1,10)
        hist, vdisp_r, vdisp_z, xs, ys = get_vdisp_grid(s.s[rfilt])
        
        ax = fig.add_subplot(2,2,i+1)

        plt.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')

        im = plt.imshow(vdisp_z,origin='lower',extent=(min(xs),max(xs),min(ys),max(ys)),
                        aspect='auto',interpolation='nearest',vmin=0,vmax=70)
        
        cb = plt.colorbar(im)
        cb.set_label('$\sigma_z$ [km/s]')
        
        plt.xlabel('$\Delta R$ [kpc]')
        plt.ylabel('Age [Gyr]')
        
        plt.title('$%d < R_{form} \\mathrm{ [kpc]} < %d$'%(rlims[0],rlims[1]))


def vdisp_deltar_rfinal(s):
    
    fig = plt.figure(figsize=(15,15))
    
    pynbody.analysis.angmom.faceon(s)

    if 'rform' not in s.s.keys():
        iso.get_rform(s.s)
    
    for i,rlims in enumerate([[2,4],[4,6],[6,8],[8,10]]) : 
        
        rfilt = pynbody.filt.BandPass('rxy',rlims[0],rlims[1])
       # mindr = pynbody.filt.BandPass('dr', 1,10)
        hist, vdisp_r, vdisp_z, vdisp_z_i, xs, ys = get_vdisp_grid(s.s[rfilt],'dr','age')
        
        ax = fig.add_subplot(2,2,i+1)

        plt.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')

        im = plt.imshow(vdisp_z-vdisp_z_i,origin='lower',extent=(min(xs),max(xs),min(ys),max(ys)),
                        aspect='auto',interpolation='nearest')
        
        cb = plt.colorbar(im)
        cb.set_label('$\Delta \sigma_z$ [km/s]')
        
        plt.xlabel('$\Delta R$ [kpc]')
        plt.ylabel('Age [Gyr]')
        
        plt.title('$%d < R_{final} \\mathrm{ [kpc]} < %d$'%(rlims[0],rlims[1]))
