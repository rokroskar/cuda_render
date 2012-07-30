




import pynbody
import isolated as iso
import numpy as np
import matplotlib.pylab as plt

def zrms_deltar(s) : 
    
    pynbody.analysis.angmom.faceon(s)

    if 'rform' not in s.s.keys():
        iso.get_rform(s.s)
    
    mindr = pynbody.filt.BandPass('dr', 1,10)
    hist, zrms, zrms_i, xs, ys = iso.get_zrms_grid(s.s[mindr])

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

    hist, vdisp_r, vdisp_z, vdisp_z_i, xs, ys = iso.get_vdisp_grid(s.s[mindr],'dr','age')

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
        drfilt = pynbody.filt.LowPass('dr', 10)
        hist, zrms, zrms_i, xs, ys = iso.get_zrms_grid(s.s[rfilt&drfilt])
        
        ax = fig.add_subplot(2,2,i+1)

        plt.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')

        im = plt.imshow(zrms-zrms_i,origin='lower',extent=(min(xs),max(xs),min(ys),max(ys)),
                        aspect='auto',vmin=-.5,vmax=1.5,interpolation='nearest')
        
        cb = plt.colorbar(im)
        cb.set_label('$\Delta z_{rms}$ [kpc]')
        
        plt.xlabel('$\Delta R$ [kpc]')
        plt.ylabel('Age [Gyr]')
        
        plt.title('$%d < R_{form} \\mathrm{ [kpc]} < %d$'%(rlims[0],rlims[1]))

def hz_deltar_rform(s, gridsize=(10,10),vmin=0,vmax=1.2):
    
    fig = plt.figure(figsize=(15,15))
    
    pynbody.analysis.angmom.faceon(s)

    if 'rform' not in s.s.keys():
        iso.get_rform(s.s)
        s.s['dr'] = s.s['rxy'] - s.s['rform']
    
    if 'jzmax' not in s.s.keys():
        iso.get_jzmax(s)
        
    s.s['jz_jzmax'] = s.s['jz']/s.s['jzmax']

    for i,rlims in enumerate([[2,4],[4,6],[6,8],[8,10]]) : 
        
        rfilt = pynbody.filt.BandPass('rform',rlims[0],rlims[1])
        drfilt = pynbody.filt.LowPass('dr', 10)
        jzfilt = pynbody.filt.BandPass('jz_jzmax', .9, 1.01)
        hist, hz, hr, hzerr, hrerr, xs, ys = iso.get_hz_grid(s.s[rfilt&drfilt], 'dr', 'age', 
                                                             rmin=0,rmax=20,zmin=0,zmax=3,gridsize=gridsize)
        

        ax = fig.add_subplot(2,2,i+1)

        plt.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')

        im = plt.imshow(hz,origin='lower',
                        extent=(min(xs), max(xs), min(ys), max(ys)),
                        aspect='auto',vmin=vmin,vmax=vmax,interpolation='nearest')
        
        cb = plt.colorbar(im)
        cb.set_label('$h_z [kpc]$',fontsize='smaller')
        
        plt.xlabel('$\Delta R$ [kpc]',fontsize='smaller')
        plt.ylabel('Age [Gyr]',fontsize='smaller')
        
        plt.title('$%d < R_{form} \\mathrm{ [kpc]} < %d$'%(rlims[0],rlims[1]), fontsize='small')

def hz_feh_ofe(s,gridsize=(10,10)):
    
    fig = plt.figure(figsize=(15,5))
    
    pynbody.analysis.angmom.faceon(s)

    if 'rform' not in s.s.keys():
        iso.get_rform(s.s)
        s.s['dr'] = s.s['rxy'] - s.s['rform']
    
        
    fehfilt = pynbody.filt.BandPass('feh',-1,.5)
    ofefilt = pynbody.filt.BandPass('ofe', -.3,.3)
    sn = pynbody.filt.BandPass('rxy',4,9)
    hist, hz, hr, hzerr, hrerr, xs, ys = iso.get_hz_grid(s.s[fehfilt&ofefilt&sn], 'feh', 'ofe',4,9,0,3,gridsize=gridsize)
    


    ax = fig.add_subplot(121)

    plt.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')
    im = plt.imshow(hz,origin='lower',
                    extent=(min(xs), max(xs), min(ys), max(ys)),
                    aspect='auto',vmin=0,vmax=0.6,interpolation='nearest')
        
    cb = plt.colorbar(im)
    cb.set_label('$h_z [kpc]$',fontsize='smaller')
        
    plt.xlabel('[Fe/H]',fontsize='smaller')
    plt.ylabel('[O/Fe]',fontsize='smaller')
        
    ax = fig.add_subplot(122)
    plt.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')
    im = plt.imshow(hr,origin='lower',
                    extent=(min(xs), max(xs), min(ys), max(ys)),
                    aspect='auto',vmin=1.5,vmax=5,interpolation='nearest')
        
    cb = plt.colorbar(im)
    cb.set_label('$h_r [kpc]$',fontsize='smaller')
        
    plt.xlabel('[Fe/H]',fontsize='smaller')
    plt.ylabel('[O/Fe]',fontsize='smaller')
        


def zrms_deltar_rfinal(s):
    
    fig = plt.figure(figsize=(15,15))
    
    pynbody.analysis.angmom.faceon(s)

    if 'rform' not in s.s.keys():
        iso.get_rform(s.s)
    
    for i,rlims in enumerate([[2,4],[4,6],[6,8],[8,10]]) : 
        
        rfilt = pynbody.filt.BandPass('rxy',rlims[0],rlims[1])
        drfilt = pynbody.filt.LowPass('dr', 10)
        hist, zrms, zrms_i, xs, ys = iso.get_zrms_grid(s.s[rfilt&drfilt])
        
        ax = fig.add_subplot(2,2,i+1)

        plt.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')

        im = plt.imshow(zrms-zrms_i,origin='lower',extent=(min(xs),max(xs),min(ys),max(ys)),
                        aspect='auto',vmin=0,vmax=1.,interpolation='nearest')
        
        cb = plt.colorbar(im)
        cb.set_label('$\Delta z_{rms}$ [kpc]')
        
        plt.xlabel('$\Delta R$ [kpc]')
        plt.ylabel('Age [Gyr]')
        
        plt.title('$%d < R_{final} \\mathrm{ [kpc]} < %d$'%(rlims[0],rlims[1]))

def vdisp_deltar_rform(s,gridsize=(10,10),vmin=0,vmax=70):
    
    fig = plt.figure(figsize=(15,15))
    
    pynbody.analysis.angmom.faceon(s)

    if 'rform' not in s.s.keys():
        iso.get_rform(s.s)
    
    for i,rlims in enumerate([[2,4],[4,6],[6,8],[8,10]]) : 
        
        rfilt = pynbody.filt.BandPass('rform',rlims[0],rlims[1])
        drfilt = pynbody.filt.LowPass('dr', 10)
        hist, vdisp_r, vdisp_z, vdisp_z_i, xs, ys = iso.get_vdisp_grid(s.s[rfilt&drfilt],
                                                                       'dr','age',gridsize=gridsize)
        
        ax = fig.add_subplot(2,2,i+1)

        plt.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')

        im = plt.imshow(vdisp_z,origin='lower',extent=(min(xs),max(xs),min(ys),max(ys)),
                        aspect='auto',interpolation='nearest',vmin=vmin,vmax=vmax)
        
        cb = plt.colorbar(im)
        cb.set_label('$\sigma_z$ [km/s]', fontsize='smaller')
        
        plt.xlabel('$\Delta R$ [kpc]',fontsize='smaller')
        plt.ylabel('Age [Gyr]',fontsize='smaller')
        
        plt.title('$%d < R_{form} \\mathrm{ [kpc]} < %d$'%(rlims[0],rlims[1]),fontsize='small')


def vdisp_deltar_rfinal(s):
    
    fig = plt.figure(figsize=(15,15))
    
    pynbody.analysis.angmom.faceon(s)

    if 'rform' not in s.s.keys():
        iso.get_rform(s.s)
    
    for i,rlims in enumerate([[2,4],[4,6],[6,8],[8,10]]) : 
        
        rfilt = pynbody.filt.BandPass('rxy',rlims[0],rlims[1])
       # mindr = pynbody.filt.BandPass('dr', 1,10)
        hist, vdisp_r, vdisp_z, vdisp_z_i, xs, ys = iso.get_vdisp_grid(s.s[rfilt],'dr','age')
        
        ax = fig.add_subplot(2,2,i+1)

        plt.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')

        im = plt.imshow(vdisp_z-vdisp_z_i,origin='lower',
                        extent=(min(xs)-np.diff(xs)[0],
                                max(xs)+np.diff(xs)[0],
                                min(ys)-np.diff(ys)[0],
                                max(ys)+np.diff(ys)[0]),
                        aspect='auto',interpolation='nearest')
        
        cb = plt.colorbar(im)
        cb.set_label('$\Delta \sigma_z$ [km/s]')
        
        plt.xlabel('$\Delta R$ [kpc]')
        plt.ylabel('Age [Gyr]')
        
        plt.title('$%d < R_{final} \\mathrm{ [kpc]} < %d$'%(rlims[0],rlims[1]))


def age_deltar_slices(s):

    fig = plt.figure(figsize=(15,15))

    pynbody.analysis.faceon(s)

    if 'rform' not in s.s.keys():
        iso.get_rform(s.s)

    for i,rlims in enumerate([[4,6],[6,8]]) : 
        
        rfilt = pynbody.filt.BandPass('rform',rlims[0],rlims[1])
        drfilt = pynbody.filt.LowPass('dr', 10)
        hist, hz, rho0, xs, ys = iso.get_hz_grid(s.s[rfilt&drfilt], 'dr', 'age', gridsize=gridsize)
        
        ax = fig.add_subplot(2,2,i+1)

        plt.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')

        im = plt.imshow(hz,origin='lower',
                        extent=(min(xs), max(xs), min(ys), max(ys)),
                        aspect='auto',vmin=vmin,vmax=vmax,interpolation='nearest')
        
        cb = plt.colorbar(im)
        cb.set_label('$h_z [kpc]$',fontsize='smaller')
        
        plt.xlabel('$\Delta R$ [kpc]',fontsize='smaller')
        plt.ylabel('Age [Gyr]',fontsize='smaller')
        
        plt.title('$%d < R_{form} \\mathrm{ [kpc]} < %d$'%(rlims[0],rlims[1]), fontsize='small')

    

