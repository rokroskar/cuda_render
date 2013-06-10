




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

def zrms_vdisp_deltar_rform(s,gridsize=(20,20)):
        
    pynbody.analysis.angmom.faceon(s)


    if 'rform' not in s.s.keys():
        iso.get_rform(s.s)


    if 'dr' not in s.s.keys():
        s.s['dr'] = s.s['rxy']-s.s['rform']

    fig, axs = plt.subplots(2,2,figsize=(13,15))

    for i,rlims in enumerate([[2,4],[4,6]]) : 
        
        rfilt = pynbody.filt.BandPass('rform',rlims[0],rlims[1])
        drfilt = pynbody.filt.LowPass('dr', 10)
        hist, zrms, zrms_i, xs, ys = iso.get_zrms_grid(s.s[rfilt&drfilt], 'dr', 'age', 
                                                       rmin = 0, rmax = 20, zmin = 0, zmax= 3,
                                                       gridsize=gridsize)

        hist2, vdisp_r, vdisp_z, vdisp_z_i, xs2, ys2 = iso.get_vdisp_grid(s.s[rfilt&drfilt],
                                                                          'dr','age',gridsize=gridsize)
                
        ax = axs[0,i]

        #ax.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red',linewidth=1.5)

        im = ax.imshow(zrms-zrms_i,origin='lower',extent=(min(xs),max(xs),min(ys),max(ys)),
                        aspect='auto',vmin=-.2,vmax=1.,interpolation='nearest')
                      
        ax.set_xlabel(r'$\Delta R$ [kpc]',fontweight='bold', fontsize='small')
        ax.set_ylabel('Age [Gyr]', fontweight='bold', fontsize='small')
        
        ax.set_title('$%d < R_{form} \\mathrm{ [kpc]} < %d$'%(rlims[0],rlims[1]), fontsize='small', fontweight='bold')

        ax = axs[1,i]

        #ax.contour(xs,ys,np.log10(hist2),np.linspace(1,4,10),colors='red',linewidth=1.5)

        im2 = ax.imshow(vdisp_z,origin='lower',extent=(min(xs2),max(xs2),min(ys2),max(ys2)),
                        aspect='auto',vmin=0,vmax=70,interpolation='nearest')
                      
        ax.set_xlabel(r'$\Delta R$ [kpc]',fontweight='bold', fontsize='small')
        ax.set_ylabel('Age [Gyr]', fontweight='bold', fontsize='small')
        
        ax.set_title('$%d < R_{form} \\mathrm{ [kpc]} < %d$'%(rlims[0],rlims[1]), fontsize='small', fontweight='bold')

    bb1 = axs[0,1].get_position()
    bb2 = axs[1,1].get_position()
    cbax = fig.add_axes([bb1.x1+.01,bb1.y0,0.02,bb1.y1-bb1.y0])
    cb1 = fig.colorbar(im,cax=cbax)
    cb1.set_label('$\Delta z_{rms} \mathrm{~[kpc]}$',fontsize='smaller', fontweight='bold')

    cbax2 = fig.add_axes([bb2.x1+.01,bb2.y0,0.02,bb2.y1-bb2.y0])
    cb2 = fig.colorbar(im2,cax=cbax2)
    cb2.set_label('$\sigma_z$ [km/s]',fontsize='smaller', fontweight='bold')

    for tick in cb1.ax.get_yticklabels():
        tick.set_fontsize('smaller')

    for tick in cb2.ax.get_yticklabels():
        tick.set_fontsize('smaller')
        

def zrms_deltar_rform(s,gridsize=(20,20)):
    
    fig = plt.figure(figsize=(13,15))
    
    pynbody.analysis.angmom.faceon(s)


    if 'rform' not in s.s.keys():
        iso.get_rform(s.s)


    if 'dr' not in s.s.keys():
        s.s['dr'] = s.s['rxy']-s.s['rform']

    for i,rlims in enumerate([[2,4],[4,6],[6,8],[8,10]]) : 
        
        rfilt = pynbody.filt.BandPass('rform',rlims[0],rlims[1])
        drfilt = pynbody.filt.LowPass('dr', 10)
        hist, zrms, zrms_i, xs, ys = iso.get_zrms_grid(s.s[rfilt&drfilt], 'dr', 'age', 
                                                       rmin = 0, rmax = 20, zmin = 0, zmax= 3,
                                                       gridsize=gridsize)
        
        ax = fig.add_subplot(2,2,i+1)

        plt.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red',linewidth=1.5)

        im = plt.imshow(zrms-zrms_i,origin='lower',extent=(min(xs),max(xs),min(ys),max(ys)),
                        aspect='auto',vmin=-.2,vmax=1.,interpolation='nearest', cmap=plt.cm.Greys)
                      
        plt.xlabel(r'$\Delta R$ [kpc]',fontweight='bold', fontsize='small')
        plt.ylabel('Age [Gyr]', fontweight='bold', fontsize='small')
        
        ax.set_title('$%d < R_{form} \\mathrm{ [kpc]} < %d$'%(rlims[0],rlims[1]), fontsize='small', fontweight='bold')

    cbax = fig.add_axes([0.91,0.1,0.02,0.8])
    cb1 = fig.colorbar(im,cax=cbax)
    cb1.set_label('$\Delta z_{rms} \mathrm{~[kpc]}$',fontsize='smaller', fontweight='bold')
    for tick in cb1.ax.get_yticklabels() :
        tick.set_fontsize('smaller')
        
    

def hz_deltar_rform(s, gridsize=(10,10),vmin=0,vmax=1.2,ncpu=pynbody.config['number_of_threads'], form = 'sech', get_errors = False):
    
    fig = plt.figure(figsize=(13,15))
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
        hist, hz, hr, hz2, hr2, hzerr, hrerr, xs, ys, fitnum = \
            iso.get_hz_grid_parallel(s.s[rfilt&drfilt], 'dr', 'age', 
                                     rmin=0,rmax=20,zmin=0,zmax=3,
                                     gridsize=gridsize,ncpu=ncpu,get_errors=get_errors,form = form)
        

        ax = fig.add_subplot(2,2,i+1)
        ax.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')
        im = ax.imshow(hz,origin='lower',
                       extent=(min(xs), max(xs), min(ys), max(ys)),
                       aspect='auto',vmin=vmin,vmax=vmax,interpolation='nearest')
        ax.set_xlabel('$\Delta R$ [kpc]',fontsize='smaller')
        ax.set_ylabel('Age [Gyr]',fontsize='smaller')
        ax.set_title('$%d < R_{form} \\mathrm{ [kpc]} < %d$'%(rlims[0],rlims[1]), fontsize='small')


    cbax = fig.add_axes([0.91,0.1,0.02,0.8])
    cb1 = fig.colorbar(im,cax=cbax)
    cb1.set_label('$h_z \mathrm{~[kpc]}$')
    for tick in cb1.ax.get_yticklabels() :
        tick.set_fontsize('smaller')

        if get_errors:
        # plotting the errors
            ax2 = fig2.add_subplot(2,2,i+1)
            ax2.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')
            im = ax2.imshow(hzerr/hz,origin='lower',
                            extent=(min(xs), max(xs), min(ys), max(ys)),
                            aspect='auto',interpolation='nearest',vmin=0,vmax=.1)
            ax2.set_xlabel('$\Delta R$ [kpc]',fontsize='smaller')
            ax2.set_ylabel('Age [Gyr]',fontsize='smaller')
            ax2.set_title('$%d < R_{form} \\mathrm{ [kpc]} < %d$'%(rlims[0],rlims[1]), fontsize='small')
        
            cbax = plt.axes([0.89,0.17,0.02,0.7])
            cb1 = fig2.colorbar(im,cax=cbax)
            cb1.set_label('h_z [kpc]')


def hz_deltaj_rform(s, gridsize=(10,10),vmin=0,vmax=1.2,ncpu=pynbody.config['number_of_threads'], get_errors = False):
    
    fig = plt.figure(figsize=(15,15))

    pynbody.analysis.angmom.faceon(s)

    if 'rform' not in s.s.keys():
        iso.get_rform(s.s)
        s.s['dr'] = s.s['rxy'] - s.s['rform']
        s.s['delta_j'] = s.s['jz'] - (s.s['xform']*s.s['vyform']-s.s['yform']*s.s['vxform'])

    if 'jzmax' not in s.s.keys():
        iso.get_jzmax(s)
        
    s.s['jz/jc'] = s.s['jz']/s.s['jzmax']

#    for i,rlims in enumerate([[2,4],[4,6],[6,8],[8,10]]) : 
    for i,rlims in enumerate([[500,600],[1000,1100],[1500,1600],[2000,2100]]) : 
        rfilt = pynbody.filt.BandPass('jzform',rlims[0],rlims[1])
        drfilt = pynbody.filt.LowPass('dr', 10)
        jzfilt = pynbody.filt.BandPass('jz/jc', .95, 1.01)
#        hist, hz, hr, hz2, hr2, hzerr, hrerr, xs, ys, fitnum = \
#            iso.get_hz_grid_parallel(s.s[rfilt], 'delta_j', 'age', 
 #                                    rmin=0,rmax=20,zmin=0,zmax=3,
  # gridsize=gridsize,ncpu=ncpu,form='sech')
        
        hist, zrms, zrms_i, xs, ys = iso.get_zrms_grid(s.s[rfilt&drfilt], 'delta_j', 'age', 
                                                       rmin = 0, rmax = 20, zmin = 0, zmax= 3,
                                                       gridsize=gridsize)
        

        ax = fig.add_subplot(2,2,i+1)
        ax.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')
        im = ax.imshow(zrms-zrms_i,origin='lower',
                       extent=(min(xs), max(xs), min(ys), max(ys)),
                       aspect='auto',vmin=vmin,vmax=vmax,interpolation='nearest')
        
        ax.set_xlabel('$\Delta J_z$ [kpc]',fontsize='smaller')
        ax.set_ylabel('Age [Gyr]',fontsize='smaller')
        ax.set_title('$%d < J_{z,form} < %d$'%(rlims[0],rlims[1]), fontsize='small')
        cbax = fig.add_axes([0.91,0.17,0.02,0.7])
        cb1 = fig.colorbar(im,cax=cbax)
        cb1.set_label('$\Delta z_{rms} \mathrm{~[kpc]}$')


        if get_errors:
        # plotting the errors
            ax2 = fig2.add_subplot(2,2,i+1)
            ax2.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')
            im = ax2.imshow(hzerr/hz,origin='lower',
                            extent=(min(xs), max(xs), min(ys), max(ys)),
                            aspect='auto',interpolation='nearest',vmin=0,vmax=.1)
            cb = fig2.colorbar(im)
            cb.set_label('$\sigma h_z [kpc]$',fontsize='smaller')
            ax2.set_xlabel('$\Delta J_z$ [kpc]',fontsize='smaller')
            ax2.set_ylabel('Age [Gyr]',fontsize='smaller')
            ax2.set_title('$%d < R_{form} \\mathrm{ [kpc]} < %d$'%(rlims[0],rlims[1]), fontsize='small')


def hz_deltaj_jjmax(s, gridsize=(10,10),vmin=0,vmax=None,ncpu=pynbody.config['number_of_threads']):
    

    fig, axes = plt.subplots(3,3,figsize=(15,15))

    pynbody.analysis.angmom.faceon(s)

    if 'rform' not in s.s.keys():
        iso.get_rform(s.s)
        s.s['dr'] = s.s['rxy'] - s.s['rform']
        s.s['jform'] = s.s['xform']*s.s['vyform']-s.s['yform']*s.s['vxform']
        s.s['delta_j'] = s.s['jz'] - s.s['jform']
        
    if 'jzmax' not in s.s.keys():
        iso.get_jzmax(s)
        
    s.s['jz_jzmax'] = s.s['jz']/s.s['jzmax']

    

    for i,jlims in enumerate([[500,600],[800,900],[1100,1200],[1300,1400],[1400,1500],[1600,1700],[1700,1800],[1900,2000],[2100,2200]]):
        jfilt = pynbody.filt.BandPass('jz',jlims[0],jlims[1])
        jzmaxfilt = pynbody.filt.BandPass('jz_jzmax', 0 , 1.01)
        djjformfilt = pynbody.filt.BandPass('jjform',0,2)
        agefilt = pynbody.filt.BandPass('age', 7, 9)
        hist, hz, hr, hz2, hr2, hzerr, hrerr, xs, ys, fitnum = \
            iso.get_hz_grid_parallel(s.s[jfilt &
                                         jzmaxfilt &
                                         agefilt], 'delta_j', 'jz_jzmax', 
                                     rmin=0,rmax=20,zmin=0,zmax=3,
                                     gridsize=gridsize,ncpu=ncpu)
        

        ax = axes.flatten()[i]
        ax.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')
        im = ax.imshow(hz,origin='lower',
                       extent=(min(xs), max(xs), min(ys), max(ys)),
                       aspect='auto',vmin=vmin,vmax=vmax,interpolation='nearest')
        #cb = fig.colorbar(im)
        #cb.set_label('$h_z [kpc]$',fontsize='smaller')
        ax.set_xlabel('$\Delta J_z$ [kpc]',fontsize='smaller')
        ax.set_ylabel('$J_z/J_c$',fontsize='smaller')
        ax.set_title('$%d < J_z \\mathrm{ [kpc]} < %d$'%(jlims[0],jlims[1]), fontsize='small')

        cbax = fig.add_axes([0.92,0.17,0.02,0.7])
        cb1 = fig.colorbar(im,cax=cbax)
        cb1.set_label('h_z [kpc]')

def hz_feh_ofe(s,gridsize=(20,20),rmin=4,rmax=9,ncpu=pynbody.config['number_of_threads']):
    
    fig = plt.figure(figsize=(15,5))
    
    pynbody.analysis.angmom.faceon(s)

    if 'rform' not in s.s.keys():
        iso.get_rform(s.s)
        s.s['dr'] = s.s['rxy'] - s.s['rform']
    
        
    fehfilt = pynbody.filt.BandPass('feh',-1,.5)
    ofefilt = pynbody.filt.BandPass('ofe', -.3,.3)
    sn = pynbody.filt.BandPass('rxy',rmin,rmax)
    hist, hz, hr, hz2,hr2,hzerr, hrerr, xs, ys,fitnum = iso.get_hz_grid_parallel(s.s[fehfilt&ofefilt&sn], 'feh', 'ofe',rmin,rmax,0,3,gridsize=gridsize,ncpu=ncpu)
    


    ax = fig.add_subplot(122)

    plt.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')
    im = plt.imshow(hz,origin='lower',
                    extent=(min(xs), max(xs), min(ys), max(ys)),
                    aspect='auto',vmin=0.1,vmax=0.5,interpolation='nearest')
        
    cb = plt.colorbar(im)
    cb.set_label('$h_z [kpc]$',fontsize='smaller')
        
    plt.xlabel('[Fe/H]',fontsize='smaller')
    plt.ylabel('[O/Fe]',fontsize='smaller')
        
    ax = fig.add_subplot(121)
    plt.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')
    im = plt.imshow(hr,origin='lower',
                    extent=(min(xs), max(xs), min(ys), max(ys)),
                    aspect='auto',vmin=1.5,vmax=4.5,interpolation='nearest')
        
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

def vdisp_deltar_rform(s,gridsize=(20,20),vmin=0,vmax=70):
    
    fig = plt.figure(figsize=(13,15))
    
    pynbody.analysis.angmom.faceon(s)

    if 'rform' not in s.s.keys():
        iso.get_rform(s.s)
    
    if 'dr' not in s.s.keys():
        s.s['dr'] = s.s['rxy']-s.s['rform']

    for i,rlims in enumerate([[2,4],[4,6],[6,8],[8,10]]) : 
        
        rfilt = pynbody.filt.BandPass('rform',rlims[0],rlims[1])
        drfilt = pynbody.filt.LowPass('dr', 10)
        hist, vdisp_r, vdisp_z, vdisp_z_i, xs, ys = iso.get_vdisp_grid(s.s[rfilt&drfilt],
                                                                       'dr','age',gridsize=gridsize)
        
        ax = fig.add_subplot(2,2,i+1)

        plt.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red',linewidth=1.5)

        im = plt.imshow(vdisp_z,origin='lower',extent=(min(xs),max(xs),min(ys),max(ys)),
                        aspect='auto',interpolation='nearest',vmin=vmin,vmax=vmax)
        
        plt.xlabel('$\Delta R$ [kpc]',fontsize='small', fontweight='bold')
        plt.ylabel('Age [Gyr]',fontsize='small', fontweight='bold')
        
        plt.title('$%d < R_{form} \\mathrm{ [kpc]} < %d$'%(rlims[0],rlims[1]),fontsize='small')

    cbax = fig.add_axes([0.91,0.1,0.02,0.8])
    cb1 = fig.colorbar(im,cax=cbax)
    cb1.set_label('$\sigma_z \mathrm{~[km/s]}$',fontsize='smaller')
    for tick in cb1.ax.get_yticklabels() :
        tick.set_fontsize('smaller')


    
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

    

def hz_jjmax(s,jmin,jmax,jcmin=.5,jcmax=1.0,nbins=10) :
    import diskfitting

    if 'jz/jc' not in s.s.keys(): 
        iso.get_jzmax(s)

    bins = np.linspace(jcmin,jcmax,nbins+1)

    hrs = np.zeros(nbins)
    hzs = np.zeros(nbins)

    for i in range(len(bins)-1):
        ind = np.where((s.s['jz']>jmin)&(s.s['jz']<jmax)&
                       (s.s['jz_jzmax']>bins[i])&
                       (s.s['jz_jzmax']<bins[i+1])&
                       (s.s['delta_j']>500))[0]

        print bins[i],bins[i+1],len(ind)

        if len(ind) > 100:
            hr,hz,fitnum=diskfitting.two_exp_fit_simple(np.array(s.s['rxy'][ind]),
                                                        np.array(s.s['z'][ind]),
                                                        0,20,0,4)

            hrs[i] = hr
            hzs[i] = hz
            
        else:
            hrs[i]=float('Nan')
            hzs[i]=float('Nan')
    
    plt.plot(.5*(bins[:-1]+bins[1:]),hzs)


def hz_deltaj_jc(s, gridsize=(10,10),vmin=0,vmax=1.2,ncpu=pynbody.config['number_of_threads']):
    
    fig = plt.figure(figsize=(15,15))
    fig2 = plt.figure(figsize=(15,15))

    pynbody.analysis.angmom.faceon(s)

    if 'delta_j' not in s.s.keys():
        iso.get_rform(s.s)
        s.s['delta_j'] = s.s['jz'] - (s.s['vyform']*s.s['x']-s.s['vxform']*s.s['y'])
    
    if 'jzmax' not in s.s.keys():
        iso.get_jzmax(s)
        
    s.s['jz_jzmax'] = s.s['jz']/s.s['jzmax']

    for i,rlims in enumerate([[500,550],[1000,1050],[1200,1250],[1500,1550]]) : 
        
        jfilt = pynbody.filt.BandPass('jz',rlims[0],rlims[1])
#333        djfilt = pynbody.filt.LowPass('delta_j', 10)
        jzfilt = pynbody.filt.BandPass('jz_jzmax', .9, 1.01)
        hist, hz, hr, hz2, hr2, hzerr, hrerr, xs, ys, fitnum = \
            iso.get_hz_grid_parallel(s.s[jfilt], 'delta_j', 'jz_jzmax', 
                                     rmin=0,rmax=20,zmin=0,zmax=3,
                                     gridsize=gridsize,ncpu=ncpu)
        

        ax = fig.add_subplot(2,2,i+1)
        ax.contour(xs,ys,np.log10(hist),np.linspace(1,4,10),colors='red')
        im = ax.imshow(hz,origin='lower',
                       extent=(min(xs), max(xs), min(ys), max(ys)),
                       aspect='auto',vmin=vmin,vmax=vmax,interpolation='nearest')
        cb = fig.colorbar(im)
        cb.set_label('$h_z [kpc]$',fontsize='smaller')
        ax.set_xlabel('$\Delta J_z$ [kpc]',fontsize='smaller')
        ax.set_ylabel('$J/J_z$',fontsize='smaller')
        ax.set_title('$%d < J_z < %d$'%(rlims[0],rlims[1]), fontsize='small')

def pos_expo(x,p) : 
    return p[0]*np.exp(x/p[1])

def make_zrms_jfinal_fig(s,rgmin,rgmax,agemin,agemax) : 
    from pynbody.filt import BandPass
    from pynbody.analysis.profile import Profile
    from fitting import expo
    from scipy import optimize
    

    # set up filters
    
    rgfilt = BandPass('Rg',rgmin,rgmax)
    agefilt = BandPass('age',agemin,agemax)

    f, ax = plt.subplots(1,3,figsize=(24,6))

    fitfunc = lambda p, x : p[0]*np.exp(x/p[1])
    errfunc = lambda p, x, y, err : (y-fitfunc(p,x))/err

    colors = ['b','g','r']

    s.s['Rg'] = s.s['jz']/250.0

    for j, jcrange in enumerate([[.85,.9],[.9,.95],[.95,1.0]]) : 

        jcfilt = BandPass('jz/jc', jcrange[0],jcrange[1])
        
        filt = agefilt & jcfilt & rgfilt

        if filt.where(s.s)[0].size > 0: 

            p = Profile(s.s[filt], calc_x=lambda x: x['delta_j']/250., min = (s.s[filt]['delta_j']/250.).min(), nbins=10)

            good = np.where(p['n']>100)

            ax[0].plot(p['rbins'][good],(p['z_rms']/p['zform_rms'])[good],
                       label="$%.2f < j_z/j_c < %.2f$"%(jcrange[0],jcrange[1]),color=colors[j])
            ax[0].set_xlabel('$\Delta R_g~[\mathrm{kpc}]$')
            ax[0].set_ylabel('$z_{rms}/z_{i,rms}$')
            
            ax[1].plot(p['rbins'][good],p['zform_rms'][good])
            ax[1].set_xlabel('$\Delta R_g~[\mathrm{kpc}]$')
            ax[1].set_ylabel('$z_{i,rms}~[\mathrm{kpc}]$')

            ax[2].plot(p['rbins'][good],p['z_rms'][good])
            ax[2].set_xlabel('$\Delta R_g~[\mathrm{kpc}]$')
            ax[2].set_ylabel('$z_{rms}~[\mathrm{kpc}]$')



                # fit
               # p1, res = optimize.leastsq(errfunc,[1.0,1000], args=(np.array(p['rbins']),
               #                                                      np.array(p['z_rms']/p['zform_rms']),
               #                                                      np.array(p['z_rms']/p['zform_rms']/np.sqrt(p['n']))))
               # print p1, res
               # x = np.linspace(p['rbins'].min(),p['rbins'].max(),100)
               # ax[i].plot(x,fitfunc(p1,x),'%s--'%colors[j])

                
                
                
    ax[1].set_title('$%.1f < R_{g,now} < %.1f$'%(rgmin,rgmax),size='large')
    ax[0].legend(loc=0,prop=dict(size='small'))


def make_zrms_jform_fig(s,agemin,agemax) : 
    from pynbody.filt import BandPass
    from pynbody.analysis.profile import Profile
    from fitting import expo
    from scipy import optimize, interp
    

    # set up filters
    

    agefilt = BandPass('age',agemin,agemax)

    f,axs = plt.subplots(1,3,figsize=(24,6))

    fitfunc = lambda p, x : p[0]*np.exp(x/p[1])
    errfunc = lambda p, x, y, err : (y-fitfunc(p,x))/err

    colors = ['b','g','r']
    
    for i, (rmin, rmax) in enumerate([[4,4.5],[5,5.5],[6,6.5]]) : 
        rformfilt = BandPass('rform',rmin,rmax)
        ax = axs[i]
        for j, jcrange in enumerate([[.85,.9],[.9,.95],[.95,1.0]]) : 

            jcfilt = BandPass('jz/jc', jcrange[0],jcrange[1])
        
            filt = agefilt & jcfilt & rformfilt

            if filt.where(s.s)[0].size > 0: 
                p = Profile(s.s[filt], calc_x=lambda x: x['delta_j']/250., min = (s.s[filt]['delta_j']/250.).min(), nbins=10)

                good = np.where(p['n'] > 100)[0]
                ax.plot(p['rbins'][good],(p['z_rms']/p['zform_rms'])[good],
                         label="$%.2f < j_z/j_c < %.2f$"%(jcrange[0],jcrange[1]),color=colors[j])
            
                ax.set_xlabel('$\Delta R_g$ [kpc]')
                ax.set_ylabel('$z_{rms}/z_{i,rms}$')

    # fit
                    
        x = np.linspace(p['rbins'][good].min(),p['rbins'][good].max(),100)
        p1, res = optimize.leastsq(errfunc,[1.0,1.0], args=(np.array(p['rbins']),
                                                            np.array(p['z_rms']/p['zform_rms']),
                                                            np.array(p['z_rms']/p['zform_rms']/np.sqrt(p['n']))))
                
        ax.plot(x,fitfunc(p1,x),'r--')
        colors2 = ['orange','blue','green']

        for k, alpha in enumerate([0.5,1.0,2.0]):
            y = interp(0,p['delta_j']/250.,p['z_rms']/p['zform_rms'])*np.exp(x/(2.5*(2+alpha)))
            ax.plot(x,y,color=colors2[k],linestyle='--')
            ax.annotate(r'%.1f'%alpha, (x[-1]+.01, y[-1]),fontsize=12,color=colors2[k])
        
            ax.set_title('$%.1f < R_{form} \mathrm{~[kpc]}< %.1f$'%(rmin,rmax),fontsize='large')
        
        ax.set_xlim(-3.5,5.5)
        if i ==0 : ax.legend(loc=0,prop=dict(size='small'))



def make_zrms_vs_r_plot(rs = [2,4,6,8,10,12,14], agefilt = pynbody.filt.BandPass('age',7,9)) : 
    
    from parallel_util import run_parallel

    flist = ['12M_hr_diff_coeff0.05/10/12M_hr_diff_coeff0.05.01000', 
             '12M_hr_25pc_soft/10/12M_hr_25pc_soft.01000.gz', 
             '12M_hr_100pc_soft/10/12M_hr_100pc_soft.01000.gz', 
             '12M_hr_x0.5N/10/12M_hr_x0.5N.01000.gz', 
             '12M_hr_x2N/10/12M_hr_x2N.01000.gz', 
             '12M_hr_x4N/10/12M_hr_x4N.01000']

    names = ['fiducial', 
             '$h_s = 25\mathrm{~pc}$', 
             '$h_s = 100\mathrm{~pc}$', 
             '$0.5 N$', 
             '$2 N$', 
             '$4 N$']

    slist = []

    for f in flist : 
        slist.append(pynbody.load(f))

#    slist = run_parallel(pynbody.load, flist, [], processes = 6)
    
    fig, axs = plt.subplots(1,2)

    iso.compare_zrms_vs_r(slist[:3], names[:3], rs, agefilt, axs[0])
    iso.compare_zrms_vs_r([slist[0],slist[3],slist[4],slist[5]], 
                          [names[0],names[3],names[4],names[5]], 
                          rs, agefilt, axs[1])

    
    
    for ax in axs : 
        ax.set_xlabel('R [kpc]')
        ax.set_ylabel('$z_{rms}$')
        ax.set_ylim(0,2)
        ax.set_xlim(0,15)
        ax.legend(loc='upper left')


def make_zprofiles(rs = [4.5,8.5,12.5]) : 
    from parallel_util import run_parallel
    from pynbody.analysis.profile import VerticalProfile

    flist = ['12M_hr_diff_coeff0.05/10/12M_hr_diff_coeff0.05.01000', 
             '12M_hr_x0.5N/10/12M_hr_x0.5N.01000.gz', 
             '12M_hr_x2N/10/12M_hr_x2N.01000.gz', 
             '12M_hr_x4N/10/12M_hr_x4N.01000']

    names = ['fiducial', 
             '$0.5 N$', 
             '$2 N$', 
             '$4 N$']

    fig, axs = plt.subplots(1,3,figsize=(20,6))

    slist = []

    for f in flist : 
        slist.append(pynbody.load(f))
        
    for i,s in enumerate(slist): 
        pynbody.analysis.angmom.faceon(s)
        for j, r in enumerate(rs): 
            p = VerticalProfile(s.s,r-.5,r+.5,3.0,nbins=50)
            axs[j].plot(p['rbins'],p['density']/p['density'][17], label = names[i])
              
    for i,ax in enumerate(axs): 
        ax.semilogy()
        ax.set_xlabel('z [kpc]')
        ax.set_title('$%.1f < R \\mathrm{~[kpc]}< %.1f$'%(rs[i]-0.5,rs[i]+0.5))
#        ax.set_ylim(1e-3,1)

    axs[0].legend(loc='upper right')
    axs[0].set_ylabel(r'$\rho/\rho_{z=1}$')
    axs[1].set_yticklabels("")
    axs[2].set_yticklabels("")

    plt.subplots_adjust(wspace=.1)
    
def two_sech2(xs,scale1=1.0,scale2=2.0,f=0.5) : 
    return (1.-f)*np.cosh(xs/scale1)**-2+f*np.cosh(xs/scale2)**-2

def make_flare_plot() : 
    import diskfitting
    import fitting

    s = pynbody.load('12M_hr_diff_coeff0.05/10/12M_hr_diff_coeff0.05.01000')
    pynbody.analysis.angmom.faceon(s)
    rs = [4,6,8,10,12,14]
    f,axs = plt.subplots(2,4,figsize=(6,14))

    fits = np.zeros(len(rs))
    errors = np.zeros(len(rs))
    
    for i,r in enumerate(rs):
        p = pynbody.analysis.profile.VerticalProfile(s.s,r-.5,r+.5,3,nbins=20)
        axs.flatten()[i].errorbar(p['rbins'],p['density']/p['density'][0],fmt='.',yerr=p['density']/p['density'][0]/np.sqrt(p['n']),label='R = %d kpc'%r)
        
        
        sn = pynbody.filt.SolarNeighborhood(r-.5,r+.5,4)
        fit,num = diskfitting.two_comp_zfit_simple(s.s[sn]['z'],0.,3.5)
        fit2,num = diskfitting.two_comp_zfit_simple(s.s[sn]['z'],0.2,3.5,func=diskfitting.negtwoexp)
#        res = diskfitting.mcerrors_simple_singlevar(s.s[sn]['y'],fit,0,4)
        #fits[i] = res[:3][fit[:2].argmax()]/2.0
        fits[i] = fit[:2].max()
 #       errors[i] = res[-3:][fit[:2].argmax()]
        axs.flatten()[i].plot(p['rbins'],two_sech2(p['rbins'],fit[0],fit[1],fit[2]),'--')
        axs.flatten()[i].plot(p['rbins'],(1-fit2[2])*np.exp(-p['rbins']/fit2[0])+
                              fit2[2]*np.exp(-p['rbins']/fit2[1]),'--')

        
        axs.flatten()[i].semilogy()
        print fit[:2].min()/2.0,fit[:2].max()/2.0,fit[2]
        print fit2[:2].min(),fit2[:2].max(),fit2[2]
        print fit[:2].min()/2.0/fit2[:2].min(),fit[:2].max()/2.0/fit2[:2].max()


#    axs[0].set_xlabel('z [kpc]')
#    axs[0].set_ylabel(r'$\rho/\rho_{z=1}$')
#    axs[0].semilogy()
#    axs[0].legend(loc='upper right')

#    axs[1].errorbar(rs,fits,yerr=errors,fmt='o')
    axs.flatten()[-1].plot(rs,fits,'o')
    axs.flatten()[-1].set_ylabel('$h_z$ [kpc]')
    axs.flatten()[-1].set_xlabel('R [kpc]')
    axs.flatten()[-1].set_ylim(0,1.5)
    axs.flatten()[-1].set_xlim(3,15)
    
    
def make_resolution_flare_plot(slist) : 
    import diskfitting

    flist = ['12M_hr_diff_coeff0.05/10/12M_hr_diff_coeff0.05.01000', 
             '12M_hr_25pc_soft/10/12M_hr_25pc_soft.01000.gz', 
             '12M_hr_100pc_soft/10/12M_hr_100pc_soft.01000.gz', 
             '12M_hr_x0.5N/10/12M_hr_x0.5N.01000.gz', 
             '12M_hr_x2N/10/12M_hr_x2N.01000.gz', 
             '12M_hr_x4N/10/12M_hr_x4N.01000']

    names = ['fiducial', 
             '$h_s = 25\mathrm{~pc}$', 
             '$h_s = 100\mathrm{~pc}$', 
             '$0.5 N$', 
             '$2 N$', 
             '$4 N$']

#    slist = []

 #   for f in flist : 
 #       slist.append(pynbody.load(f))

#    slist = run_parallel(pynbody.load, flist, [], processes = 6)
    
    fig, axs = plt.subplots(2,1,figsize=(6,14))

    rs = [4,6,8,10,12,14]

    fits = np.zeros((len(flist),len(rs),3))

    for i,s in enumerate(slist) : 
        pynbody.analysis.angmom.faceon(s)
        for j, r in enumerate(rs) : 
            sn = pynbody.filt.SolarNeighborhood(r-.5,r+.5,4)
            fit,num = diskfitting.two_sech_fit_simple(s.s[sn]['z'],0.0,4.0)
        #res = diskfitting.mcerrors_simple_singlevar(s.s[sn]['z'],fit,0,4)
            fits[i,j] = fit
            print flist[i], r, fit

    return fits


def low_alpha_corner(s, rmin=4,rmax=9) : 
    fehfilt = pynbody.filt.BandPass('feh',.1,.2)
    ofefilt = pynbody.filt.LowPass('ofe', -.1)
    sn = pynbody.filt.BandPass('rxy',rmin,rmax)

    f,axs=plt.subplots(2,3,figsize=(10,10))

    axs = axs.flatten()
    print len(s.s[fehfilt&ofefilt&sn])
    axs[0].hist(s.s[fehfilt&ofefilt&sn]['age'],bins=100,histtype='step')
    axs[0].set_xlabel('Age [Gyr]')
    
    axs[1].hist(s.s[fehfilt&ofefilt&sn]['rform'],bins=100,histtype='step',cumulative=True, normed=True)
    axs[1].set_xlabel('$R_{form}$ [kpc]')
    axs[1].set_ylim(0,1.0)
    
    axs[2].hist(s.s[fehfilt&ofefilt&sn]['jz']/s.s[fehfilt&ofefilt&sn]['jzmaxe'],bins=100,histtype='step')
    axs[2].set_xlabel('$J_z/J_{circ}$')
    
    axs[3].plot(s.s[fehfilt&ofefilt&sn]['rform'], 
                s.s[fehfilt&ofefilt&sn]['age'], '.', alpha=.1)
    axs[3].set_xlabel('$R_{form}$')
    axs[3].set_ylabel('Age [Gyr]')

    axs[4].plot(s.s[fehfilt&ofefilt&sn]['rform'], 
                s.s[fehfilt&ofefilt&sn]['jz']/s.s[fehfilt&ofefilt&sn]['jzmaxe'], '.', alpha=.1)
    axs[4].set_xlabel('$R_{form}$')
    axs[4].set_ylabel('$J_z/J_{circ}$')

    prof = pynbody.analysis.profile.Profile(s.s[fehfilt&ofefilt&sn],calc_x=lambda x: x['age'],type='log',nbins=10,min=2)

    prof2 = pynbody.analysis.profile.Profile(s.s[sn],calc_x=lambda x: x['age'],type='log',nbins=10,min=2)

    axs[5].plot(prof['rbins'],np.sqrt(prof['vr_disp']**2+prof['vt_disp']**2+prof['vz_disp']**2),'k-',label=r'$\sigma_{tot}$',linewidth=2)
    axs[5].plot(prof['rbins'],prof['vr_disp'],'k--',label=r'$\sigma_{tot}$',linewidth=2)
    axs[5].plot(prof2['rbins'],np.sqrt(prof2['vr_disp']**2+prof2['vt_disp']**2+prof2['vz_disp']**2),'r-',label=r'$\sigma_{tot}$',linewidth=2)
    axs[5].plot(prof2['rbins'],prof2['vr_disp'],'r--',label=r'$\sigma_{tot}$',linewidth=2)
    axs[5].set_xlabel('Age [Gyr]')
    axs[5].set_ylabel('$\sigma$')

    
