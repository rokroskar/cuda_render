"""

plots to address referee report #1 of Radial Migration in Disk Galaxies I

"""

import pynbody
from pynbody.array import SimArray
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import gaussian_kde as kde
from parallel_util import interruptible, run_parallel

def first() : 
    figs = []

    s1 = pynbody.load('/home/itp/roskar/isolated_runs/12M_hr/6/12M_hr.00650')
    s2 = pynbody.load('/home/itp/roskar/isolated_runs/12M_hr/7/12M_hr.00700')

    for s in [s1,s2] :
        # center and align
        pynbody.analysis.angmom.faceon(s.g, disk_size='3 kpc')
        s.s['jz'] = s.s['j'][:,2]
        
    s1.s['dj'] = s2.s['jz'][:len(s1.s)] - s1.s['jz']
    s1.s['de'] = s2.s['te'][:len(s1.s)] - s1.s['te']

        # 43 km/s/kpc pattern
    one = (pynbody.filt.BandPass('jz', 1250, 1800) & pynbody.filt.LowPass('dj', -300)) | \
        (pynbody.filt.BandPass('jz', 750, 1250) & pynbody.filt.HighPass('dj', 300))

    # 68 km/s/kpc pattern
    two = (pynbody.filt.BandPass('jz', 950, 1100) & pynbody.filt.LowPass('dj', -250)) | \
        (pynbody.filt.BandPass('jz', 700, 850) & pynbody.filt.HighPass('dj', 250))

    # 20 km/s/kpc pattern
    three = pynbody.filt.HighPass('jz', 2200) & pynbody.filt.HighPass('dj', 300)

    k_one = kde(s1.s[one]['de']/s1.s[one]['dj']-43)
    k_two = kde(s1.s[two]['de']/s1.s[two]['dj']-68)
    k_three = kde(s1.s[three]['de']/s1.s[three]['dj']-20)

    omega = np.linspace(-50,50,1000)

    # first plot from #3

    figs.append(plt.figure())

    plt.plot(omega,k_one(omega),label='$\Omega_p$ = 43 km/s/kpc')
    plt.plot(omega,k_two(omega),label='$\Omega_p$ = 68 km/s/kpc')
    plt.plot(omega,k_three(omega),label='$\Omega_p$ = 20 km/s/kpc')
    plt.xlabel('$dE/dj_z - \Omega_p $ [km s$^{-1}$ kpc$^{-1}$]', fontsize='large')
    plt.legend()

    # second plot from #3

    figs.append(plt.figure())
    
    pynbody.plot.generic.gauss_kde(SimArray(np.abs(s1.s[one]['dj'])), SimArray(np.abs(s1.s[one]['de']/s1.s[one]['dj']-43)), scalemin=.1, xlabel='$|\Delta j_z|$ [kpc km s$^{-1}$]', ylabel='$|\Delta E/\Delta j_z - 43| $ [km s$^{-1}$ kpc$^{-1}$]', gridsize=(200,200))

    figs.append(plt.figure())

    pynbody.plot.generic.gauss_kde(SimArray(np.abs(s1.s[two]['dj'])), SimArray(np.abs(s1.s[two]['de']/s1.s[two]['dj']-68)), scalemin=.1, xlabel='$|\Delta j_z|$ [kpc km s$^{-1}$]', ylabel='$|dE/dj_z - 68| $ [km s$^{-1}$ kpc$^{-1}$]', x_range=[300,400])
    
    figs.append(plt.figure())

    pynbody.plot.generic.gauss_kde(SimArray(np.abs(s1.s[three]['dj'])), SimArray(np.abs(s1.s[three]['de']/s1.s[three]['dj']-20)), scalemin=.1, xlabel='$|\Delta j_z|$ [kpc km s$^{-1}$]', ylabel='$|dE/dj_z - 20| $ [km s$^{-1}$ kpc$^{-1}$]')

    for i,fig in enumerate(figs) : 
        fig.savefig('/home/itp/roskar/my_papers/spiral_structure/ref_report1/fig7Gyr_'+str(i+1)+'.pdf',format='pdf')

    del(figs)

    del((s1, s2))

def second():

    s1 = pynbody.load('/home/itp/roskar/isolated_runs/12M_hr/4/12M_hr.00480')
    s2 = pynbody.load('/home/itp/roskar/isolated_runs/12M_hr/5/12M_hr.00500')

    for s in [s1,s2] :
        # center and align
        pynbody.analysis.angmom.faceon(s.g, disk_size='3 kpc')
        s.s['jz'] = s.s['j'][:,2]
        
    s1.s['dj'] = s2.s['jz'][:len(s1.s)] - s1.s['jz']
    s1.s['de'] = s2.s['te'][:len(s1.s)] - s1.s['te']

    # 42 km/s/kpc pattern
    one = (pynbody.filt.BandPass('jz', 1450, 1800) & pynbody.filt.LowPass('dj', -300)) | \
        (pynbody.filt.BandPass('jz', 1000, 1400) & pynbody.filt.HighPass('dj', 300))

    # 59 km/s/kpc pattern
    two = (pynbody.filt.BandPass('jz', 1000, 1300) & pynbody.filt.LowPass('dj', -300)) | \
        (pynbody.filt.BandPass('jz', 650, 850) & pynbody.filt.HighPass('dj', 300))

    # 24 km/s/kpc pattern
    three = (pynbody.filt.HighPass('jz', 1800) & pynbody.filt.HighPass('dj', 300)) | \
        (pynbody.filt.HighPass('jz', 2100) & pynbody.filt.LowPass('dj', -300))

    k_one = kde(s1.s[one]['de']/s1.s[one]['dj']-41)
    k_two = kde(s1.s[two]['de']/s1.s[two]['dj']-58)
    k_three = kde(s1.s[three]['de']/s1.s[three]['dj']-25)

    omega = np.linspace(-50,50,1000)

    # first plot from #3

    plt.figure()

#plt.plot(omega,k_one(omega),label='$\Omega_p$ = 41 km/s/kpc')
    plt.plot(omega,k_two(omega),label='$\Omega_p$ = 58 km/s/kpc')
    plt.plot(omega,k_three(omega),label='$\Omega_p$ = 25 km/s/kpc')
    plt.xlabel('$dE/dj_z - \Omega_p $ [km s$^{-1}$ kpc$^{-1}$]')
    plt.legend()

    plt.savefig('/home/itp/roskar/my_papers/spiral_structure/ref_report1/fig5Gyr.pdf',format='pdf')

    del((s1, s2))


def third():
    s1 = pynbody.load('/home/itp/roskar/isolated_runs/12M_hr/2/12M_hr.00270')
    s2 = pynbody.load('/home/itp/roskar/isolated_runs/12M_hr/3/12M_hr.00300')

    for s in [s1,s2] :
        # center and align
        pynbody.analysis.angmom.faceon(s.g, disk_size='3 kpc')
        s.s['jz'] = s.s['j'][:,2]
        
    s1.s['dj'] = s2.s['jz'][:len(s1.s)] - s1.s['jz']
    s1.s['de'] = s2.s['te'][:len(s1.s)] - s1.s['te']

    # 55 km/s/kpc pattern
    one = (pynbody.filt.BandPass('jz', 1000, 1300) & pynbody.filt.LowPass('dj', -300)) | \
        (pynbody.filt.BandPass('jz', 500, 750) & pynbody.filt.HighPass('dj', 300))

    # 71 km/s/kpc pattern
    two = (pynbody.filt.BandPass('jz', 1000, 1300) & pynbody.filt.LowPass('dj', -300)) | \
        (pynbody.filt.BandPass('jz', 650, 850) & pynbody.filt.HighPass('dj', 300))

    # 28 km/s/kpc pattern
    three = (pynbody.filt.HighPass('jz', 1400) & pynbody.filt.HighPass('dj', 300)) | \
        (pynbody.filt.HighPass('jz', 1650) & pynbody.filt.LowPass('dj', -300))

    k_one = kde(s1.s[one]['de']/s1.s[one]['dj']-55)
#k_two = kde(s1.s[two]['de']/s1.s[two]['dj']-58)
    k_three = kde(s1.s[three]['de']/s1.s[three]['dj']-28)

    omega = np.linspace(-50,50,1000)

    # first plot from #3

    plt.figure()

    plt.plot(omega,k_one(omega),label='$\Omega_p$ = 55 km/s/kpc')
#plt.plot(omega,k_two(omega),label='$\Omega_p$ = 58 km/s/kpc')
    plt.plot(omega,k_three(omega),label='$\Omega_p$ = 28 km/s/kpc')
    #plt.hist(s1.s[three]['de']/s1.s[three]['dj']-28, range = [-50,50], bins=100, histtype='step',normed=True)
    plt.xlabel('$dE/dj_z - \Omega_p $ [km s$^{-1}$ kpc$^{-1}$]')
    plt.legend()

    plt.savefig('/home/itp/roskar/my_papers/spiral_structure/ref_report1/fig3Gyr.pdf',format='pdf')


def horseshoe_orbits (processors=4) : 
    import glob, orbits
    from pynbody.analysis.profile import Profile
    import scipy.interpolate as interpol

    s1 = pynbody.load('/home/itp/roskar/isolated_runs/12M_hr/6/12M_hr.00650')
    s2 = pynbody.load('/home/itp/roskar/isolated_runs/12M_hr/7/12M_hr.00700')
#    s2 = pynbody.load('/home/itp/roskar/isolated_runs/12M_hr/6/12M_hr.00670')


    

    for s in [s1,s2] :
    # center and align
        pynbody.analysis.angmom.faceon(s.g, disk_size='3 kpc')
        s.s['jz'] = s.s['j'][:,2]
        
    s1.s['dj'] = s2.s['jz'][:len(s1.s)] - s1.s['jz']
    s1.s['de'] = s2.s['te'][:len(s1.s)] - s1.s['te']
    
    # determine corotation resonance locations
    p1 = Profile(s1, max=15, nbins=20, load_from_file=True)
    p2 = Profile(s2, max=15, nbins=20, load_from_file=True)
    
    patspeeds = [21, 43, 70]

    cr_r  = SimArray(interpol.interp1d(p1['omega'][::-1], 
                                       p1['rbins'][::-1])(patspeeds),p1['rbins'].units)
    cr_jz = SimArray(interpol.interp1d(p1['rbins'], p1['j_circ'])(cr_r),p1['j_circ'].units)
    cr_e  = SimArray(interpol.interp1d(p1['rbins'], p1['E_circ'])(cr_r),p1['E_circ'].units)
   
    flist = glob.glob('/home/itp/roskar/isolated_runs/12M_hr/6/*.00???.gz')
#    flist += glob.glob('/home/itp/roskar/isolated_runs/12M_hr/7/*.00???.gz')
    flist.sort()

    sfirst = pynbody.load(flist[0], only_header=True)

    # 43 km/s/kpc pattern
    one = ((pynbody.filt.BandPass('jz', 1250, 1800) & pynbody.filt.LowPass('dj', -400)) | \
        (pynbody.filt.BandPass('jz', 750, 1250) & pynbody.filt.HighPass('dj', 400))) & \
        pynbody.filt.LowPass('tform', sfirst.properties['a'])

    # 68 km/s/kpc pattern
    two = (pynbody.filt.BandPass('jz', 950, 1100) & pynbody.filt.LowPass('dj', -250)) | \
        (pynbody.filt.BandPass('jz', 700, 850) & pynbody.filt.HighPass('dj', 250))

    # 20 km/s/kpc pattern
    three = (pynbody.filt.HighPass('jz', 2200) & pynbody.filt.HighPass('dj', 300) | \
        pynbody.filt.HighPass('jz', 2200) & pynbody.filt.LowPass('dj', -300)) & \
        pynbody.filt.LowPass('tform', sfirst.properties['a'])
    

    simple = ((pynbody.filt.HighPass('dj', 250) |
               pynbody.filt.LowPass('dj', -250)) & 
              pynbody.filt.LowPass('tform',sfirst.properties['a']))

#    spec_inds = [640778,1052173,1312193,632547,691094,945673]
#    spec_inds.sort()

 #   pinds = simple.where(s1.s)[0]
    pinds = one.where(s1.s)[0] # used for the horseshoe figures
    #pinds = three.where(s1.s)[0]
    #pinds = spec_inds
    cr_in_jz = cr_jz[1] - 100
    cr_out_jz = cr_jz[1] + 100
    cr_in_e = interpol.interp1d(p1['j_circ'],p1['E_circ'])(cr_in_jz)
    cr_out_e = interpol.interp1d(p1['j_circ'],p1['E_circ'])(cr_out_jz)
       
    #pinds = np.squeeze(np.where(
    #        (
    #            (np.abs(s1.s['j'][:,2] - cr_in_jz) < 0.01*cr_in_jz) &
    #            (np.abs(s1.s['te'] -     cr_in_e)  < np.abs(0.01*cr_in_e)) &
    #            (s2.s['j'][:len(s1.s),2] - s1.s['j'][:,2] > 200) &
    #            (s1.s['tform'] < sfirst.properties['a'])) |
    #        (
    #            (np.abs(s1.s['j'][:,2] - cr_out_jz) < 0.01*cr_out_jz) &
    #            (np.abs(s1.s['te'] -     cr_out_e)  < np.abs(0.01*cr_out_e)) &
    #            (s2.s['j'][:len(s1.s),2] - s1.s['j'][:,2] < 200) &
    #            (s1.s['tform'] < sfirst.properties['a'])))
    #                   )

    
    print len(pinds)



    pos, vel, mass, phi, time = orbits.trace_orbits_parallel(flist,pinds,processors)
#    pos, vel, time = orbits.trace_orbits(flist,pinds)

    x = pos[:,:,0].T
    y = pos[:,:,1].T
    z = pos[:,:,2].T
    vx = vel[:,:,0].T
    vy = vel[:,:,1].T
    vz = vel[:,:,2].T
    phi = phi.T

    dt = time - time[0]

    omega = 39

    x_rot = x*np.cos(-omega*dt) - y*np.sin(-omega*dt)
    y_rot = x*np.sin(-omega*dt) + y*np.cos(-omega*dt)
    
    Rc = 5.5 # corotation radius?
    #phi = np.arctan2(y_rot, x_rot)
    
    xs = Rc*np.cos(phi)
    ys = Rc*np.sin(phi)

    return x, y, z, x_rot, y_rot, vx, vy, vz, mass, phi, time



def plot_horseshoes_wavelet(x,y,t,omega,p = None, npages=2) : 
    from matplotlib.collections import LineCollection
    from matplotlib.backends.backend_pdf import PdfPages
    
    import spiral_structure as ss
    import orbits

    plt.rc('font', size=15)

    pp = PdfPages('horseshoe_orbit_plots.pdf')

    nrow = 4
    ncol = 4
    perpage = nrow*ncol/2
    nmax = len(x)
    
    ntot = npages*perpage

    inds = np.arange(0,nmax,ntot)
    
    #inds = np.random.rand(ntot)*len(x)

    if p is None: 
        sp_amp, sp_time, ft,fqs,gauss = ss.get_fulldisk_band_amplitude('/home/itp/roskar/isolated_runs/12M_hr/complete_fourier.npz', 2.0,9.0,omega-5,omega+5, window=True)
    else : 
        sp_amp, sp_time, ft,fqs,gauss = ss.get_band_amplitude('/home/itp/roskar/isolated_runs/12M_hr/complete_fourier.npz', 2.0,9.0,omega-5,omega+5, get_crs(p,omega)[0], window=True)


    sp_ind = np.where((sp_time >= t[0]) & (sp_time <= t[-1]))[0]
    sp_amp = sp_amp[sp_ind]
    sp_amp = np.abs(sp_amp)
    sp_time = sp_time[sp_ind]

    dt = t - t[0]
    xr = x*np.cos(-omega*dt) - y*np.sin(-omega*dt)
    yr = x*np.sin(-omega*dt) + y*np.cos(-omega*dt)
    r = np.sqrt(x**2 + y**2)

    for i in range(npages) : 
        fig = plt.figure(figsize=(15,15))
        
        for j in range(perpage) : 

            if j < perpage-1 : 
                ax = plt.subplot(nrow,ncol,j*2+1)
                pind = inds[i*perpage+j]
                points = np.array([xr[pind], yr[pind]]).T.reshape(-1, 1, 2)
                plt.xlim(-10,10)
                plt.ylim(-10,10)
                ax.set_aspect('equal')
                
                if j != perpage-ncol/2 : 
                    ax.xaxis.set_ticklabels("")
                    ax.yaxis.set_ticklabels("")
                else : 
                    plt.xlabel('$x~\\mathrm{[kpc]}$')
                    plt.ylabel('$y~\\mathrm{[kpc]}$')
                
                plt.annotate(str(j+1), (-8,8), fontsize=12)

            else  :
                ax = plt.axes([0.60,0.095,0.3,0.18])
                points = np.array([sp_time,sp_amp]).T.reshape(-1,1,2)
                plt.ylim(sp_amp.min(), sp_amp.max())

            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=plt.get_cmap('jet'),
                                norm=plt.Normalize(sp_amp.min(), sp_amp.max()))
            lc.set_array(sp_amp)

            ax.add_collection(lc)


            if j == perpage-1 :
                plt.xlabel('$t~\\mathrm{[Gyr]}$')
                plt.ylabel('$A_2$')
                plt.xlim(sp_time[0],sp_time[-1])
                plt.ylim(sp_amp.min(),sp_amp.max())
                plt.annotate('spiral amplitude', (6.1,0.04), fontsize=15)
                
            else :
                ax = plt.subplot(nrow,ncol,j*2+2, aspect='equal')
                cf = orbits.orbit_cwt(x[pind],y[pind],t,ax=ax,plot_ridges=False)
                
                ax2 = ax.twinx()
                ax2.plot(t,r[pind],'k',linewidth=2)
                ax.set_xlim(t.min(),t.max())
                ax2.set_ylim(r.min(),r.max())

                if j != perpage-ncol/2 : 
                    ax.xaxis.set_ticklabels("")
                    ax.yaxis.set_ticklabels("")
                    ax2.yaxis.set_ticklabels("")
                    
                else : 
                    ax.set_ylabel('$\\mathrm{scale}$')
                    ax.set_xlabel('$t~\\mathrm{[Gyr]}$')
                    ax2.set_ylabel('$R\\mathrm{ [kpc]}$')

                    
                    #cb = plt.colorbar(cf, format = '$%.2f$')
                    #cb.set_label('$\\mathrm{log(Power)}$')

        pp.savefig()

    pp.close()


def plot_horseshoes(x,y,t,omega,p,npages=1) : 

    """

    multicolored line example taken from 

    http://matplotlib.sourceforge.net/examples/pylab_examples/multicolored_line.html

    """

    from matplotlib.collections import LineCollection
    from matplotlib.backends.backend_pdf import PdfPages
    from mpl_toolkits.axes_grid1 import ImageGrid
    import spiral_structure as ss
    import orbits

    pp = PdfPages('horseshoe_orbit_plots.pdf')

    nrow = 3
    ncol = 2
    perpage = nrow*ncol
    nmax = len(x)
    
    ntot = npages*perpage

#
    inds = np.arange(0,nmax)
    #inds = np.random.rand(ntot)*len(x)

    sp_amp, sp_time, ft,fqs,gauss = ss.get_band_amplitude('/home/itp/roskar/isolated_runs/12M_hr/complete_fourier.npz',
                                            2.0,9.0,omega-5,omega+5,get_crs(p,omega)[0], window=True)

    sp_ind = np.where((sp_time >= t[0]) & (sp_time <= t[-1]))[0]
    sp_amp = sp_amp[sp_ind]
    sp_amp = np.abs(sp_amp)
    sp_time = sp_time[sp_ind]
    sp_phi = np.arctan2(sp_amp.imag, sp_amp.real)
    
#    r = np.sqrt(x**2 + y**2)
#    jz = np.cross(np.array([x,y]).T, np.array([vx,vy]).T).T
    
    dt = t - t[0]
    xr = x*np.cos(-omega*dt) - y*np.sin(-omega*dt)
    yr = x*np.sin(-omega*dt) + y*np.cos(-omega*dt)
    
    for i in range(npages) : 
        fig = plt.figure(figsize=(13,15))
        #grid = ImageGrid(fig,111,nrows_ncols=(nrow,ncol),
         #            axes_pad=.1,
         #            label_mode='1')
        
        for j in range(perpage) : 
            #plt.subplots_adjust(wspace=-.14,hspace=0.03)    
            #ax = grid[j]
            ax = plt.subplot(nrow,ncol,j+1)

            if j < perpage-1 :
                pind = inds[i*perpage+j]
                points = np.array([xr[pind], yr[pind]]).T.reshape(-1, 1, 2)
                plt.xlim(-10,10)
                plt.ylim(-10,10)
                ax.set_aspect('equal')

            else : 
                points = np.array([sp_time,sp_amp]).T.reshape(-1,1,2)
                plt.ylim(sp_amp.min(),sp_amp.max())
                
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=plt.get_cmap('jet'),
                                norm=plt.Normalize(sp_amp.min(), sp_amp.max()))
            lc.set_array(sp_amp)

            ax.add_collection(lc)

            if j != perpage-ncol: 
                ax.xaxis.set_ticklabels("")
                ax.yaxis.set_ticklabels("")

            else : 
                plt.xlabel('$x$ [kpc]')
                plt.ylabel('$y$ [kpc]')

            if j == perpage-1 :
                plt.xlabel('$t$ [Gyr]')
                axt = ax.twinx()
                plt.ylabel('$A_2$')
                plt.plot(0,0)
                plt.xlim(sp_time[0],sp_time[-1])
                plt.ylim(sp_amp.min(),sp_amp.max())
                #axt.set_yticks([0.015,0.03,0.045])
                plt.annotate('spiral amplitude', (6.1,0.04), fontsize=15)
                
            plt.annotate(str(j+1), (-8,8), fontsize=12)
            
            
        


 #       ax = plt.subplot(nrow,ncol,ntot-ncol)
 #       box = ax.get_frame().get_window_extent()
 #       plt.annotate('', xy=(box.x1,box.y0-10), xycoords='figure pixels', 
 #                    xytext=(box.x0,box.y0-10), textcoords='figure pixels', 
 #                    arrowprops=dict(arrowstyle="->"))

        pp.savefig()

    pp.close()

    return sp_time, sp_amp


def dedj_phaseplots(s1, s2, s3, s4) : 
    import scipy.interpolate as interpol
    import mpl_toolkits.axes_grid.parasite_axes as pa
    import matplotlib.transforms as transforms
    from pynbody.array import SimArray
    from pynbody.analysis.profile import Profile

#    s1 = pynbody.load('/home/itp/roskar/isolated_runs/12M_hr/6/12M_hr.00600')
#    s2 = pynbody.load('/home/itp/roskar/isolated_runs/12M_hr/6/12M_hr.00650')
#    s3 = pynbody.load('/home/itp/roskar/isolated_runs/12M_hr/7/12M_hr.00700')
#    s4 = pynbody.load('/home/itp/roskar/isolated_runs/12M_hr/7/12M_hr.00750')



    for s in [s1,s2,s3,s4] :
        cen = pynbody.analysis.halo.center(s, retcen=True)
        if cen.any() > 1e-5 :
            pynbody.analysis.angmom.faceon(s, disk_size='3 kpc')
        

    s1.s['dj'] = s4.s['jz'][:len(s1.s)] - s1.s['jz']
    s1.s['de'] = s4.s['te'][:len(s1.s)] - s1.s['te']
    s2.s['dj'] = s3.s['jz'][:len(s2.s)] - s2.s['jz']
    s2.s['de'] = s3.s['te'][:len(s2.s)] - s2.s['te']

    # we are looking at changes from 6-7.5 and 6.5-7 so generate profiles for s1 and s2

    p1 = Profile(s1, max=15, nbins=20)
    p2 = Profile(s2, max=15, nbins=20)
    p3 = Profile(s3, max=15, nbins=20)
    p4 = Profile(s4, max=15, nbins=20)
    
   
    colors1 = ['bo', 'go', 'ro', 'yo']
    colors2 = ['bx', 'gx', 'rx', 'yx']
    
    #fig = plt.figure(figsize=(12,12))
    #ax1 = fig.add_subplot(221)
    #ax = [pa.SubplotHost(fig,222), pa.SubplotHost(fig,223), pa.SubplotHost(fig,224)]

    patspeeds = [21,43,70]

    bp_low = pynbody.filt.BandPass('dj', -300,-200)
    bp_high = pynbody.filt.BandPass('dj', 200, 300)
    # 43 km/s/kpc pattern
    one = ((pynbody.filt.BandPass('jz', 1250, 1600) & bp_low) | \
        (pynbody.filt.BandPass('jz', 1100, 1250) & bp_high)) & \
        pynbody.filt.LowPass('tform', s2.properties['a'])

    # 68 km/s/kpc pattern
    two = ((pynbody.filt.BandPass('jz', 950, 1100) & bp_low) | \
        (pynbody.filt.BandPass('jz', 700, 850) & bp_high)) & \
        pynbody.filt.LowPass('tform', s2.properties['a'])

    # 20 km/s/kpc pattern
    three = ((pynbody.filt.BandPass('jz', 2300, 2500) & bp_high | \
        pynbody.filt.BandPass('jz', 2500,2700) & bp_low)) & \
        pynbody.filt.LowPass('tform', s1.properties['a'])
    

#    kd, omega, ind1 = get_kd(p2,21.0, s2, s3,three) 
#    plt.plot(omega,kd(omega),label="$\Omega_p = 21$", linewidth=2)
    kd, omega, ind2 = get_kd(p2,43.0, s2, s3,one)  
    plt.plot(omega,kd(omega), label="$\Omega_p = 43$", linewidth=2)
#    kd, omega, ind3 = get_kd(p2,70.0, s2, s3,two)  
#    plt.plot(omega,kd(omega), label="$\Omega_p = 70$",linewidth=2)
    plt.xlabel('$\Delta E / \Delta J_z - \Omega_p$',fontweight='bold')
    plt.legend()

    def get_plot_e_vs_j(ax, s1, s2, p1, p2, ps, ind, color):
        cr_r, cr_jz, cr_e = get_crs(p1, ps)

        rand_ind = np.round(np.random.rand(np.min([len(ind),50]))*(len(ind)-1))
        print 'rand ind = ', len(rand_ind)

        for j, k in enumerate(rand_ind) : 
            k = np.int(k)
            # plot the connecting lines
            ax.plot([s2.s['j'][ind[k],2], s1.s['j'][ind[k],2]],
                       [s2.s['te'][ind[k]],  s1.s['te'][ind[k]]],'black', alpha = 0.4)
            
            ax.plot(s1.s['j'][ind[k],2], s1.s['te'][ind[k]], color+"o", alpha = 0.4)
            ax.plot(s2.s['j'][ind[k],2], s2.s['te'][ind[k]], color+"x", alpha = 0.4)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.plot(p1['j_circ'], p1['E_circ'], scalex=False, scaley=False, linewidth=2, color = 'orange')
        ax.plot(cr_jz, cr_e, 'ro')
        ax.plot(p2['j_circ'], p2['E_circ'], '--', scalex=False, scaley=False, linewidth=2, color = 'orange')
        ax.set_xlabel('$J_z$')
        ax.set_ylabel('$E$')
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

        # draw the E = Omega_p * J tangent lines
        #        for j, op in enumerate(patspeeds) :

        ps = SimArray(ps, "km s**-1 kpc**-1")
        ax.plot([xlim[0], xlim[1]],[cr_e-ps*(cr_jz-xlim[0]), 
                                    cr_e-ps*(cr_jz-xlim[1])], color,linewidth=2)
#        ax.plot([xlim[0], xlim[1]],[cr_e-(ps-5)*(cr_jz-xlim[0]), 
#                                       cr_e-(ps-5)*(cr_jz-xlim[1])], color,linewidth=2, linestyle='dashed')
#        ax.plot([xlim[0], xlim[1]],[cr_e-(ps+5)*(cr_jz-xlim[0]), 
#                                       cr_e-(ps+5)*(cr_jz-xlim[1])], color,linewidth=2, linestyle='dashed')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
    #get_plot_e_vs_j(ax[0],s2,s3,p2,p3,patspeeds[0],ind1,'b')
    #fig.add_subplot(ax[0])
    #get_plot_e_vs_j(ax[1],s2,s3,p2,p3,patspeeds[1],ind2,'g')
    #fig.add_subplot(ax[1])
    #get_plot_e_vs_j(ax[2],s2,s3,p2,p3,patspeeds[2],ind3,'r')
    #fig.add_subplot(ax[2])
    
def get_crs(p,patspeed):
    import scipy.interpolate as interpol
    
    cr_r  = SimArray(interpol.interp1d(p['omega'][::-1], 
                                       p['rbins'][::-1])(patspeed),p['rbins'].units)
    cr_jz = SimArray(interpol.interp1d(p['rbins'], p['j_circ'])(cr_r),p['j_circ'].units)
    cr_e  = SimArray(interpol.interp1d(p['rbins'], p['E_circ'])(cr_r),p['E_circ'].units)

    return cr_r, cr_jz, cr_e

def get_kd(p, patspeed, s_loc1, s_loc2, filt) : 
    from scipy import interpolate as interpol

    cr_r, cr_jz, cr_e = get_crs(p, patspeed)
    
    cr_in_jz = cr_jz - cr_jz*.1
    cr_out_jz = cr_jz + cr_jz*.1
    cr_in_e = interpol.interp1d(p['j_circ'],p['E_circ'])(cr_in_jz)
    cr_out_e = interpol.interp1d(p['j_circ'],p['E_circ'])(cr_out_jz)
        
    s_loc1.s['dj'] = s_loc2.s['jz'][:len(s_loc1.s)] - s_loc1.s['jz']
    s_loc1.s['de'] = s_loc2.s['te'][:len(s_loc1.s)] - s_loc1.s['te']
    
    
    ind1 = np.squeeze(np.where(
            (
                (np.abs(s_loc1.s['j'][:,2] - cr_in_jz) < 0.01*cr_in_jz) &
                (np.abs(s_loc1.s['te'] -     cr_in_e)  < np.abs(0.01*cr_in_e))) |
            (
                (np.abs(s_loc1.s['j'][:,2] - cr_out_jz) < 0.01*cr_out_jz) &
                (np.abs(s_loc1.s['te'] -     cr_out_e)  < np.abs(0.01*cr_out_e)))
                     ))

    dj = np.abs(s_loc1.s['dj'][ind1])
    dj.sort()
    jcut = dj[len(dj)*.95]
    
    print jcut
    
#    ind = ind1[np.squeeze(np.where(np.abs(s_loc1.s['dj'][ind1])))]

    ind = np.squeeze(np.where(
            (
                (np.abs(s_loc1.s['j'][:,2] - cr_in_jz) < 0.01*cr_in_jz) &
                (np.abs(s_loc1.s['te'] -     cr_in_e)  < np.abs(0.1*cr_in_e)) &
                (s_loc1.s['dj'] > jcut)) |
            (
                (np.abs(s_loc1.s['j'][:,2] - cr_out_jz) < 0.01*cr_out_jz) &
                (np.abs(s_loc1.s['te'] -     cr_out_e)  < np.abs(0.1*cr_out_e)) &
                (s_loc1.s['dj'] < -jcut))
                     ))

#        ind = np.squeeze(filt.where(s_loc1.s))
    omega = np.linspace(-40,40,1000)
    kd = kde(s_loc1.s[ind]['de']/s_loc1.s[ind]['dj']-patspeed)
    
    return kd, omega, ind
    

def plot_kd(s1,s2,patspeeds,filt=None) :

    for s in [s1,s2] :
        cen = pynbody.analysis.halo.center(s, retcen=True)
        if cen.any() > 1e-5 :
            pynbody.analysis.angmom.faceon(s, disk_size='3 kpc')
        

    p = pynbody.analysis.profile.Profile(s1,max=15, nbins=20, min=1e-3, type='lin')

    for patspeed in patspeeds:

        kd,omega,ind = get_kd(p,patspeed,s1,s2,filt)
    
        plt.plot(omega,kd(omega),label='$\Omega_p = ' + str(patspeed) + '$',linewidth=2)

    plt.xlabel('$\Delta E/\Delta j_z - \Omega_p$')
    plt.title('%.1f - %.1f Gyr'%(s1.properties['a'], s2.properties['a']))
    plt.legend()

def spiral_amps() : 
    import spiral_structure as ss
    plt.figure()
    sa1,st1,ft,fqs,w = ss.get_fulldisk_band_amplitude('/home/itp/roskar/isolated_runs/12M_hr/complete_fourier_fulldisk.npz', 2.0, 9.0, 15, 25)#, 11.5,window=True)
    sa2,st2,ft,fqs,w = ss.get_fulldisk_band_amplitude('/home/itp/roskar/isolated_runs/12M_hr/complete_fourier_fulldisk.npz', 2.0, 9.0, 40, 50)#, 5.5,window=True)
    sa3,st3,ft,fqs,w = ss.get_fulldisk_band_amplitude('/home/itp/roskar/isolated_runs/12M_hr/complete_fourier_fulldisk.npz', 2.0, 9.0, 65, 75)#, 4,window=True)

    ax = plt.subplot(111)
    l1 = plt.plot(st1,abs(sa1[0:len(st1)]),'r', linewidth=2)
    plt.ylabel('$A_2$',fontsize=20,fontweight='bold')
    plt.xlabel('$t/Gyr$', fontsize=20,fontweight='bold')
#    axt = ax.twinx()
    l2 = plt.plot(st2,abs(sa2[0:len(st2)]),'g', linewidth=2)
    l3 = plt.plot(st3,abs(sa3[0:len(st3)]),'b', linewidth=2)
    
    plt.xlim(5,8)
    #plt.ylim(0,.004)
    plt.ylabel('$A_2$',fontsize=20,fontweight='bold')
    

    plt.legend([l1,l2,l3], ['$\Omega_p = 15-25$','$\Omega_p = 40-50$', '$\Omega_p = 65-75$'], loc='upper left')


def plot_horseshoes_wavelet_small(x,y,z,vx,vy,vz,phi,t,p,omega1,omega2):
    #
    # used omega1 = 65 and omega2 = 39 for figure 11 in the paper
    #

    from matplotlib.collections import LineCollection
    from matplotlib.backends.backend_pdf import PdfPages
    import spiral_structure as ss
    import orbits

    plt.rc('font', size=15)
    fig = plt.figure(figsize=(13.5,11))
    
    #set up spiral amps
    sp_amp1, sp_time1, ft1,fqs1,gauss1 = ss.get_band_amplitude('/home/itp/roskar/isolated_runs/12M_hr/complete_fourier.npz', 4.0,7.0,omega1-5,omega1+5, get_crs(p,omega1)[0],window=True)
    sp_ind1 = np.where((sp_time1 >= t[0]) & (sp_time1 <= t[-1]))[0]
    sp_amp1 = sp_amp1[sp_ind1]
    sp_amp1 = np.abs(sp_amp1)
    sp_time1 = sp_time1[sp_ind1]

    sp_amp2, sp_time2, ft2,fqs2,gauss2 = ss.get_band_amplitude('/home/itp/roskar/isolated_runs/12M_hr/complete_fourier.npz', 4.0,7.0,omega2-5,omega2+5, get_crs(p,omega2)[0],window=True)
    sp_ind2 = np.where((sp_time2 >= t[0]) & (sp_time2 <= t[-2]))[0]
    sp_amp2 = sp_amp2[sp_ind2]
    sp_amp2 = np.abs(sp_amp2)
    sp_time2 = sp_time2[sp_ind2]


    dt = t - t[0]
    xr1 = x*np.cos(-omega1*dt) - y*np.sin(-omega1*dt)
    yr1 = x*np.sin(-omega1*dt) + y*np.cos(-omega1*dt)
    xr2 = x*np.cos(-omega2*dt) - y*np.sin(-omega2*dt)
    yr2 = x*np.sin(-omega2*dt) + y*np.cos(-omega2*dt)
    r = np.sqrt(x**2 + y**2)
        
    for j in range(2) : 
        ax = plt.subplot(3,3,j*3+1)
        pind = j
        points = np.array([xr1[pind], yr1[pind]]).T.reshape(-1, 1, 2)
        plt.xlim(-10,10)
        plt.ylim(-10,10)
        ax.set_aspect('equal')
        plt.xlabel('$x~\\mathrm{[kpc]}$')
        plt.ylabel('$y~\\mathrm{[kpc]}$')
       

        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=plt.get_cmap('jet'),
                            norm=plt.Normalize(sp_amp1.min(), sp_amp1.max()))
        lc.set_array(sp_amp1)

        ax.add_collection(lc)

        ax = plt.subplot(3,3,j*3+2)
        points = np.array([xr2[pind], yr2[pind]]).T.reshape(-1, 1, 2)
        plt.xlim(-10,10)
        plt.ylim(-10,10)
        ax.set_aspect('equal')
        plt.xlabel('$x~\\mathrm{[kpc]}$')
        plt.ylabel('$y~\\mathrm{[kpc]}$')
        

        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=plt.get_cmap('jet'),
                            norm=plt.Normalize(sp_amp2.min(), sp_amp2.max()))
        lc.set_array(sp_amp2)

        ax.add_collection(lc)

        ax = plt.subplot(3,3,j*3+3)
        cf = orbits.orbit_cwt(x[pind],y[pind],t,ax=ax,plot_ridges=False)
        ax.set_aspect('equal')
        ax2 = ax.twinx()
        ax2.plot(t,r[pind],'k',linewidth=2)
        ax.set_xlim(t.min(),t.max())
        ax2.set_ylim(0,10)
        ax.set_ylabel('$\\mathrm{scale}$')
        ax.set_xlabel('$t~\\mathrm{[Gyr]}$')
        ax2.set_ylabel('$R\\mathrm{ [kpc]}$')
        

    ax = plt.subplot(3,3,7, aspect='auto')
    points = np.array([sp_time1,sp_amp1]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=plt.get_cmap('jet'),
                        norm=plt.Normalize(sp_amp1.min(), sp_amp1.max()))
    lc.set_array(sp_amp1)

    ax.add_collection(lc)

    plt.ylim(sp_amp1.min(), sp_amp1.max())
    plt.xlabel('$t~\\mathrm{[Gyr]}$')
    plt.ylabel('$A_2$')
    plt.xlim(sp_time1[0],sp_time1[-1])
    plt.ylim(sp_amp1.min(),sp_amp1.max())
    plt.annotate('spiral amplitude', (6.1,0.04), fontsize=15)

    ax = plt.subplot(3,3,8,aspect='auto')
    points = np.array([sp_time2,sp_amp2]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=plt.get_cmap('jet'),
                        norm=plt.Normalize(sp_amp2.min(), sp_amp2.max()))
    lc.set_array(sp_amp2)

    ax.add_collection(lc)
    plt.ylim(sp_amp2.min(), sp_amp2.max())
    plt.xlabel('$t~\\mathrm{[Gyr]}$')
    plt.ylabel('$A_2$')
    plt.xlim(sp_time2[0],sp_time2[-1])
    plt.ylim(sp_amp2.min(),sp_amp2.max())
    plt.annotate('spiral amplitude', (6.1,0.04), fontsize=15)

    ax = plt.subplot(3,3,9, aspect = 'auto')
    make_jacobi_figure(x,y,z,vx,vy,vz,phi,t,omega1,omega2)

    plt.subplots_adjust(hspace=.3,wspace=.3)

    

def make_redist_plots(x,y,t) : 

    from matplotlib.collections import LineCollection
    from matplotlib.backends.backend_pdf import PdfPages
    import spiral_structure as ss
    import orbits

    plt.rc('font', size=15)
    fig = plt.figure(figsize=(13.5,11))
    r = np.sqrt(x**2 + y**2)
    for j in range(len(x)) : 
        ax = plt.subplot(3,4,j*2+1,frameon=False)
        pind = j
       # good = np.where(r[pind] > 0)[0]
        points = np.array([x[pind], y[pind]]).T.reshape(-1, 1, 2)
        plt.xlim(-13,13)
        plt.ylim(-13,13)
        ax.set_aspect('equal')
#        plt.xlabel('$x~\\mathrm{[kpc]}$')
#        plt.ylabel('$y~\\mathrm{[kpc]}$')
       
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=plt.get_cmap('jet'),
                            norm=plt.Normalize(t.min(), t.max()))
        lc.set_array(t)

        ax.add_collection(lc)

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

        ax = plt.subplot(3,4,j*2+2)
        points = np.array([t, r[pind]]).T.reshape(-1, 1, 2)
        plt.xlim(0,10)
        plt.ylim(0,13)
        plt.xlabel('$t~\\mathrm{[Gyr]}$')
        plt.ylabel('$R~\\mathrm{[kpc]}$')
        

        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=plt.get_cmap('jet'),
                            norm=plt.Normalize(t.min(), t.max()))
        lc.set_array(t)

        ax.add_collection(lc)

def toomre_q_fig():
    from matplotlib.font_manager import FontProperties
    fontsize = 20

    flist = ['2/12M_hr.00200', '5/12M_hr.00500', '8/12M_hr.00800', 
             '10/12M_hr.01000']

    labels = ['2 Gyr', '5 Gyr', '8 Gyr', '10 Gyr']
    lines = ['-','--','-.',':']
    plt.figure(figsize=(8,12))
    ps = []
    for i,f in enumerate(flist) :
        s = pynbody.load(f)
        pynbody.analysis.angmom.faceon(s)
        ps.append(pynbody.analysis.profile.Profile(s.s,max=10))
    
    ax = plt.subplot(2,1,1)

    for i,p in enumerate(ps) :
        plt.plot(p['rbins'],p['Q'],'k'+lines[i],label=labels[i],linewidth=2)

        if i == len(flist)-1:
            ax.set_ylabel('$Q$',fontsize=fontsize,fontweight='bold')
            ax.set_ylim(0,10)
            ax.xaxis.set_ticklabels("")
    plt.legend(loc='upper left',prop=FontProperties(size='small'))
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')

    ax = plt.subplot(2,1,2)
    ax2 = ax.twinx()
    
    for i,p in enumerate(ps) :
        ax.plot(p['rbins'],p['density'].in_units('Msol kpc^-2'),'k'+lines[i],linewidth=2)
        ax2.plot(p['rbins'],p['vr_disp'],'r'+lines[i],linewidth=2)

        if i == len(flist)-1:
            ax.set_ylabel('$\Sigma_{\star}$ [M$_{\odot}$ kpc$^{-2}$]',fontsize=fontsize,fontweight='bold')
            ax.set_xlabel('$R$ [kpc]',fontsize=fontsize,fontweight='bold')
            ax.semilogy()
            ax2.set_ylabel('$\sigma_{r}$ [km/s]',fontsize=fontsize,fontweight='bold')
    
    for ax in [ax,ax2]:
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            tick.label1.set_fontweight('bold')

@interruptible
def load_single_file(filename):
     return pynbody.load(filename[0])


def make_jacobi_figure(x=None,y=None,z=None,vx=None,vy=None,vz=None,phi=None,t=None,omega1=65,omega2=39):
    import spiral_structure as ss
    from matplotlib.font_manager import FontProperties

    # these are the indices of particles from Fig. 7

    if x is None: 
        spec_inds = [640778,1052173,1312193,632547,691094,945673]
        spec_inds.sort()

        x,y,z,xr,yr,vx,vy,vz,mass,phi,t = horseshoe_orbits(20)
        x=x[[1,4]]
        y=y[[1,4]]
        z=z[[1,4]]
        vx=vx[[1,4]]
        vy=vy[[1,4]]
        vz=vz[[1,4]]
        phi=phi[[1,4]]
        

    E1, L1, J1 = ss.get_jacobi_integral(x,y,z,vx,vy,vz,phi,t,omega1)
    E2, L2, J2 = ss.get_jacobi_integral(x,y,z,vx,vy,vz,phi,t,omega2)
    
#    J1 /= 1e5
#    J2 /= 1e5

    ax = plt.gca().add_subplot(121)

    plt.plot(t, J1[0]/J1[0][0], 'g-', label = '$\Omega_p$ = %d km/s/kpc'%(int(omega1)))
    plt.plot(t, J1[1]/J1[1][0], 'g--')
    plt.plot(t, J2[0]/J2[0][-1], 'b-', label = '$\Omega_p$ = %d km/s/kpc'%(int(omega2)))
    plt.plot(t, J2[1]/J2[1][-1], 'b--')
    
        
    
    plt.xlabel('$t~\mathrm{[Gyr]}$')
    plt.ylabel('$E_j/E_{j,0}$')

    plt.legend(loc='upper center',prop=FontProperties(size=11),frameon=False)
    
#    ax2 = plt.gca().twinx()
    
    ax2 = plt.gca().add_subplot(122)

    plt.plot(t, E1[0]/E1[0][0], 'r-', label = '$E$')
    plt.plot(t, E1[1]/E1[1][0], 'r--')
    #plt.plot(t, L1[0]/L1[0][0], 'k-', label = '$L$')
    #plt.plot(t, L1[1]/L1[1][0], 'k--')
    
    ax2.set_ylabel('$E/E_0$')

    plt.xlim(t[0],t[-1])


        
    



    

    
