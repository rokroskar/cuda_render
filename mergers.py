"""

a set of routines for analysing merger simulations


Rok Roskar
University of Zurich


"""

import pynbody
import numpy as np
import isolated as iso
import parallel_util
from pynbody.array import SimArray
import matplotlib.pylab as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.font_manager import FontProperties
import glob

def satellite_orbit(parent_dir='./') :
    import glob
    import sys
    import parallel_util

    filelist = glob.glob(parent_dir+'/[5,6,7,8]/*.00???')
    filelist.sort()

    s = pynbody.load(filelist[0])

    inds = np.where(s.s['mass']<.1)

    t = SimArray(np.zeros(len(filelist)), 'Gyr')
    cen = SimArray(np.zeros((len(filelist),3)), s['x'].units)

    res = parallel_util.run_parallel(single_center,filelist,[inds],processes=30)

    for i in range(len(res)) : t[i], cen[i] = res[i]

    plt.figure()
    r = np.sqrt((cen**2).sum(axis=1))
    plt.plot(t, r)
    plt.xlabel(r'$t$ [Gyr]')
    plt.ylabel(r'$r$ [kpc]')

    return t, cen


@parallel_util.interruptible
def single_center(a) : 
    filename, ind = a
    
    s = pynbody.load(filename)
    pynbody.analysis.halo.center(s,mode='pot')
    
    ssub = s.s[ind]
        
        #cen = pynbody.analysis.halo.center(ssub,mode='ssc',retcen=True)
        
    return s.properties['time'].in_units('Gyr'), np.array(pynbody.analysis.halo.center(ssub,mode='ssc',retcen=True))

    

    

def combine_starlog(sim1, sim2, starlog1, starlog2, tcut, mcut) : 
    """
    
    Take two starlogs and combine -- this is done when 
    *sim1* is run from a checkpoint of *sim2*. 

    Makes a copy of the *sim1* starlog and writes a new starlog 
    to include data for particles that existed before *sim1* was
    started and iords were not preserved.

    """
    
    from pynbody.tipsy import StarLog
    import os, glob
    
    # make a copy of the existing starlog file

#    os.system("cp " + starlog1 + " " + starlog1 + "_backup")

    sl1 = StarLog(starlog1)
    sl2 = StarLog(starlog2)

    sl_new = StarLog(starlog1)

    # reset the snapshot and resize

    for key in sl_new.keys() : 
        del(sl_new[key])

    sl_new._num_particles = len(sim1.s)
    print len(sl_new)
    sl_new._family_slice = {pynbody.family.star: slice(0,len(sl1.s))}

    sl_new._create_arrays(['tform','massform','rhoform','tempform'],dtype='float64')
    sl_new._create_arrays(['pos','vel'],3,dtype='float64')
    sl_new._create_arrays(['iord', 'iorderGas'], dtype='int32')

    old = (np.where((sim1.s['mass'] >= mcut)&(sim1.s['tform'] <= tcut))[0])
    old2 = (np.where(sl2['tform'] <= tcut))[0]

    assert((old == old2).all())

    new = (np.where(sim1.s['tform'] > tcut))[0]

    assert(len(new) == len(sl1))

    for arr in ['pos','vel','tform','massform','rhoform','tempform','iord','iorderGas'] : 
        sl_new[arr][old] = np.array(sl2[arr][old],dtype=sl_new[arr].dtype)
        sl_new[arr][new] = np.array(sl1[arr],sl_new[arr].dtype)
        

    sl_new['iord'][old] = sim1.s['iord'][old]
    sl_new['iord'][new] = sim1.s['iord'][new]

    sl_new.write(filename=starlog1)
    print len(sl_new)


###########################
# Routines for making plots
###########################


def make_image_figure(si,sm) : 

    
    fig = plt.figure(figsize=(14,10))

    grid = ImageGrid(fig,111,(2,3), label_mode="L", axes_pad=.2, direction='column')

    inds_old = np.where(sm.s['mass'] > .1)[0]
    inds_sat = np.where(sm.s['mass'] < .1)[0]


    vmin, vmax = 6, 12
    
    for s in [si,sm] : pynbody.analysis.angmom.faceon(s)

    # isolated
    # faceon
    ax = grid[0]
    pynbody.plot.image(si.s,width=29,units='Msol kpc^-2',threaded=True,
                       subplot=ax,colorbar=False,vmin=vmin,vmax=vmax)
    ax.set_title('isolated')

    # edgeon
    ax = grid[1]
    si.rotate_x(90)
    pynbody.plot.image(si.s,width=29,units='Msol kpc^-2',threaded=True,subplot=ax,colorbar=False,
                       vmin=vmin,vmax=vmax)
    ax.set_ylabel('$z/\\mathrm{kpc}$')
    
    
    # merger run -- in-situ stars
    #faceon
    ax = grid[2]
    pynbody.analysis.angmom.faceon(sm)
    pynbody.plot.image(sm.s[inds_old],width=29,units='Msol kpc^-2',threaded=True,subplot=ax,colorbar=False,
                       vmin=vmin,vmax=vmax)
    ax.set_title('merger in situ')

    # merger run -- satellite stars 
    # faceon
    ax = grid[4]
    pynbody.plot.image(sm.s[inds_sat],width=29,units='Msol kpc^-2',threaded=True,subplot=ax,colorbar=False)
                       
    ax.set_title('merger satellite')
    
    # merger run -- in-situ stars
    # edgeon
    sm.rotate_x(90)
    ax = grid[3]
    pynbody.plot.image(sm.s[inds_old],width=29,units='Msol kpc^-2',threaded=True, subplot=ax,colorbar=False,
                       vmin=vmin,vmax=vmax)

    # merger run -- satellite stars
    # edgeon
    ax = grid[5]
    pynbody.plot.image(sm.s[inds_sat],width=29,units='Msol kpc^-2',threaded=True, subplot=ax,colorbar=False)
    
    # reset to faceon
    pynbody.analysis.angmom.faceon(si)
    pynbody.analysis.angmom.faceon(sm)


def make_radial_profile(si,sm) : 

    fig = plt.figure(figsize=(8,10))

    #grid = ImageGrid(fig,111,(2,1), label_mode="L", axes_pad=.2, direction='column')

    inds_old = np.where(sm.s['mass'] > .1)[0]
    inds_sat = np.where(sm.s['mass'] < .1)[0]

    
    # make the profile objects
    pynbody.analysis.angmom.faceon(si)
    pynbody.analysis.angmom.faceon(sm)
    pi = pynbody.analysis.profile.Profile(si.s,max=15,nbins=30,ndim=2)
    pm_old = pynbody.analysis.profile.Profile(sm.s[inds_old],max=15,nbins=30,ndim=2)
    pm_sat = pynbody.analysis.profile.Profile(sm.s[inds_sat],max=15,nbins=30,ndim=2)

    #params = {'font.family': 'serif',
    #          'font.serif': ['Times','Utopia']}
    #plt.rcParams.update(params)

    # plot the density profiles
    ax = plt.subplot(311)

    plt.plot(pi['rbins'],pi['density'].in_units('Msol kpc^-2'),label='isolated')
    plt.ylabel('$\Sigma_{\star}~\mathrm{[M_{\odot}~kpc^{-2}]}$')
    
    plt.plot(pm_old['rbins'],pm_old['density'].in_units('Msol kpc^-2'),label='merger in-situ')
    plt.plot(pm_sat['rbins'],pm_sat['density'].in_units('Msol kpc^-2'),label='merger satellite')
    plt.semilogy()
    plt.xlim(0,15)
    plt.legend(prop=FontProperties(size='small'))
    ax.set_xticklabels("")

    # plot the age profiles
    ax = plt.subplot(312)
    
    plt.plot(pi['rbins'],pi['age'].in_units('Gyr'),label='isolated')
    plt.ylabel('Age [Gyr]')
    plt.xlim(0,15)
    plt.plot(pm_old['rbins'],pm_old['age'].in_units('Gyr'),label='merger in-situ')
    ax.set_xticklabels("")
    fig.subplots_adjust(hspace=0.05)
    


    # plot the dispersion profiles
    ax = plt.subplot(313)
    plt.plot(pi['rbins'],pi['vr_disp'],'b-')
    plt.plot(pi['rbins'],pi['vt_disp'],'b--')
    plt.plot(pi['rbins'],pi['vz_disp'],'b-.')

    plt.plot(pm_old['rbins'],pm_old['vr_disp'],'g-')
    plt.plot(pm_old['rbins'],pm_old['vt_disp'],'g--')
    plt.plot(pm_old['rbins'],pm_old['vz_disp'],'g-.')

    plt.ylabel(r'$\sigma_{r,\phi,z}$')
    plt.xlabel('$R$ [kpc]')

def make_vertical_profiles(si,sm) : 

    fig = plt.figure(figsize=(8,10))

    #grid = ImageGrid(fig,111,(2,1), label_mode="L", axes_pad=.2, direction='column')

    inds_old = np.where(sm.s['mass'] > .1)[0]
    inds_sat = np.where(sm.s['mass'] < .1)[0]

    pynbody.analysis.angmom.faceon(si)
    pynbody.analysis.angmom.faceon(sm)

    pi = pynbody.analysis.profile.VerticalProfile(si.s,7,8,3.0,nbins=30,ndim=3)
    pm_old = pynbody.analysis.profile.VerticalProfile(sm.s[inds_old],7,8,3.0,nbins=30,ndim=3)
    pm_sat = pynbody.analysis.profile.VerticalProfile(sm.s[inds_sat],7,8,3.0,nbins=30,ndim=3)

    ax = plt.subplot(211)

    plt.plot(pi['rbins'],pi['density']/pi['density'][0],label='isolated')
    plt.plot(pm_old['rbins'],pm_old['density']/pm_old['density'][0],label='merger in-situ')
    plt.plot(pm_sat['rbins'],pm_sat['density']/pm_old['density'][0],label='merger satellite')
    plt.legend(prop=FontProperties(size='small'))
    ax.set_xticklabels("")
    ax.set_ylabel(r'$\rho_{\star}/\rho_0$')
    ax.semilogy()
    ax = plt.subplot(212)
    
    plt.plot(pi['rbins'],pi['vz_disp'].in_units('km s^-1'),label='isolated')
    plt.plot(pm_old['rbins'],pm_old['vz_disp'].in_units('km s^-1'),label='merger in-situ')
    plt.plot(pm_sat['rbins'],pm_sat['vz_disp'].in_units('km s^-1'),label='merger satellite')
    
    ax.set_xlabel(r'$z$ [kpc]')
    ax.set_ylabel(r'$\sigma_{v_z}$ [km/s]')
    
def make_fourier_comparison(dir1,dir2) : 

    data1 = np.load(dir1+'/complete_fourier_fulldisk.npz')
    data2 = np.load(dir2+'/complete_fourier_fulldisk.npz')

    plt.figure()
    
    plt.plot(data1['t'],abs(data1['c'][:,2,0]),label='isolated')
    plt.plot(data2['t'][0::10],abs(data2['c'][0::10][:,2,0]),label='merger')
    plt.xlabel(r'$t$ [Gyr]')
    plt.ylabel(r'$A_2$')
    plt.legend(loc='upper left')
    
    
def angular_momentum_vector(dir1) : 
    
    flist = glob.glob(dir1+'[1-9]/*.0??01')
    flist.sort()
    flist.append(glob.glob(dir1+'10/*.0??00')[0])

    jvec = np.zeros((len(flist),3))

    for i,filename in enumerate(flist) : 
        s = pynbody.load(filename)
        pynbody.analysis.halo.center(s)

        f = pynbody.filt.Disc(4.0,.2) #& pynbody.filt.LowPass('temp',5e4)

        jvec[i] = pynbody.analysis.angmom.ang_mom_vec(s.g[f])

    return jvec, np.arcsin(jvec[:,2]/np.sqrt((jvec**2).sum(axis=1)))*180./np.pi


def plot_compare_rform(outs, labels=None): 
    import matplotlib.pylab as plt
    import os

    sims = []

#    sn = pynbody.filt.SolarNeighborhood(7,8,.5)
    sn = pynbody.filt.Annulus(7,8)
    main = pynbody.filt.HighPass('mass', 0.1)

    plt.figure(figsize=(8,10))
    
    for i,s in enumerate(outs) : 
        #s = pynbody.load(out)
        
        pynbody.analysis.halo.center(s)
        pynbody.analysis.angmom.faceon(s)
        
        iso.get_rform(s.s)
        
        if labels is not None:
            label = labels[i]
        else : 
            label = ''
        print 'total stars'
        print len(s.s[sn&main])
        
        ax = plt.subplot(211)
        plt.hist(s.s[sn&main]['rform'], bins = 50, range=[0,15], label=label,normed=True,histtype='step')
        plt.xlim(0,12)
        ax.set_xticklabels("")
        plt.subplot(212)
        plt.hist(s.s[sn&main]['rform'], bins = 50, range=[0,15], label=label,normed=True,histtype='step',
                 cumulative=True)
        plt.xlim(0,12)

    plt.xlabel(r'$R_{form}$ [kpc]')
 
    plt.legend(loc = 'upper left', prop=FontProperties(size='small'))


def make_rform_rfinal_figs(s1,s2,vmin=1e-3,vmax=.1) : 

    plt.figure(figsize=(15,4))

    ax = plt.subplot(131)

    g1, x1, y1 = pynbody.plot.generic.prob_plot(s1.s['rxy'],s1.s['rform'],s1.s['mass'],extent=(0,15,0,15), axes = ax,vmin=vmin,vmax=vmax,interpolation='nearest')
    ax.set_ylabel(r'$R_{form}$ [kpc]')
    ax.set_xlabel(r'$R_{final}$ [kpc]')
    ax.set_title('isolated')
    ax.plot([0,15],[0,15],'y--',linewidth=2)
    ax.set_xlim(0,15)
    ax.set_ylim(0,15)

    ax = plt.subplot(132)
    g2, x2, y2 = pynbody.plot.generic.prob_plot(s2.s['rxy'],s2.s['rform'],s2.s['mass'],extent=(0,15,0,15), axes = ax, vmin=vmin,vmax=vmax,interpolation='nearest')
    ax.set_xlabel(r'$R_{final}$ [kpc]')
    ax.set_title('merger')
    ax.set_yticklabels("")
    ax.plot([0,15],[0,15],'y--',linewidth=2)
    ax.set_xlim(0,15)
    ax.set_ylim(0,15)


    ax = plt.subplot(133)
    
    im = ax.imshow(np.log10(g1/g2), extent=[0,15,0,15], origin='lower',interpolation='nearest')
    cb = plt.colorbar(im).set_label('$log_{10}(P_i/P_m)$')
    ax.set_title(r'$P_{isolated}/P_{merger}$')
    ax.set_xlabel(r'$R_{final}$ [kpc]')
    ax.set_yticklabels("")
    ax.plot([0,15],[0,15],'y--',linewidth=2)
    ax.set_xlim(0,15)
    ax.set_ylim(0,15)


def merger_profile_evolution(list1,list2) : 

    from pynbody.analysis.profile import Profile, VerticalProfile
    
    fig_r = plt.figure(figsize=(12,12))
    fig_z = plt.figure(figsize=(12,12))

    assert(len(list1)==len(list2))

    
    for i in xrange(len(list1)) :

        s1 = pynbody.load(list1[i])
        s2 = pynbody.load(list2[i])

        for s in [s1,s2]: pynbody.analysis.angmom.faceon(s)
        
        ind = np.where(s2.s['mass'] > 0.1)[0]

        ax_r = fig_r.add_subplot(4,5,i+1)
        ax_z = fig_z.add_subplot(4,5,i+1)

        p1_r = Profile(s1.s,max=15)
        p1_z = VerticalProfile(s1.s,2.5,3.5,3,nbins=30)
        p2_r = Profile(s2.s[ind],max=15)
        p2_z = VerticalProfile(s2.s[ind],2.5,3.5,3,nbins=30)
        

        ax_r.plot(p1_r['rbins'],p1_r['density'].in_units('Msol kpc^-2'))
        ax_r.plot(p2_r['rbins'],p2_r['density'].in_units('Msol kpc^-2'))

        ax_z.plot(p1_z['rbins'],p1_z['density']/p1_z['density'][0])
        ax_z.plot(p2_z['rbins'],p2_z['density']/p2_z['density'][0])
        
        ax_r.semilogy()
        ax_r.set_ylim(1e3,1e10)
        ax_z.semilogy()
        ax_z.set_ylim(1e-4, 1.0)

        label = '$t = $ %.2f Gyr'%s2.properties['a']
        ax_r.annotate(label,(6,1e9))
        ax_z.annotate(label,(1.,.5))

        if i%5 == 0: 
            ax_r.set_ylabel(r'$\Sigma_{\star}$ [M$_{\odot}$ kpc$^{-2}$]')
            ax_z.set_ylabel(r'$\rho/\rho_0$')
        else : 
            for ax in [ax_r,ax_z] : ax.set_yticklabels("")
        if i >= 3*5 :
            ax_r.set_xlabel(r'$R$ [kpc]')
            ax_z.set_xlabel(r'$z$ [kpc]')
        else :
            for ax in [ax_r,ax_z] : ax.set_xticklabels("")



def plot_structural_parameters(flist1,flist2) : 
    
    fits1 = iso.disk_structure_evolution(flist1)
    fits2 = iso.disk_structure_evolution(flist2,merger=True)
    
    fig = plt.figure(figsize=(6,10))
    

    # radial scale length
    ax = fig.add_subplot(211)
    
    ax.plot(fits1[0],fits1[1][:,1],label='isolated')
    ax.plot(fits2[0],fits2[1][:,1],label='merger')
    ax.set_xticklabels("")
    plt.legend(loc='upper left', prop=FontProperties(size='small'))
    ax.set_ylabel(r'$h_R$ [kpc]')
    
    
    # vertical scale height

    ax = fig.add_subplot(212)

    ax.plot(fits1[0],fits1[3][:,1], 'b-', fits1[0],fits1[3][:,3], 'b--')
    ax.plot(fits2[0],fits2[3][:,1], 'g-', fits2[0],fits2[3][:,3], 'g--')

    ax.set_ylabel(r'$h_z$ [kpc]')
    ax.set_xlabel(r'$t$ [Gyr]')

    fig.subplots_adjust(hspace=.1)
    
    return fits1, fits2
