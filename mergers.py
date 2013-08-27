"""

a set of routines for the paper on migration with mergers


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

runlist_lmc = ['merge2Gyr_lmc_low',
               'merge2Gyr_lmc_high',
               'merge5Gyr_lmc_low',
               'merge5Gyr_lmc_high']

runlist_llmc = ['merge2Gyr_llmc_low',
               'merge2Gyr_llmc_high',
               'merge5Gyr_llmc_low',
               'merge5Gyr_llmc_high']

runlist_2Gyr = ['merge2Gyr_lmc_low',
                'merge2Gyr_lmc_high',
                'merge2Gyr_llmc_low',
                'merge2Gyr_llmc_high']

runlist_5Gyr = ['merge5Gyr_lmc_low',
                'merge5Gyr_lmc_high',
                'merge5Gyr_llmc_low',
                'merge5Gyr_llmc_high']

runlist_all = runlist_lmc + runlist_llmc

def get_color(i,n,cmap = plt.cm.rainbow):
    return cmap(int(i*256./n))

colors_all = []
for i in range(len(runlist_all)) :
    colors_all.append(get_color(i,len(runlist_all)))

lmc_titles = []
for run in runlist_lmc : 
    lmc_titles.append(run.replace('_','-')[5:])

llmc_titles = []
for run in runlist_llmc : 
    llmc_titles.append(run.replace('_','-')[5:])

titles_all = lmc_titles + llmc_titles
 
def load_runs(runlist=runlist_all) : 
    slist = []
    for i,f in enumerate(runlist) : 
        slist.append(pynbody.load(f+'/10/'+f+'.01000'))
        pynbody.analysis.angmom.faceon(slist[i])
    return slist

def satellite_orbit(parent_dir='./',plot=False, save=False) :
    import glob
    import sys
    import parallel_util

    filelist = glob.glob(parent_dir+'/[2,3,4,5,6,7,8]/*.00???.gz')
    filelist.sort()

    s = pynbody.load(filelist[0])

    inds = np.where(s.s['mass']<.1)

    t = SimArray(np.zeros(len(filelist)), 'Gyr')
    cen = SimArray(np.zeros((len(filelist),3)), s['x'].units)

    res = parallel_util.run_parallel(single_center,filelist,[inds],processes=20)

    for i in range(len(res)) : t[i], cen[i] = res[i]
    
    if plot:
        plt.figure()
        r = np.sqrt((cen**2).sum(axis=1))
        plt.plot(t, r)
        plt.xlabel(r'$t$ [Gyr]')
        plt.ylabel(r'$r$ [kpc]')

    if save : 
        np.savez(parent_dir+'/sat_orbit.npz',t=t,cen=cen)

    return t, cen




@parallel_util.interruptible
def single_center(a) : 
    filename, ind = a
    
    s = pynbody.load(filename)
    pynbody.analysis.halo.center(s,mode='pot')
    
    ssub = s.s[ind]
        
        #cen = pynbody.analysis.halo.center(ssub,mode='ssc',retcen=True)
        
    return s.properties['time'].in_units('Gyr'), np.array(pynbody.analysis.halo.center(ssub,mode='ssc',retcen=True))

    

def get_tform(s,si,mcut,tcut) :
    starlog = s.filename.split('/')[0]+'/'+s.filename.split('/')[0]+'.starlog'
    sat = np.where((s.s['mass'] <= mcut))[0]
    old = np.where(si.s['tform'] <= tcut)[0]
    sl = pynbody.tipsy.StarLog(starlog)
    s.s['tform'][sat[-1]+1:] = sl['tform']
    s.s['tform'][0:sat[0]] = si.s['tform'][old]
    


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
    sl_new._family_slice = {pynbody.family.star: slice(0,len(sim1.s))}

    sl_new._create_arrays(['tform','massform','rhoform','tempform'],dtype='float64')
    sl_new._create_arrays(['pos','vel'],3,dtype='float64')
    sl_new._create_arrays(['iord', 'iorderGas'], dtype='int32')

    old = (np.where((sim1.s['mass'] >= mcut)&(sim1.s['tform'] <= tcut))[0])
    old2 = (np.where(sim2.s['tform'] <= tcut))[0]
    sat = (np.where(sim1.s['mass'] < mcut)[0])

    assert((old == old2).all())

    new = (np.where(sim1.s['tform'] > tcut))[0]

    assert(len(new) == len(sl1))

    for arr in ['pos','vel','tform','massform','rhoform','tempform','iord','iorderGas'] : 
        sl_new[arr][old] = np.array(sl2[arr][old],dtype=sl_new[arr].dtype)
        sl_new[arr][new] = np.array(sl1[arr],sl_new[arr].dtype)
        if 'iord' not in arr : sl_new[arr][sat] = np.nan

    sl_new['iord'] = sim1.s['iord']
    #sl_new['iord'][new] = sim1.s['iord'][new]
    #sl_new['iord']
#    import pdb; pdb.set_trace()
    sl_new.write(filename=starlog1)
    print len(sl_new)


def get_rform(si,so,tcut,nbins=200,mcut=.05,plots=False):
    from matplotlib.backends.backend_pdf import PdfPages
    s = so.s[pynbody.filt.HighPass('mass',mcut)]
    old = pynbody.filt.LowPass('tform',tcut)
    cenfilt = pynbody.filt.LowPass('r','5 kpc')

    if plots:
        pp = PdfPages('alignments.pdf')
        plt.ioff()
    bin_edges = np.linspace(np.min(s.s['tform']),
                            np.max(s.s['tform']),nbins+1)
    bins = .5*(bin_edges[1:]+bin_edges[:-1])
    cens  = np.zeros((nbins,3))*np.nan
    vcens = np.zeros((nbins,3))*np.nan
    js = np.zeros((nbins,3))*np.nan

    
    news = pynbody.new(stars=len(s.s))
    news.s['mass'] = s.s['massform']
    for i, arr in enumerate(['x','y','z']): 
        news.s[arr] = s.s['posform'][:,i]
        news.s['v'+arr] = s.s['velform'][:,i]
    news.s['tform']=s.s['tform']
    news.properties={'z':0,'a':1.0}

    # get center of mass for each bin using shrinking spheres
    print "Finding center positions"
    for ibe,below in enumerate(bin_edges[:-1]):
        tstars = news.s[pynbody.filt.BandPass('tform',below,bin_edges[ibe+1])]
        if len(tstars) > 0:
            cens[ibe]=pynbody.analysis.halo.center(tstars,mode='ssc',retcen=True)
        print below, cens[ibe]
#        import pdb; pdb.set_trace()

    print "Setting positions according to new centers"
    for i, arr in enumerate(['x','y','z']) : 
        news.s[arr] -= np.interp(news.s['tform'],bins,cens[:,i])

#    plt.plot(news.s['tform'].in_units('Gyr'),np.interp(news.s['tform'].in_units('Gyr'),bins,cens[:,0]),'.')
#    plt.plot(bins,cens[:,0])

    #import pdb; pdb.set_trace()

    print "Finding centers of mass velocities"
    for ibe,below in enumerate(bin_edges[:-1]):
        tstars = news.s[pynbody.filt.BandPass('tform',str(below)+' Gyr',str(bin_edges[ibe+1])+' Gyr')]
        if len(tstars[cenfilt]) > 5:
            vcens[ibe]=pynbody.analysis.halo.vel_center(tstars[cenfilt],retcen=True,cen_size='5 kpc')
        #else:
            #import pdb; pdb.set_trace()

    goodvcen = np.isfinite(vcens).all(axis=1)
    print "Setting velocities"
    for i, arr in enumerate(['x','y','z']) : 
        news.s['v'+arr] -= np.interp(news.s['tform'],bins[goodvcen],vcens[goodvcen,i])

    r_array = news.s['r']
    print "Rotating angular momenta"
    for ibe,below in enumerate(bin_edges[:-1]):
        tstars = news.s[pynbody.filt.BandPass('tform',below,bin_edges[ibe+1])]
        if ((len(tstars[cenfilt]) > 5) & np.isfinite(vcens[ibe]).all()):
            js[ibe] = pynbody.analysis.angmom.ang_mom_vec(tstars[cenfilt])
            trans = pynbody.analysis.angmom.calc_faceon_matrix(js[ibe])

            jstars = news.s[pynbody.filt.BandPass('tform',below,bin_edges[ibe+1])]
            jstars.transform(trans)
            if plots:
                f,ax = plt.subplots(1,3)
                ax[0].plot(jstars['x'].in_units('kpc'),jstars['y'].in_units('kpc'),'.')
                ax[1].plot(jstars['x'].in_units('kpc'),jstars['z'].in_units('kpc'),'.')
                ax[2].plot(jstars['y'].in_units('kpc'),jstars['z'].in_units('kpc'),'.')
                for a in ax:
                    a.set_aspect('equal')
                    a.set_xlim(-25,25)
                    a.set_ylim(-25,25)
                pp.savefig()
                plt.clf()
                plt.close()

    s.s['rxyform'] = news.s['rxy']
    s.s[old]['rxyform'] = si.s[old]['rform']
    s.s['rxyform'].write(overwrite=True)
    
    if plots:
        pp.close()    
 #   news.write(filename='new.starlog.tipsy',fmt=pynbody.tipsy.TipsySnap)
    import pickle
    pickle.dump({'cen':cens,'j':js, 't':bins},open(so.filename+'.jcen','w'))

def two_sech2(xs,scale1=1.0,scale2=2.0,f=0.5) : 
    return (1.-f)*np.cosh(xs/scale1)**-2+f*np.cosh(xs/scale2)**-2

def get_hz_vs_r(s,rs,write=True) : 
    import diskfitting
    from matplotlib.backends.backend_pdf import PdfPages

    up = s

    while hasattr(up,'base') : 
        up = up.base
    filename = up.filename

    pp = PdfPages(filename+'.scaleheights.pdf')
    plt.ioff()

    f = plt.figure()

    h1 = np.zeros(len(rs))
    h2 = np.zeros(len(rs))
    frac = np.zeros(len(rs))
    n = np.zeros(len(rs))
    errors = np.zeros(len(rs))
        
    print 'getting scale heights for %s'%s.filename 
    for i, r in enumerate(rs) : 
        sn = pynbody.filt.SolarNeighborhood(r-.5,r+.5,5)
        zrms = np.sqrt((s.s[sn]['z']**2).sum()/len(s.s[sn]))
        fit,num = diskfitting.two_comp_zfit_simple(s.s[sn]['z'],0.,zrms*5.0)
        print zrms
        h1[i] = fit[:2].min()
        h2[i] = fit[:2].max()
        frac[i] = fit[2]
        n[i] = num
        print r, h1[i], h2[i], frac[i]
        
        p = pynbody.analysis.profile.VerticalProfile(s.s[sn],r-.5,r+.5,5,nbins=20)
        plt.errorbar(p['rbins'],p['density']/p['density'][0],fmt='.',yerr=p['density']/p['density'][0]/np.sqrt(p['n']),label='R = %d kpc'%r)
        plt.plot(p['rbins'],two_sech2(p['rbins'],fit[0],fit[1],fit[2]),'--')
        plt.semilogy()
        pp.savefig()
        plt.clf()
        
        
    h1 /= 2.0
    h2 /= 2.0
    pp.close()
    if write : 
        np.savez(s.filename+'.scaleheights.npz', h1=h1,h2=h2,frac=frac,n=n,rs=rs)

    return h1,h2,frac,n

###########################
# Routines for making plots
###########################

def make_orbits_figure() :
    f,axs = plt.subplots(2,1,figsize=(10,14))
    
    for i,run in enumerate(runlist_all) :
        dat = np.load(run+'/sat_orbit.npz')
        
        if i < 4: row = 0
        else : row = 1
        axs[row].plot(dat['t'],np.sqrt((dat['cen']**2).sum(axis=1)),label=titles_all[i],color=get_color(i,len(runlist_all)))

    for ax in axs : 
        ax.legend(frameon=False,prop=dict(size=16))
        ax.set_xlabel('$r$ [kpc]')
        ax.set_ylabel('$t$ [Gyr]')

                    

def make_image_figure(si,sm_list, titles) : 

    ncol = 4
    nrow = len(sm_list)/2+1
    fig = plt.figure(figsize=(9.5,12))
    
    vmin, vmax = 4,9
    width=25
    axs = []

# make the isolated panels

    ax = fig.add_subplot(nrow,ncol,1)
    pynbody.plot.image(si.s,width=width,av_z=True,threaded=True,
                       subplot=ax,show_cbar=False,vmin=vmin,vmax=vmax,
                       qty='i_lum_den',cmap=plt.cm.Greys_r)
    ax.set_title('isolated',fontsize=14)
    axs.append(ax)
    
    ax = fig.add_subplot(nrow,ncol,2)
    si.rotate_x(90)
    pynbody.plot.image(si.s,width=width,av_z=True,threaded=True,subplot=ax,
                       show_cbar=False, vmin=vmin,vmax=vmax,
                       qty='i_lum_den', cmap=plt.cm.Greys_r)

    si.rotate_x(-90)
    axs.append(ax)

    for i,s in enumerate(sm_list) : 
        pynbody.analysis.angmom.faceon(s)

        inds_old = np.where(s.s['mass'] > .05)[0]
        inds_sat = np.where(s.s['mass'] < .05)[0]
        
        if len(inds_sat) > 0 :
            s.s[inds_sat]['tform'] = .01
            
        if i < 4 : 
            index = i*4+5
        else : 
            index = (i-4)*4+7

        ax = fig.add_subplot(nrow,ncol,index)
        pynbody.plot.image(s.s[inds_old],width=width,av_z=True,threaded=True,
                           subplot=ax,show_cbar=False,vmin=vmin,vmax=vmax,
                           qty='i_lum_den',cmap=plt.cm.Greys_r)
        ax.set_title(titles[i],fontsize=14)
        axs.append(ax)

        ax = fig.add_subplot(nrow,ncol,index+1)
        s.rotate_x(90)
        pynbody.plot.image(s.s[inds_old],width=width,av_z=True,threaded=True,subplot=ax,
                           show_cbar=False, vmin=vmin,vmax=vmax,
                           qty='i_lum_den', cmap=plt.cm.Greys_r)

        axs.append(ax)

        #if i > 0 : 
            # merger runs -- show satellite star distributions
            # faceon
        #    ax = fig.add_subplot(nrow,ncol,i*ncol+3)

#            s.rotate_x(-90)
        
#            pynbody.plot.image(s.s[inds_sat],width=29,av_z=True,threaded=True,
#                               subplot=ax,show_cbar=False, vmin=vmin,vmax=vmax,
#                               qty='i_lum_den',cmap=plt.cm.Greys_r)
#            if i == 1:
#                ax.set_title('accreted',fontsize=14)
#            axs.append(ax)

            # sideon
#            ax = fig.add_subplot(nrow,ncol,i*ncol+4)
#            s.rotate_x(90)
#            pynbody.plot.image(s.s[inds_sat],width=29,av_z=True,threaded=True,
#                               subplot=ax,show_cbar=False,vmin=vmin,vmax=vmax,
#                               qty='i_lum_den',cmap=plt.cm.Greys_r)
#            ax.set_ylabel('$z/\\mathrm{kpc}$',fontsize=14)
#            axs.append(ax)
        # reset to faceon
        s.rotate_x(-90)
    
    for i,ax in enumerate(axs):
        if i != len(axs)-10:
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_yticklabels('')
            ax.set_xticklabels('')
        else : 
            for tl in ax.get_yticklabels()+ax.get_xticklabels():
                tl.set_fontsize(14)

    plt.subplots_adjust(wspace=.01)

def make_profile_comparison_figure(si,slist) : 
    f,axs = plt.subplots(2,2,figsize=(10,10))

    rs = [8,12]

    p = pynbody.analysis.profile.Profile(si.s, min=.01,max=15,nbins=30)
    pv1 = pynbody.analysis.profile.VerticalProfile(si.s, rs[0]-.5,rs[0]+.5,3,nbins=20)
    pv2 = pynbody.analysis.profile.VerticalProfile(si.s, rs[1]-.5,rs[1]+.5,3,nbins=20)
    #pv3 = pynbody.analysis.profile.VerticalProfile(si.s, rs[2]-.5,rs[2]+.5,5,nbins=20)
                                         
    for i in range(2) : 
        axs[0,i].plot(p['rbins'],p['density'].in_units('Msol kpc^-2'),'k-',label='isolated')
        axs[1,i].plot(pv1['rbins'],pv1['density'].in_units('Msol kpc^-3'),'k-',label='%d kpc'%rs[0])
        axs[1,i].plot(pv2['rbins'],pv2['density'].in_units('Msol kpc^-3'),'k--',label='%d kpc'%rs[1])
        axs[0,i].set_xlabel('$R$ [kpc]')
        axs[1,i].set_xlabel('$z$ [kpc]')
        axs[0,i].set_ylim(1e5,1e11)
        axs[0,i].set_xlim(0,14.9)
        axs[1,i].set_ylim(1e4,1e9)
        axs[1,i].set_xlim(0,2.9)

    for j,s in enumerate(slist) : 
        color = plt.cm.rainbow(int(j*256.0/len(slist)))
#        nonsat = np.where(s.s['mass']>0.05)[0]
        ps   = pynbody.analysis.profile.Profile(s.s,min=.01,max=15,nbins=30)
        psv1 = pynbody.analysis.profile.VerticalProfile(s.s, rs[0]-.5,rs[0]+.5,3,nbins=20)
        psv2 = pynbody.analysis.profile.VerticalProfile(s.s, rs[1]-.5,rs[1]+.5,3,nbins=20)
        if j < 4 : col = 0
        else : col = 1
        axs[0,col].plot(ps['rbins'],ps['density'].in_units('Msol kpc^-2'),color=color,label=titles_all[j])
        axs[1,col].plot(psv1['rbins'],psv1['density'].in_units('Msol kpc^-3'),color=color)
        axs[1,col].plot(psv2['rbins'],psv2['density'].in_units('Msol kpc^-3'),color=color,linestyle='--')
    
    
    for ax in axs.flatten(): ax.semilogy()
    
    axs[0,0].set_ylabel(r'$\Sigma_{\star}$ [M$_{\odot}$ kpc$^{-2}$]')
    axs[1,0].set_ylabel(r'$\rho_{\star}$ [M$_{\odot}$ kpc$^{-3}$]')
    axs[0,1].set_yticklabels('')
    axs[1,1].set_yticklabels('')
    axs[0,0].legend(frameon=False,prop=dict(size=16))
    axs[0,1].legend(frameon=False,prop=dict(size=16))
    axs[1,0].legend(frameon=False,prop=dict(size=16))

    plt.subplots_adjust(wspace=.02)


def make_age_veldispersion_figure(si,slist) : 
    f,axs = plt.subplots(2,2,figsize=(10,10))
    p = pynbody.analysis.profile.Profile(si.s, min=.01,max=15,nbins=30)
                                            
    for i in range(2) : 
        axs[0,i].plot(p['rbins'],p['vr_disp'].in_units('km s^-1'),'k-',label='isolated')
        axs[0,i].plot(p['rbins'],p['vz_disp'].in_units('km s^-1'),'k--')
        axs[1,i].plot(p['rbins'],p['age'].in_units('Gyr'),'k-')
        axs[0,i].set_xlabel('$R$ [kpc]')
        axs[1,i].set_xlabel('$R$ [kpc]')

    for j,s in enumerate(slist) : 
        color = plt.cm.rainbow(int(j*256.0/len(slist)))
        nonsat = np.where(s.s['mass']>0.05)[0]
        ps   = pynbody.analysis.profile.Profile(s.s[nonsat],min=.01,max=15,nbins=30)
        if j < 4 : col = 0
        else : col = 1
        axs[0,col].plot(ps['rbins'],ps['vr_disp'].in_units('km s^-1'),color=color,label=titles_all[j])
        axs[0,col].plot(ps['rbins'],ps['vz_disp'].in_units('km s^-1'),color=color,linestyle='--')
        axs[1,col].plot(ps['rbins'],ps['age'].in_units('Gyr'),color=color)
    

    for i in range(2) : 
        axs[0,i].set_ylim(0,200)
        axs[1,i].set_ylim(0,10)
    for ax in axs.flatten() : ax.set_xlim(0,14.9)

    axs[0,0].set_ylabel(r'$\sigma_R, \sigma_z$ [km/s]')
    axs[1,0].set_ylabel(r'Age [Gyr]')
    axs[0,1].set_yticklabels('')
    axs[1,1].set_yticklabels('')
    axs[0,0].legend(frameon=False,prop=dict(size=16))
    axs[0,1].legend(frameon=False,prop=dict(size=16))

    plt.subplots_adjust(wspace=.02)


def make_rform_histograms(si,slist,titles) : 
    f,axs = plt.subplots(2,3,figsize=(20,10))
    rs = [4,8,12]

    if 'rxyform' not in si.s.keys() : 
        import isolated as iso
        iso.get_rform(si.s)
        si.s['rxyform'] = si.s['rform']

    for i in range(2) : 
        for j,r in enumerate(rs) : 
            filt = pynbody.filt.BandPass('rxy', r-.5,r+.5)
            axs[i,j].hist(si.s[filt]['rxyform'], range=[0,15],bins=50, normed=True,
                        histtype='step',color='k',label='isolated')
            
            if i == 1 : 
                axs[i,j].set_xlabel('$R_{form}$ [kpc]')
            else : 
                axs[i,j].set_xticklabels('')
                axs[i,j].set_title('$R_{final} = $ %d [kpc]'%rs[j])

            if j > 0 : 
                axs[i,j].set_yticklabels('')
            else : 
                axs[i,j].set_ylabel('$N$')

            axs[i,j].set_ylim(0,.35)
            axs[i,j].set_xlim(0,15)
            
    insitu = pynbody.filt.HighPass('mass', .05)

    for i,s in enumerate(slist) : 
        if i < 4 : ind=0
        else : ax = ind=1
           
        for j,r in enumerate(rs) : 
            ax = axs[ind,j]
            filt = pynbody.filt.BandPass('rxy', r-.5,r+.5)
            ax.hist(s.s[filt&insitu]['rxyform'], range=[0,15],bins=50, normed=True, 
                     color = get_color(i,len(slist)), histtype='step',label=titles[i])

            
    axs[0,0].legend(frameon=False,prop=dict(size=16))
    axs[1,0].legend(frameon=False,prop=dict(size=16))
    
    plt.subplots_adjust(wspace=.05,hspace=.05)

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








def make_hz_figure(si,slist,titles) : 
    import diskfitting
    
    f, axs = plt.subplots(2,1,figsize=(10,14))

    dat = np.load(si.filename+'.scaleheights.npz')
    
    markersize = 15
    alpha=.7
    for ax in axs: 
        ax.plot(dat['rs'],dat['h1'],'.',color='k',label='isolated',markersize=markersize,alpha=1)
        ax.plot(dat['rs'],dat['h2'],'+',color='k',markersize=markersize,alpha=1)


    for i,s in enumerate(slist) : 
        dat = np.load(s.filename+'.scaleheights.npz')
        color = get_color(i,len(slist))
        if i < 4 : ax = axs.flatten()[0]
        else : ax = axs.flatten()[1]
        ax.plot(dat['rs'],dat['h1'],'.',color=color,label=titles[i],markersize=markersize,alpha=alpha)
        ax.plot(dat['rs'],dat['h2'],'+',color=color,markersize=markersize,alpha=alpha)

    for ax in axs :
        ax.set_xlim(3,15)
        ax.set_ylim(0,.95)
        ax.legend(loc='upper left',frameon=False,prop=dict(size=16),scatterpoints=1)
        ax.set_ylabel('$h_z$ [kpc]')

    axs.flatten()[1].set_xlabel('$R$ [kpc]')
    axs.flatten()[0].set_xticklabels('')
    plt.subplots_adjust(hspace=.05)


def make_vert_sat_profiles(si,slist,titles) : 
    f,ax = plt.subplots(figsize=(10,8))


    sat = pynbody.filt.LowPass('mass',0.05)
    p = pynbody.analysis.profile.VerticalProfile(si.s,7,9,4,nbins=20)
    
    ax.plot(p['rbins'],p['density'].in_units('Msol kpc^-3'),'k--',label='isolated')

    for i,s in enumerate(slist) : 
        p = pynbody.analysis.profile.VerticalProfile(s.s[sat],7,9,4,nbins=20)
        ax.plot(p['rbins'],p['density'].in_units('Msol kpc^-3'),color = get_color(i,len(slist)),label=titles[i])

    ax.semilogy()
    ax.set_xlabel('$z$ [kpc]')
    ax.set_ylabel(r'$\rho$ [M$_{\odot}$ kpc$^{-3}$]')
    ax.legend(frameon=False,prop=dict(size=16))

def make_combined_frames(dirlist,prefix) : 
    import glob,utils
    
    fo = map(lambda x: np.sort(glob.glob(x+'/ppms/fo/*ppm')),dirlist)
    eo = map(lambda x: np.sort(glob.glob(x+'/ppms/eo/*ppm')),dirlist)


    plt.ioff()

    

    for i in range(1114,2000) : 
        f,axs = plt.subplots(2,4,figsize=(20,10),dpi=80)
        for j in range(len(dirlist)) : 
            im_fo = plt.imread(fo[j][i])
            im_eo = plt.imread(eo[j][i])
            axs[0,j].imshow(im_fo,origin='lower')
            axs[1,j].imshow(np.fliplr(im_eo),origin='lower')
            axs[0,j].set_title(dirlist[j].replace('_','-'),color='white')
        for ax in axs.flatten(): utils.clear_labels(ax,True)
        plt.subplots_adjust(hspace=.1,wspace=.1)
        f.savefig(prefix+'%d.png'%i,format='png',facecolor='black')
        plt.close()
    
    plt.ion()


def make_age_velocity_plots(si,slist) :
    f,axs = plt.subplots(1,2,figsize=(12,5))
    nosat = pynbody.filt.HighPass('mass',0.05)
    sn = pynbody.filt.SolarNeighborhood(7,9,3)
    prof = pynbody.analysis.profile.Profile(si.s[sn],calc_x=lambda x: x['age'],type='log',nbins=10,min=1)
    for ax in axs:
        ax.plot(prof['rbins'],prof['vr_disp'],color = 'k',label='isolated')
        ax.plot(prof['rbins'],prof['vz_disp'],color = 'k',linestyle='--')

    for i, s in enumerate(slist) :
        prof = pynbody.analysis.profile.Profile(s.s[nosat&sn],calc_x=lambda x: x['age'],type='log',nbins=10,min=1)
        ind = 0 if i < 4 else 1
        axs[ind].plot(prof['rbins'],prof['vr_disp'],color = get_color(i,len(slist)),label=titles_all[i])
        axs[ind].plot(prof['rbins'],prof['vz_disp'],color = get_color(i,len(slist)),linestyle='--')

        
    for ax in axs: 
        ax.legend(frameon=False,prop=dict(size=11),loc='lower right')
        ax.set_xlabel('Age [Gyr]')
        ax.set_ylabel('$\sigma_r, \sigma_z$ [km/s]')
        ax.set_ylim(0,80)


def savefig(name, formats = ['eps','pdf']) : 
    for fmt in formats :
        plt.savefig('migration_paper/'+name+'.%s'%fmt,format=fmt,bbox_inches='tight')

@parallel_util.interruptible
def disp_single(a):
    f,ind = a
    s = pynbody.load(f)
    pynbody.analysis.angmom.faceon(s)
    return s.properties['time'].in_units('Gyr'), np.std(s.s[ind]['vz']), np.std(s.s[ind]['vr'])

def get_z_disp(dir) : 
    from pickle import dump
    import glob
    
    flist = glob.glob(dir+'/*.00??[0,2,4,6,8].gz')
    if len(flist) == 0 :
        flist = glob.glob(dir+'/*.00??[0,2,4,6,8]')
    flist.sort()
    
    time = np.zeros(len(flist))
    sigz = np.zeros(len(flist))
    sigr = np.zeros(len(flist))
    
    rfilt = pynbody.filt.BandPass('rxy',5,7)
    nonsat = pynbody.filt.HighPass('mass',.05)

    s = pynbody.load(flist[0])
    pynbody.analysis.angmom.faceon(s)
    ind = (rfilt&nonsat).where(s.s)[0]
    
    
    res = parallel_util.run_parallel(disp_single,flist,[ind],16)
    
    t = np.zeros(len(flist))
    sigr = np.zeros(len(flist))
    sigz = np.zeros(len(flist))
    
    for i in range(len(res)) : t[i], sigz[i], sigr[i] = res[i]
    
#333    dump({'time':time,'sigz':sigz,'sigr':sigr}, open(dir+'/sigmas.dump','w'))
    return t,sigz,sigr


def make_sig_fig():
    f,axs = plt.subplots(1,2,figsize=(12,5))

    t2,sigz2,sigr2 = get_z_disp('iso/2')
    t5,sigz5,sigr5 = get_z_disp('iso/5')
    
    axs[0].plot(t2,sigr2,color='k',label='isolated')
    axs[0].plot(t2,sigz2,color='k',linestyle='--')
    axs[1].plot(t5,sigr5,color='k',label='isolated')
    axs[1].plot(t5,sigz5,color='k',linestyle='--')

    for i,run in enumerate(runlist_2Gyr):
        t,sigz,sigr = get_z_disp(run+'/2')
        color = colors_all[i] if i < 2 else colors_all[i+2]
        title = titles_all[i] if i < 2 else titles_all[i+2]
        axs[0].plot(t,sigr,color=color,label=title)
        axs[0].plot(t,sigz,color=color,linestyle='--')

    for i,run in enumerate(runlist_5Gyr):
        t,sigz,sigr = get_z_disp(run+'/5')
        color = colors_all[i+2] if i < 2 else colors_all[i+4]
        title = titles_all[i+2] if i < 2 else titles_all[i+4]
        axs[1].plot(t,sigr,color=color,label=title)
        axs[1].plot(t,sigz,color=color,linestyle='--')

    for ax in axs: 
        ax.set_xlabel('$t$ [Gyr]')
        ax.set_ylabel('$\sigma_r, \sigma_z$ [km/s]')
        ax.set_ylim(30,110)
        ax.legend(frameon=False,prop=dict(size=11),loc='upper left')
        
        
