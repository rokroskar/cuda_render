"""
rad_fbk/README:

Non radiative runs
==================

std:
Supernovae feedback with non-thermal energy decay
with 20% of stellar in massive stars
With metal cooling.

nof:
No feedback,
and no metal cooling as a consequence.

noy:
Feedback but no metal enrichments,
and no metal cooling as a consequence.

lowT:
Feedback and metal enrichments,
but only low temperature metal cooling.

Radiative runs with fixed opacity
=================================

rad5 and rad_kappa5:
Everything like run std, but
including radiation from massive stars and
kappa_dust=5 cm^2/g
Give the preference to rad_kappa5.

rad_kappa10:
Same as above but with
kappa_dust=10 cm^2/g

rad4:
Same as above but with
kappa_dust=15 cm^2/g

rad6:
Same as above but with
kappa_dust=30 cm^2/g

rad3:
Same as above but with
kappa_dust=50 cm^2/g

Radiative runs with varying opacity
===================================

rad_cr:
Same as above but with the opacity scales
with the non-thermal energy as

rad_cr_low:
Same as above but with the opacity scales
with the non-thermal energy as
kappa_dust=1 cm^2/g (E_CR/1e-3)

rad_imf:
Same as above but with the opacity scales
with the non-thermal energy as
kappa_dust=1 cm^2/g (E_CR/1e-3)
and the IMF becomes more top-heavy
with increasing E_CR.

rad_imf3: ? 

rad_imf4: ?

"""


import pynbody
import numpy as np
import ramses_pynbody as ram
import isolated as iso
import matplotlib.pylab as plt
from matplotlib.colors import Normalize, LogNorm

non_rad = ['std',
           'nof',
           'noy',
           'lowT']

names_non_rad = non_rad

rad_fixed_kappa = ['rad_kappa5',
                   'rad_kappa10',
                   'rad4',
                   'rad6',
                   'rad3']

names_fixed_kappa = []
for l in rad_fixed_kappa : 
    names_fixed_kappa.append(l.replace('_','-'))


rad_var_kappa = ['rad_cr',
                 'rad_cr_low',
                 'rad_imf3',
                 'rad_imf4']

names_var_kappa = []
for l in rad_var_kappa : 
    names_var_kappa.append(l.replace('_','-'))

names_all = names_non_rad + names_fixed_kappa + names_var_kappa

list_all = non_rad + rad_fixed_kappa + rad_var_kappa

paper_runs = ['nof','std','sf_0p1','kap5','kap30','kap50']

paper_names = ['no feedback',
               'SN only',
               'var. $\kappa$',
               '$\kappa = 5$',
               '$\kappa = 30$',
               '$\kappa = 50$']

paper_rh = [2.1455694165778789, 3.2874142875689363, 4.5211726499787055, 4.5732686281007693, 4.9141374287838229, 3.6425269399441951]

def get_color(i,n,cmap=plt.cm.gist_ncar) : 
    return cmap(int(i*256./n))

paper_linewidths = [1.5,1,1,1,1,1.5]
paper_colors = [get_color(i,len(paper_runs)) for i in range(len(paper_runs))]

kappa_runs  = ['kap1','kap5','kap10','kap15','kap20', 'kap25','kap30','kap40','kap50','kap100']
kappas = np.array([1,5,10,15,20,25,30,40,50,100])
kappa_names = [r'$\kappa = %s$'%(kap[3:]) for kap in kappa_runs]

kappa_colors = plt.cm.Greys(Normalize(vmin=-10,vmax=60)(kappas))

def load_outputs(flist=list_all, outnum = 101, align = True): 
    return map(lambda s: ram.load_center(s+'/output_%05d'%outnum,align), flist)

def load_tipsy_outputs(flist=list_all, outnum = 101, align = True): 
    sl = []
    hs = []
    for run in flist:
        s = pynbody.load('%s/output_%05d_fullbox.tipsy'%(run,outnum))
        h = s.halos(make_grp=True)
        pynbody.analysis.halo.center(h[1],mode='ssc')
        sl.append(s)
        hs.append(h)
        if align: 
            pynbody.analysis.angmom.faceon(h[1].s,mode='ssc',disk_size='10 kpc')

    return sl, hs
    

def make_profile_comparisons(slist, names, load_profile = False, write_profile = False, linewidths = None):
    f,axs = plt.subplots(1,4,figsize=(18.5,4))
    
    axs = axs.flatten()

    disk = pynbody.filt.Disc(30,10)
    cold = pynbody.filt.LowPass('temp', 1.5e4)
    yng = pynbody.filt.LowPass('age', '3 Gyr')

    if linewidths is None: linewidths = np.ones(len(slist))

    for i,s in enumerate(slist) : 
        max_r = '%.2f kpc'%(5.0*paper_rh[i])
        p = pynbody.analysis.profile.Profile(s,min='0.4 kpc',max=max_r,nbins=25, type = 'log', load_from_file=load_profile)
        ps = pynbody.analysis.profile.Profile(s.s[disk],min='0 kpc',max=max_r,nbins=50,load_from_file=load_profile)
        ps_yng = pynbody.analysis.profile.Profile(s.s[yng&disk],min='0 kpc',max=max_r,nbins=50,load_from_file=load_profile)
#        pg = pynbody.analysis.profile.Profile(s.g[disk],min='0 kpc',max=max_r,nbins=100,load_from_file=load_profile)
#        pg_cold = pynbody.analysis.profile.Profile(s.g[disk&cold],min='0 kpc',max=max_r,nbins=100,load_from_file=load_profile)
        
        color = get_color(i,len(slist))

        axs[0].plot(ps['rbins'].in_units('kpc')/paper_rh[i],ps['density'].in_units('Msol kpc^-2'),color=color,label=names[i],
                    linewidth=linewidths[i])
        axs[1].plot(p['rbins'].in_units('kpc')/paper_rh[i],p['v_circ'].in_units('km s^-1'), color = color,linewidth=linewidths[i])

        axs[2].plot(ps['rbins'].in_units('kpc')/paper_rh[i],ps['vt'].in_units('km s^-1')/ps['speed_disp'].in_units('km s^-1'),color=color,linewidth=linewidths[i])
#        axs[2].plot(ps_yng['rbins'].in_units('kpc')/paper_rh[i],ps_yng['vt'].in_units('km s^-1')/ps_yng['speed_disp'].in_units('km s^-1'),color=color,linewidth=linewidths[i], linestyle='--')

        axs[3].plot(ps['rbins'].in_units('kpc')/paper_rh[i],ps['z_rms'].in_units('kpc'), color = color, linewidth=linewidths[i])
        axs[3].plot(ps_yng['rbins'].in_units('kpc')/paper_rh[i],ps_yng['z_rms'].in_units('kpc'), color = color, linewidth=linewidths[i], linestyle='--')
#np.interp(ps['rbins'],p['rbins'],p['v_circ'].in_units('km s^-1'))
        if write_profile: 
            p.write()
            ps.write()
            pg.write()

    for ax in axs : 
        ax.set_xlim(0,5)
        ax.set_xlabel('$R/R_{\mathrm{d}}$')

    ax = axs[0]  
    ax.legend(frameon=False, prop = dict(size=12))
    ax.set_ylim(1e6,3e10)
    ax.semilogy()
   
    ax = axs[1]
    ax.set_ylim(0,400)

   # axs[2].semilogy()
#    axs[2].set_ylim((0,5))
    
    axs[3].set_ylim((0,4))
    

    axs[0].set_ylabel('$\Sigma_{\star}$ [M$_{\odot}$ kpc$^{-2}$]')
    axs[2].set_ylabel('$V_{\mathrm{rot}}/\sigma$')
    axs[3].set_ylabel('$z_{\mathrm{rms},\star}$ [kpc]')
    axs[1].set_ylabel('$V_{\mathrm{circ}}$ [km/s]')

    plt.subplots_adjust(wspace=.3)

def make_comparison_grid(slist, names, load_profile = False, write_profile = False) : 
    if all_runs: ncolumns = 3 
    else: ncolumns=1
    
    f,axs = plt.subplots(4,ncolumns,figsize=(12,ncolumns*4))
    
    disk = pynbody.filt.Disc(20,1)

    for i,s in enumerate(slist) : 
        if all_runs : 
            if i < len(non_rad) : ind = 0
            elif i < len(rad_fixed_kappa) + len(non_rad): ind = 1
            else: ind = 2
        else : ind = 0
        
        p = pynbody.analysis.profile.Profile(s,min=0.01,max=20,nbins=100, type = 'log', load_from_file=load_profile)
        ps = pynbody.analysis.profile.Profile(s.s[disk],min=0,max=20,nbins=20,load_from_file=load_profile)
        pg = pynbody.analysis.profile.Profile(s.g[disk],min=0,max=20,nbins=20,load_from_file=load_profile)

        color = get_color(i,len(slist))

        axs[0,ind].plot(ps['rbins'],ps['density'].in_units('Msol kpc^-2'),color=color,label=names[i])
        axs[1,ind].plot(pg['rbins'],pg['density'].in_units('Msol kpc^-2'),color=color)
        axs[2,ind].plot(ps['rbins'],ps['vr_disp'].in_units('km s^-1'), color = color)
        axs[2,ind].plot(ps['rbins'],ps['vz_disp'].in_units('km s^-1'), color = color,linestyle='--')
        axs[3,ind].plot(p['rbins'],p['v_circ'].in_units('km s^-1'), color = color)
        
        if write_profile: 
            p.write()
            ps.write()
            pg.write()

    for ax in axs.flatten() : ax.set_xlim(0,19.5)

    for ax in axs[0,:].flatten() : 
        ax.legend(frameon=False, prop = dict(size=12))
        ax.set_ylim(1e5,9e9)
        ax.semilogy()
        ax.set_xticklabels('')

    for ax in axs[1,:].flatten() : 
        ax.set_ylim(1e5,9e8)
        ax.semilogy()
        ax.set_xticklabels('')
                
    for ax in axs[2,:].flatten() : 
        ax.set_ylim(0,171)
        ax.set_xticklabels('')
        
    for ax in axs[:,1:].flatten() : 
        ax.set_yticklabels('')

    for ax in axs[3,:].flatten() : 
        ax.set_xlabel('$R$ [kpc]')
        ax.set_ylim(0,390)

    axs[0,0].set_ylabel('$\Sigma_{\star}$ [M$_{\odot}$ kpc$^{-2}$]')
    axs[1,0].set_ylabel('$\Sigma_{gas}$ [M$_{\odot}$ kpc$^{-2}$]')
    axs[2,0].set_ylabel('$\sigma_R$, $\sigma_z$ [km/s]')
    axs[3,0].set_ylabel('$v_{circ}$ [km/s]')

    plt.subplots_adjust(hspace=.05,wspace=.05)



def make_j_jmax_single(slist,titles,linewidths=None) : 
    f, ax = plt.subplots()

    sph = pynbody.filt.Sphere('50 kpc')

    if linewidths is None:
        linewidths=np.ones(len(slist))

    for i,s in enumerate(slist) : 
        ax.hist(s.s[sph]['jz']/s.s[sph]['jzmaxe'],
                range=[-2,2],color=get_color(i,len(slist)),histtype='step',
                bins=100, normed = True, label = titles[i],linewidth=linewidths[i])
    
    ax.legend(loc = 'upper left', frameon=False, prop = dict(size=12))
    ax.set_xlabel('$j_z/j_c(E)$')

    
def make_j_jmax_plot(slist,titles,linewidths=None) : 
    f,axs = plt.subplots(3,1,figsize=(8,10))

    sph = pynbody.filt.Sphere('50 kpc')

    if linewidths is None:
        linewidths=np.ones(len(slist))

    for i,s in enumerate(slist) : 
        if i < len(non_rad) : ind = 0
        elif i < len(non_rad) + len(rad_fixed_kappa) : ind = 1
        else: ind = 2
        axs.flatten()[ind].hist(s.s[sph]['jz']/s.s[sph]['jzmaxr'],
                                range=[-3,3],color=get_color(i,len(slist)),histtype='step',
                                bins=100, normed = True, label = titles[i],linewidth=linewidths[i])
    
    for ax in axs : ax.legend(loc = 'upper left', frameon=False, prop = dict(size=12))
    for ax in axs.flatten()[:2]: ax.set_xticklabels('')
    axs.flatten()[2].set_xlabel('$j_z/j_c(R)$')

def make_image_figure(slist, names) : 
    import matplotlib.image as mpimg
    
    plt.ioff()

    f,axs = plt.subplots(len(slist)/2,4,figsize=(14,1.5*len(slist)))

    axs = axs.flatten()

    for i,s in enumerate(slist): 
        s['pos'].convert_units('kpc')
        s['vel'].convert_units('km s^-1')
        s.g['rho'].convert_units('m_p cm^-3')

        sph = s[pynbody.filt.Sphere('100 kpc')]

        pynbody.plot.image(sph.g, width=40, qty='rho',subplot=axs[i*2],cmap=plt.cm.Greys_r,
                           show_cbar=False, approximate_fast=False, vmin=-4,vmax=2,threaded=10,denoise=True)
        s.rotate_x(90)

        pynbody.plot.image(sph.g, width=40, qty='rho',subplot=axs[i*2+1],cmap=plt.cm.Greys_r,
                           show_cbar=False, approximate_fast=False,vmin=-4,vmax=2,threaded=10,denoise=True)
        axs[i*2].annotate(names[i],(0.1,.87),xycoords='axes fraction', color = 'white')
        s.rotate_x(-90)


    # set the colorbar
    bb1 = axs[3].get_position()
    bb2 = axs[-1].get_position()
    cbax = f.add_axes([bb1.x1+.01,bb2.y0,0.02,bb1.y1-bb2.y0])
    cb1 = f.colorbar(axs[-1].get_images()[0],cax=cbax)
    cb1.set_label(r'log($\rho$) [$m_p$ cm$^{-3}$]',fontsize='smaller', fontweight='bold')

    for tick in cb1.ax.get_yticklabels():
        tick.set_fontsize('smaller')
    #

    for i,ax in enumerate(axs) : 
        if not (ax.is_last_row()) & (i%4 == 0):
            ax.set_xticklabels('')
            ax.set_yticklabels('')
            ax.set_xlabel('')
            ax.set_ylabel('')
        else : 
            plt.setp(ax.get_xticklabels(), fontsize=10)
            plt.setp(ax.get_yticklabels(), fontsize=10)

    plt.subplots_adjust(hspace=.1,wspace=.05)
    
    plt.ion()

def make_sfh_figure_singlepanel(slist,names,linewidths=None) : 
    from scipy.interpolate import interp1d

    f,ax = plt.subplots()

    if linewidths is None: linewidths=np.ones(len(slist))

    sph = pynbody.filt.Sphere('20 kpc')

    # read observational data

    tt,aa,sfr,high,low = np.genfromtxt('/home/itp/roskar/rad_fbk/sfr_obs_12.txt').T

#    ax.plot(tt[::10],sfr[::10],color='k')
#    ax.fill_between(tt[::10],sfr[::10]+high[::10],sfr[::10]-low[::10],alpha=.2,color='k',label='Moster')
#    ax.plot(tt[::10],sfr[::10],'--k')
    
    # leitner data

    t,z,sfr = get_leitner_data('10.40')
    t2,z2,sfr2 = get_leitner_data('10.60')

    ax.fill_between(pynbody.analysis.cosmology.age(slist[0],z),sfr,sfr2,color='#C2C2FF',label='Leitner')


    for i, s in enumerate(slist) : 
        sub = s[sph]
        with sub.immediate_mode:
            masses = sub.s['mass'].in_units('Msol')

        ind = np.where(pynbody.analysis.cosmology.age(sub)-sub.s['tform'] > .2)[0]
        masses[ind] *= 1.2
        
        sfh,bins = np.histogram(sub.s['tform'].in_units('Gyr'),weights=masses.in_units('Msol'),
                                range=[0,13.76],bins=50)
        bins = .5*(bins[:-1]+bins[1:])
        width = bins[1] - bins[0]
        sfh /= width
        
        #if i == 0: 
        #    ax.plot(bins,sfh/1e9,'--r', label=names[i])
        #else:
        ax.plot(bins,sfh/1e9,color = get_color(i,len(slist)), label=names[i], linewidth=linewidths[i])


    ax.set_ylabel('SFR [M$_{\odot}$/yr]')
    ax.set_xlim(0,14)
    ax.legend(loc = 'upper right', frameon=False, prop = dict(size=12))
    ax.set_ylim(1e-3,20)
    ax.set_xlim(.3,14)
    ax.set_xlabel('$t$ [Gyr]')
    
    add_redshift_axis(slist[0],ax)


def make_cummulative_mass(slist,names,kap_runs = None,linewidths=None) : 
    from scipy.interpolate import interp1d

    f,ax = plt.subplots()

    if linewidths is None: linewidths=np.ones(len(slist))

    sph = pynbody.filt.Sphere('20 kpc')
    
    # leitner data

    t,z,sfr = get_leitner_data('10.40',False)
    t2,z2,sfr2 = get_leitner_data('10.60',False)

    ax.fill_between(pynbody.analysis.cosmology.age(slist[0],z),sfr/1e10,sfr2/1e10,zorder=-100,color='#C2C2FF',label='Leitner')


    for i, s in enumerate(slist[:3]) : 
        sub = s[sph]
        with sub.immediate_mode:
            masses = sub.s['mass'].in_units('Msol').view(np.ndarray)
            tform = sub.s['tform'].in_units('Gyr').view(np.ndarray)
        sfh,bins = np.histogram(tform,weights=masses/1e10,range=[0,13.76],bins=50)
        
        bins = .5*(bins[:-1]+bins[1:])
#        return sfh, bins
        ax.plot(bins,np.cumsum(sfh), color = paper_colors[i], 
                label=names[i], linewidth=3)
        
    if kap_runs is not None : 
        for i, s in enumerate(kap_runs) : 
            sub = s[sph]
            with sub.immediate_mode:
                masses = sub.s['mass'].in_units('Msol').view(np.ndarray)
                tform = sub.s['tform'].in_units('Gyr').view(np.ndarray)
            sfh,bins = np.histogram(tform,weights=masses/1e10,range=[0,13.76],bins=50)
            bins = .5*(bins[:-1]+bins[1:])
            ax.plot(bins,np.cumsum(sfh), color = kappa_colors[i],zorder=-50,linewidth=4)
       
        # make the colorbar

        cb_ax = f.add_axes([.15,.7,.4,.05])
        a=np.outer(np.arange(1,100,0.01),np.ones(10))
        cb_ax.imshow(a.T,cmap=plt.cm.Greys,extent=(1,100,0,5),vmin=-10,vmax=70)
        cb_ax.set_yticklabels('')
        cb_ax.set_yticks([])
        plt.setp(cb_ax.get_xticklabels(), fontsize=10)
        cb_ax.set_xticks(kappas)
        cb_ax.set_title('$\kappa$',fontsize=10)


    ax.set_ylabel('$M_{\star}$ [10$^{10}$ M$_{\odot}$]')
    ax.legend(loc = 'upper left', frameon=False, prop = dict(size=12))
#    ax.set_ylim(1e-3,20)
    ax.set_xlim(0,s.s['tform'].max())
    ax.set_xlabel('$t$ [Gyr]')
    
    add_redshift_axis(slist[0],ax)

def make_cummulative_mass_byjjc(slist,names,kap_runs = None,linewidths=None) : 
    from scipy.interpolate import interp1d

    f,ax = plt.subplots()

    if linewidths is None: linewidths=np.ones(len(slist))

    sph = pynbody.filt.Sphere('20 kpc')

    for i, s in enumerate(slist) : 
        sub = s[sph]
        with sub.immediate_mode:
            masses = sub.s['mass'].in_units('Msol').view(np.ndarray)
            tform = sub.s['tform'].in_units('Gyr').view(np.ndarray)
        
        disk = np.where(sub.s['jz']/sub.s['jzmaxe'] > .8)[0]
        bulge = np.where(sub.s['jz']/sub.s['jzmaxe'] < .5)[0]

        sfh_d,bins_d = np.histogram(tform[disk],weights=masses[disk]/1e10,range=[0,13.76],bins=50)
        sfh_b,bins_b = np.histogram(tform[bulge],weights=masses[bulge]/1e10,range=[0,13.76],bins=50)
        
        bins_d = .5*(bins_d[:-1]+bins_d[1:])
        bins_b = .5*(bins_b[:-1]+bins_b[1:])
        
        ax.plot(bins_d,np.cumsum(sfh_d), color = paper_colors[i], 
                label=names[i], linewidth=3)
        ax.plot(bins_b,np.cumsum(sfh_b), linestyle = '--', color = paper_colors[i], linewidth=3)
        
        

    ax.set_ylabel('$M_{\star}$ [10$^{10}$ M$_{\odot}$]')
    ax.legend(loc = 'upper left', frameon=False, prop = dict(size=12))
#    ax.set_ylim(1e-3,20)
    ax.set_xlim(0,13.76)
    ax.set_xlabel('$t$ [Gyr]')
    
    add_redshift_axis(slist[0],ax)

def make_sfh_byjjc(slist,names,kap_runs = None,linewidths=None) : 
    from scipy.interpolate import interp1d

    f,ax = plt.subplots()

    if linewidths is None: linewidths=np.ones(len(slist))

    sph = pynbody.filt.Sphere('20 kpc')

    for i, s in enumerate(slist[:3]) : 
        sub = s[sph]
        with sub.immediate_mode:
            masses = sub.s['mass'].in_units('Msol').view(np.ndarray)
            tform = sub.s['tform'].in_units('Gyr').view(np.ndarray)
        
        disk = np.where(sub.s['jz']/sub.s['jzmaxe'] > .7)[0]
        bulge = np.where(sub.s['jz']/sub.s['jzmaxe'] < .5)[0]

        sfh_d,bins_d = np.histogram(tform[disk],weights=masses[disk]/1e10,range=[0,13.76],bins=50)
        sfh_b,bins_b = np.histogram(tform[bulge],weights=masses[bulge]/1e10,range=[0,13.76],bins=50)
        
        bins_d = .5*(bins_d[:-1]+bins_d[1:])
        bins_b = .5*(bins_b[:-1]+bins_b[1:])
        
        ax.plot(bins_d,sfh_d, color = paper_colors[i], 
                label=names[i], linewidth=3)
        ax.plot(bins_b,sfh_b, linestyle = '--', color = paper_colors[i], linewidth=3)
        
        

    ax.set_ylabel('$M_{\star}$ [10$^{10}$ M$_{\odot}$]')
    ax.legend(loc = 'upper left', frameon=False, prop = dict(size=12))
#    ax.set_ylim(1e-3,20)
    ax.set_xlim(0,13.76)
    ax.set_xlabel('$t$ [Gyr]')
    
    add_redshift_axis(slist[0],ax)


def gen_halomass_data(sl) : 
    from pickle import dump
    from utils import get_r200

    for s in sl : 
        r200 = get_r200(s,pynbody.analysis.profile.Profile(s,ndim=3,min=.4,max=200,type = 'log'))
        print r200
        sph = s[pynbody.filt.Sphere(r200/10.0)]
        sph2 = s[pynbody.filt.Sphere(r200)]

        smass = sph.s['mass'].sum().in_units('Msol')
        dmass = sph2.d['mass'].sum().in_units('Msol')
        gmass = sph2.g['mass'].sum().in_units('Msol')
        hmass = sph2['mass'].sum().in_units('Msol')
        
        print '%s %e %e %e %e'%(s.filename,smass/1e11,gmass/1e11,dmass/1e11,hmass/1e11)

        dump({'smass':smass, 'hmass':hmass, 'dmass':dmass, 'gmass':gmass, 'r200':r200},open('%s.smhm'%s.filename,'w'))

def make_abundance_matching_figure(slist, names, kappa_runs = None) : 
    from pickle import load

    xmasses = np.logspace(11.5,12.,20)
    ystarmasses, errors = pynbody.plot.stars.moster(xmasses,0.0)

    f,ax = plt.subplots()

    ax.fill_between(xmasses,np.array(ystarmasses)/np.array(errors)/xmasses, 
                    y2 = np.array(ystarmasses)*np.array(errors)/xmasses, facecolor='#BBBBBB',color='#BBBBBB')

    ax.plot(xmasses,ystarmasses/xmasses,'--k')

    for i,s in enumerate(slist) :
        smhm = load(open('%s.smhm'%s.filename,'r'))
        ax.plot(smhm['hmass'],smhm['smass']/smhm['hmass'],'o',ms=10,color=paper_colors[i],label=names[i],linewidth=1)

    if kappa_runs is not None: 
        for i, s in enumerate(kappa_runs) : 
            smhm = load(open('%s.smhm'%s.filename,'r'))
            ax.plot(smhm['hmass'],smhm['smass']/smhm['hmass'],'o',ms=10, markeredgewidth=.5,
                    color=kappa_colors[i])

        # make the colorbar

        cb_ax = f.add_axes([.15,.7,.4,.05])
        a=np.outer(np.arange(1,100,0.01),np.ones(10))
        cb_ax.imshow(a.T,cmap=plt.cm.Greys,extent=(1,100,0,5),vmin=-10,vmax=60)
        cb_ax.set_yticklabels('')
        cb_ax.set_yticks([])
        plt.setp(cb_ax.get_xticklabels(), fontsize=10)
        cb_ax.set_xticks(kappas)
        cb_ax.set_title('$\kappa$',fontsize=10)


    ax.legend(frameon=False, prop = dict(size=12),loc='upper left',scatterpoints=1)
    ax.set_xlim(.4e12,.7e12)
 #  ax.set_ylim(10.0,11.5)
    ax.set_ylabel('$M_{\star}/M_{h}$')
    ax.set_xlabel('$M_{h}$ [M$_{\odot}$]')


def add_redshift_axis(s,ax) : 
    newax = ax.twiny()

    z_arr = np.array([4,3,2,1,.5,.25,.1,0])
    snap_arg = []
    for i in range(len(z_arr)) : 
        snap_arg.append(s)
    times = map(pynbody.analysis.cosmology.age, snap_arg, z_arr)

    newax.set_xticks(times)
    newax.set_xticklabels(z_arr)
    newax.set_xlabel('redshift')
    plt.setp(newax.get_xticklabels(),fontsize=15)

def savefig(name, formats = ['eps','pdf']) : 
    for fmt in formats :
        plt.savefig('feedback_comparison_paper/'+name+'.%s'%fmt,format=fmt,bbox_inches='tight')


def get_leitner_data(mass, sfr=True) :
    if sfr: 
        data=np.genfromtxt('/home/itp/roskar/rad_fbk/sfrs/sfr%s'%mass)
    else : 
        data=np.genfromtxt('/home/itp/roskar/rad_fbk/masses/ms_%s.dat'%mass)

    return data[:,0],data[:,1],data[:,2]


def make_rgb_figure(runs,names,outnum,width) : 
    from utils import clear_labels

    f,axs = plt.subplots(2,3,figsize=(12.5,8))

    for i, run in enumerate(runs) : 
        ax = axs.flatten()[i]
        ax.imshow(plt.imread("%s_output_%05d.png"%(run,outnum)),origin='lower')
        ax.annotate(names[i],(50,450),color='white')
        if i == 0: 
            ax.annotate("", xy=(0.01,0.05),xytext=(0.99,0.05), xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='<->',color='white',linewidth=2))
            ax.annotate("%s kpc"%str(width), xy=(0.38,0.065), color ="white",fontsize='smaller', 
                        xycoords = 'axes fraction')

    for ax in axs.flatten() : clear_labels(ax)
    plt.subplots_adjust(hspace=.05,wspace=.05)

def make_rgb_stellar_figure(runs, names, outnum, width) : 
    from utils import clear_labels

    f,axs = plt.subplots(len(runs)/2,4,figsize=(14,1.5*len(runs)))
    
    for i, run in enumerate(runs) : 
        ax = axs.flatten()[i*2]
        ax.imshow(plt.imread("%s_output_%05d_stars_fo.png"%(run,outnum)),origin='lower')
        ax.annotate(names[i],(0.1,.87),xycoords='axes fraction',color='white')
        if i == 0: 
            ax.annotate("", xy=(0.01,0.05),xytext=(0.99,0.05), xycoords='axes fraction',
                        arrowprops=dict(arrowstyle='<->',color='white',linewidth=2))
            ax.annotate("%s kpc"%str(width), xy=(0.38,0.065), color ="white",fontsize='smaller', 
                        xycoords = 'axes fraction')

        ax = axs.flatten()[i*2+1]
        ax.imshow(plt.imread("%s_output_%05d_stars_eo.png"%(run,outnum)),origin='lower')

    for ax in axs.flatten() : clear_labels(ax)
    plt.subplots_adjust(hspace=.05,wspace=.05)


def make_metallicity_profiles(sl,names,kappa_runs = None) : 
    f,axs = plt.subplots(2,1,figsize=(8,10))
    axs = axs.flatten()
    f2,ax2 = plt.subplots()
    sph = pynbody.filt.Sphere('500 kpc')
    
    # observational data

    obsrho=np.array([265, 126, 255, 181, 156, 124, 155, 130, 140, 87, 294, 64,
                 83, 209, 221, 32])
    obsnhi=np.array([13.8,15.14,14.64,13.46, 14.04, 13.64, 14.4, 15.67,
                 15.73, 14.8, 14.83, 15.35, 15.29, 15.01, 14.3, 15.11])
    obsnovi = np.array([13.74, 14.9, 13.71, 13.6, 13.35, 13.77, 14.08, 14.3,
                    14.16, 14.7, 14.38, 13.75, 14.16, 13.93, 13.61, 13.1])

    lowlrho=np.array([147,93,73,286,207,172,68,86,34,98,294,79,299,285,150,282,296,272,113])
    lowlnhi=np.array([13.8,15.47,13.90,14.00,13.64,15.67,15.73,19.3,13.1,13.94,13.87,15.38,13.3,13.67,13.1,14.08,14.3,15.11,13.56])
    lowlnovi=np.array([13.48,14.64,13.6,5,13.77,14.3,14.16,14.2,14.2,14.2,14,13.2,13.5,13.1,13.5,13.9,13.6,13.1,13])

    tumrho = np.array([77,54,90,113,89,16,19,85,98,114,14,83,30,101,93,118,60,22,51,150,140,29,20,109,33,88,36,140,109,101,32,96,115,450,155,100,46,46,55,94,32,84,32,153])
    tumnovi = np.array([13.53,14.21,14.19,13.96,14.28,14.26,13.64,14.38,14.17,14.14,14.78,14.57,14.43,14.69,13.94,13.56,15.00,14.86,14.70,14.37,14.38,14.68,14.81,14.34,14.48,14.59,14.73,14.57,14.29,13.73,14.56,14.27,14.7,13.46,13.8,14.44,15.07,14.7,14.43,14.76,13.38,14.57,14.51,13.68])

    werk_names, werk_rho, werk_ovi = get_werk_data()

    for i,s in enumerate(sl) :
        pg = pynbody.analysis.profile.Profile(s.g,ndim=3,nbins=100,type='log',min='.1 kpc', max = '450 kpc')
        axs[0].plot(pg['rbins'],np.log10(pg['temp']),label=names[i],color=paper_colors[i],linewidth=2)
        if i == 0: pass
        else : 
            im = pynbody.plot.image(s.g,qty='rho_ovi',width='900 kpc', units = 'm_p cm^-2',subplot=ax2)
            rad_prof = pynbody.plot.sph.image_radial_profile(im)
        #       ax.plot(p['rbins'], p['metal'], color = paper_colors[i], linewidth=2, label = names[i])
            axs[1].plot(rad_prof[-1],rad_prof[0],color=paper_colors[i],linewidth=2)
        
    for ax in axs: ax.plot([175,175],[3,1e18],'r--')

    axs[1].plot(obsrho,10**obsnovi,'o',markersize=8, color='#C2C2FF',label = 'Prochaska et al. 2011')
    axs[1].plot(tumrho,10**tumnovi,'s',markersize=8, color='#C5FFC2',label = 'Tumlinson et al. 2011')
    axs[1].plot(werk_rho[werk_ovi>0], 10**werk_ovi[werk_ovi>10], '*', markersize=8, color='#FFC2CC',label = 'Werk et al. 2013')
    axs[1].set_xlabel('Radius [kpc]')
    axs[0].set_xlabel('Radius [kpc]')
    axs[0].set_ylabel('log$_{10}$(T)')
    axs[1].set_ylabel('O~$\mathrm{VI}$ column density [cm$^{-2}$]')
    axs[0].legend(frameon=False, prop = dict(size=12),loc='lower right')
    axs[1].legend(frameon=False, prop = dict(size=12),loc='lower left')


    axs[1].loglog()
    axs[0].semilogx()
    axs[0].set_xlim(10,450)
    axs[0].set_ylim(3.8,6.5)
    axs[1].set_xlim(10,450)
    axs[1].set_ylim(1e11,3e17)

def get_werk_data():
    werk_gal_props = np.genfromtxt('/home/itp/roskar/rad_fbk/werk2013_galproperties.txt', dtype=None,skip_footer=2,skip_header=5)
    werk_data = data = np.genfromtxt('werk2013_table.txt', skip_header=32, dtype=None,delimiter=[18,7,7,7,6,6,6,6,5,2,6,5])

    werk_names = []
    werk_rho = np.array([])
    werk_ovi = np.zeros(len(werk_gal_props))

    for gal in werk_gal_props : 
        werk_names.append(gal[0])
        werk_rho = np.append(werk_rho,gal[2])
        
    werk_names = np.array(werk_names)

    for i,gal in enumerate(werk_data) : 
        if gal[2].strip() == 'O VI' :
            werk_ovi[np.where(gal[0].strip() == werk_names)[0]] = gal[10]
            
            
    return werk_names, werk_rho, werk_ovi
    
def make_all_plots(sl,sl2,kap,names) : 
    
    # image figure 

    make_image_figure(sl,names)
    savefig('allruns_images')

    # profile comparisons figure

    make_profile_comparisons(sl,names,True,True)
    savefig('radial_properties')

    # j/jc figure

    make_j_jmax_single(sl,names)
    savefig('jjc')

    # SFR figure

    make_sfh_figure_singlepanel(sl,names)
    savefig('sfh')
    
    # abundance matching
    
    make_abundance_matching_figure(sl[:3],names,kap)
    savefig('abundance_matching')

    # cumulative mass

    make_cummulative_mass(sl,names,kap)
    savefig('cumulative_mass')

    # highz maps
    for s in sl2 : 
        ram.make_rgb_image(s.g[pynbody.filt.Sphere('300 kpc')],'400 kpc',filename='%s.png'%(s.filename.replace('/','_')))
    make_rgb_figure(paper_runs,paper_names,34,400)
    savefig('rgb_highz')

    # lowz maps
    for s in sl : 
        s.rotate_x(90)
        ram.make_rgb_image(s.g[pynbody.filt.Sphere('350 kpc')],'500 kpc',filename='%s.png'%(s.filename.replace('/','_')))
        s.rotate_x(-90)
    make_rgb_figure(paper_runs,paper_names,101,500)
    savefig('rgb_lowz')


    # stellar maps 

    for s in sl : 
        ram.make_rgb_stellar_image(s,'40 kpc', filename = '%s_stars_fo.png'%(s.filename.replace('/','_')))
        s.rotate_x(90)
        ram.make_rgb_stellar_image(s,'40 kpc', filename = '%s_stars_eo.png'%(s.filename.replace('/','_')))
        s.rotate_x(-90)

    make_rgb_stellar_figure(paper_runs,paper_names,101,500)
    
    
def compare_to_high_res(slr1, slr2, shr) : 
    f,axs = plt.subplots(1,4,figsize=(18.5,4))
    sph = pynbody.filt.Sphere('20 kpc')
    disk = pynbody.filt.Disc('30 kpc','10 kpc')
    axs = axs.flatten()
    
    for i, s in enumerate([slr1,slr2,shr]) : 
        
        if i < 2: 
            color = paper_colors[i]
            linestyle = '-'
            label = paper_names[i]
            rd = paper_rh[i]
        else : 
            color = 'k'
            linestyle = '--'
            label = 'high res'
            rd = 3.35
        
        max_r = '%.2f kpc'%(5*rd)
        p = pynbody.analysis.profile.Profile(s,max=max_r,type = 'log', min='.1 kpc', nbins=50)
        ps = pynbody.analysis.profile.Profile(s.s[disk],max=max_r)
        
        axs[0].plot(ps['rbins']/rd,ps['density'].in_units('Msol kpc^-2'),color=color,linestyle=linestyle,label=label)
        axs[1].plot(p['rbins']/rd,p['v_circ'].in_units('km s^-1'),color=color,linestyle=linestyle,label=label)
        axs[2].plot(ps['rbins']/rd,ps['z_rms'].in_units('kpc'),color=color,linestyle=linestyle)

        sub = s[sph]
        with sub.immediate_mode:
            masses = sub.s['mass'].in_units('Msol')

        ind = np.where(pynbody.analysis.cosmology.age(sub)-sub.s['tform'] > .2)[0]
        masses[ind] *= 1.2
        
        sfh,bins = np.histogram(sub.s['tform'].in_units('Gyr'),weights=masses.in_units('Msol'),
                                range=[0,13.76],bins=50)
        bins = .5*(bins[:-1]+bins[1:])
        width = bins[1] - bins[0]
        sfh /= width
        
        #if i == 0: 
        #    ax.plot(bins,sfh/1e9,'--r', label=names[i])
        #else:
        axs[3].plot(bins,sfh/1e9,color = color, linestyle=linestyle)

    axs[0].semilogy()
    axs[0].set_xlabel('$R/R_{\mathrm{d}}$ [kpc]')
    axs[0].set_ylabel('$\Sigma_{\star}$ [M$_{\odot}$/kpc]')
    axs[0].legend(frameon=False, prop = dict(size=12))
    axs[1].set_xlabel('$R/R_{\mathrm{d}}$ [kpc]')
    axs[1].set_ylabel('$V_{\mathrm{circ}}$')
    axs[2].set_xlabel('$R/R_{\mathrm{d}}$ [kpc]')
    axs[2].set_ylabel('$z_{\mathrm{rms}}$')
    axs[3].set_xlabel('$t$ [Gyr]')
    axs[3].set_ylabel('SFR [M$_{\odot}$/yr]')
    
    plt.subplots_adjust(wspace=.25)



def snrad() : 
    plt.figure()

    kappa_IR=1. # Draine dust opacity in cm^2/g
    parsec=3e18
    n_star=2.4
    delta_x=200.*parsec
    n=10**(np.linspace(0,11,500)-6.) # Gas H density in H/cc
    tau=kappa_IR*delta_x*n*1.66e-24/0.76
    fuv=1.-np.exp(-1000.*tau)
    Tsn=1e51/2e34/1.38e-16*1.66e-24
    Trad=1e53/2e34/1.38e-16*1.66e-24

    plt.plot(n,Tsn*0.1*n_star/(n+0.1*n_star), 'k--', label = 'supernovae')
    
    eta_rad = 1.
    eta_sig = 1.
    plt.plot(n,eta_rad*Trad*fuv*(1.-np.exp(-eta_sig*tau))*0.1*n_star/(n+0.1*n_star), 
             label='$\kappa_{\mathrm{IR}}=1 {\mathrm{~cm}}^2/{\mathrm{g}}$')

    eta_rad=1.0
    eta_sig=5.
    plt.plot(n,eta_rad*Trad*fuv*(1.-np.exp(-eta_sig*tau))*0.1*n_star/(n+0.1*n_star), 
             label='$\kappa_{\mathrm{IR}}=5 {\mathrm{~cm}}^2/{\mathrm{g}}$')

    eta_rad=1.
    eta_sig=25.
    plt.plot(n,eta_rad*Trad*fuv*(1.-np.exp(-eta_sig*tau))*0.1*n_star/(n+0.1*n_star), 
             label='$\kappa_{\mathrm{IR}}=25 {\mathrm{~cm}}^2/{\mathrm{g}}$')

#    eta_rad=0.1
#    eta_sig=10.
#    plt.plot(n,eta_rad*Trad*fuv*(1.-np.exp(-eta_sig*tau))*0.1*n_star/(n+0.1*n_star), 
#             label='$\kappa_{\mathrm{IR}}=1 {\mathrm{~cm}}^2/{\mathrm{g}}$')

    plt.plot([1e-4,1e4],[1e7,1e7], 'k:')

    plt.loglog()
    
    plt.xlabel('Hydrogen number density [amu/cm$^3$]')
    plt.ylabel('Cell temperature [K]')

    plt.xlim(1e-4,1e4)
    plt.ylim(1e6,1e10)

    plt.legend(frameon=False)


def sb99(): 
    plt.figure()

    data = np.genfromtxt('/home/itp/roskar/homegrown/romain_plots/Lbol.dat',names=['t','lrad','erad','lwind','ewind','lsn','esn'])

    plt.plot(data['t'],10**data['erad'],'k-', label='radiation')
    plt.plot(data['t'],10**data['ewind'],'k--', label='winds')
    plt.plot(data['t'],10**data['esn'],'k:', label='supernovae')
    
    plt.loglog()
    plt.xlim(1e4,1e9)
    plt.ylim(1e46,1e52)
    plt.legend(loc='upper left', frameon=False)

    plt.xlabel('Age [yr]')
    plt.ylabel('Specific energy released [erg M${_\odot}^{-1}$]')
    
def dust() : 
    plt.figure()

    data = np.genfromtxt('/home/itp/roskar/homegrown/romain_plots/kext_albedo_WD_MW_3.1_60_D03.all',
                         names = ['l','a','cos','sig','kap','cos2'],usecols=range(6),skip_header=80)

    
    
    #ll = 10.**(np.linspace(0,4,100)-1)
    
    plt.plot(data['l'],data['kap']/125.,'k')
    
    data = np.genfromtxt('/home/itp/roskar/homegrown/romain_plots/0__comp.Gofn',
                         names=['l1','ka','kb','kc','kd'],usecols=range(5))
    plt.plot(data['l1'],data['kc'],'k')

    data = np.genfromtxt('/home/itp/roskar/homegrown/romain_plots/0__Gofn.kap',
                         names=['l1','ka','kb','kc','kd'],usecols=range(5))
    plt.plot(data['l1'],data['kc'],'k')

    data = np.genfromtxt('/home/itp/roskar/homegrown/romain_plots/0__komp.kap',
                         names=['l1','ka','kb','kc','kd'],usecols=range(5))
    plt.plot(data['l1'],data['kc'],'k')

    data = np.genfromtxt('/home/itp/roskar/homegrown/romain_plots/1.dat',
                         names=['l1','ka','kb','kc','kd'],usecols=range(5))
    plt.plot(data['l1'],data['kc'],'k')
    

    plt.loglog()
    
    plt.xlim(1e-1,1e3)
    plt.ylim(1e-2,1e3)

    plt.xlabel('$\lambda$ [micron]')
    plt.ylabel('$\kappa$ [cm$^2$/g]')
    
