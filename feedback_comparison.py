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

"""


import pynbody
import numpy as np
import ramses_pynbody as ram
import isolated as iso
import matplotlib.pylab as plt


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
                 'rad_cr_low']

names_var_kappa = []
for l in rad_var_kappa : 
    names_var_kappa.append(l.replace('_','-'))

names_all = names_non_rad + names_fixed_kappa + names_var_kappa

list_all = non_rad + rad_fixed_kappa + rad_var_kappa

def load_outputs(flist=list_all, outnum = 101): 
    return map(lambda s: ram.load_center(s+'/output_%05d'%outnum), flist)
    

def make_comparison_grid(slist, names, load_profile = False, write_profile = False, cmap=plt.cm.jet) : 
    f,axs = plt.subplots(4,3,figsize=(12,12))
    
    disk = pynbody.filt.Disc(20,5)

    for i,s in enumerate(slist) : 
        if i < len(non_rad) : ind = 0
        elif i < len(rad_fixed_kappa) + len(non_rad): ind = 1
        else: ind = 2
        
        p = pynbody.analysis.profile.Profile(s,min=0.01,max=20,nbins=100, type = 'log', load_from_file=load_profile)
        ps = pynbody.analysis.profile.Profile(s.s[disk],min=0,max=20,nbins=20,load_from_file=load_profile)
        pg = pynbody.analysis.profile.Profile(s.g[disk],min=0,max=20,nbins=20,load_from_file=load_profile)

        color = get_color(i,len(slist),cmap=cmap)

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
def get_color(i,n,cmap=plt.cm.gist_ncar) : 
    return cmap(int(i*256./n))


def make_j_jmax_plot(slist,titles) : 
    f,axs = plt.subplots(3,1,figsize=(8,10))

    sph = pynbody.filt.Sphere('50 kpc')

    for i,s in enumerate(slist) : 
        if i < len(non_rad) : ind = 0
        elif i < len(non_rad) + len(rad_fixed_kappa) : ind = 1
        else: ind = 2
        axs.flatten()[ind].hist(s.s[sph]['jz']/s.s[sph]['jzmaxr'],
                                range=[-3,3],color=get_color(i,len(slist)),histtype='step',
                                bins=100, normed = True, label = titles[i])
    
    for ax in axs : ax.legend(loc = 'upper left', frameon=False, prop = dict(size=12))
    for ax in axs.flatten()[:2]: ax.set_xticklabels('')
    axs.flatten()[2].set_xlabel('$j/j_c(R)$')

def make_image_figure(slist, names, figname) : 
    import matplotlib.image as mpimg
    
    plt.ioff()

    f,axs = plt.subplots(len(slist),2,figsize=(7,3*len(slist)))

    sph = pynbody.filt.Sphere('100 kpc')

    for i,s in enumerate(slist): 
        s['pos'].convert_units('kpc')
        s['vel'].convert_units('km s^-1')
        pynbody.plot.image(s[sph].g, width=50, units = 'Msol kpc^-2',subplot=axs[i,0], show_cbar=False)
        s.rotate_x(90)
        pynbody.plot.image(s[sph].g, width=50, units = 'Msol kpc^-2',subplot=axs[i,1], show_cbar=False)
        axs[i,0].annotate(names[i],(0.1,.87),xycoords='axes fraction', color = 'white')
        s.rotate_x(-90)

    for i,ax in enumerate(axs.flatten()) : 
        if i != len(slist)*2-2 : 
            ax.set_xticklabels('')
            ax.set_yticklabels('')
            ax.set_xlabel('')
            ax.set_ylabel('')
    plt.subplots_adjust(hspace=.1,wspace=.05)
    
    savefig(figname)
    
    plt.ion()
    
def make_sfh_figure(slist, names) : 
    from scipy.interpolate import interp1d

    f,axs = plt.subplots(3,1,figsize=(8,10))

    sph = pynbody.filt.Sphere('50 kpc')

    # read observational data

    tt,aa,sfr,high,low = np.genfromtxt('/home/itp/roskar/rad_fbk/sfr_obs_12.txt').T

    

    for ax in axs.flatten(): 
        ax.plot(tt[::10],sfr[::10],color='k')
        ax.fill_between(tt[::10],sfr[::10]+high[::10],sfr[::10]-low[::10],alpha=.2,color='k')
        ax.loglog()

    for i, s in enumerate(slist) : 
        sfh,bins = np.histogram(s.s['tform'].in_units('Gyr'),weights=s.s['mass'].in_units('Msol'),
                                range=[0,13.76],bins=50)
        bins = .5*(bins[:-1]+bins[1:])
        width = bins[1] - bins[0]
        sfh /= width
        if i < len(non_rad) : ind = 0
        elif i < len(non_rad) + len(rad_fixed_kappa) : ind = 1
        else: ind = 2
        
        
        axs.flatten()[ind].plot(bins,sfh/1e9,color = get_color(i,len(slist)), label=names[i])

    for ax in axs.flatten() : 
        ax.set_ylabel('SFR [M$_{\odot}$/yr]')
        ax.set_xlim(0,14)
        ax.legend(loc = 'upper left', frameon=False, prop = dict(size=12))
        ax.set_ylim(1e-3,50)
        ax.set_xlim(.3,14)

    for ax in axs.flatten()[:2] : ax.set_xticklabels('')
    axs.flatten()[-1].set_xlabel('$t$ [Gyr]')
    
    
        
def savefig(name, formats = ['eps','pdf']) : 
    for fmt in formats :
        plt.savefig('feedback_comparison_paper/'+name+'.%s'%fmt,format=fmt,bbox_inches='tight')

