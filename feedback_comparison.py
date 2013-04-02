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

paper_runs = ['nof','std','rad_kappa5','rad_kappa10','rad3','rad_imf3']

paper_names = ['no feedback',
               '$\kappa = 0$',
               '$\kappa = 5$',
               '$\kappa = 10$',
               '$\kappa = 50$',
               'var. $\kappa$']




def load_outputs(flist=list_all, outnum = 101, align = True): 
    return map(lambda s: ram.load_center(s+'/output_%05d'%outnum,align), flist)
    

def make_profile_comparisons(slist, names, load_profile = False, write_profile = False):
    f,axs = plt.subplots(1,4,figsize=(18.5,4))
    
    axs = axs.flatten()

    disk = pynbody.filt.Disc(30,1)

    for i,s in enumerate(slist) : 
        p = pynbody.analysis.profile.Profile(s,min=0.4,max=30,nbins=20, type = 'log', load_from_file=load_profile)
        ps = pynbody.analysis.profile.Profile(s.s[disk],min=0,max=30,nbins=20,load_from_file=load_profile)
        pg = pynbody.analysis.profile.Profile(s.g[disk],min=0,max=30,nbins=20,load_from_file=load_profile)

        color = get_color(i,len(slist))

        axs[0].plot(ps['rbins'],ps['density'].in_units('Msol kpc^-2'),color=color,label=names[i])
        axs[1].plot(pg['rbins'],pg['density'].in_units('Msol kpc^-2'),color=color)
        axs[2].plot(ps['rbins'],ps['vr_disp'].in_units('km s^-1'), color = color)
        axs[2].plot(ps['rbins'],ps['vz_disp'].in_units('km s^-1'), color = color,linestyle='--')
        axs[3].plot(p['rbins'],p['v_circ'].in_units('km s^-1'), color = color)
        
        if write_profile: 
            p.write()
            ps.write()
            pg.write()

    for ax in axs : 
        ax.set_xlim(0,25)
        ax.set_xlabel('$R$ [kpc]')

    ax = axs[0]  
    ax.legend(frameon=False, prop = dict(size=12))
    ax.set_ylim(1e5,9e9)
    ax.semilogy()
   
    ax = axs[1]
    ax.set_ylim(1e5,9e8)
    ax.semilogy()
                
    ax = axs[2]
    ax.set_ylim(0,171)
      
    ax = axs[3]
    ax.set_ylim(0,390)

    axs[0].set_ylabel('$\Sigma_{\star}$ [M$_{\odot}$ kpc$^{-2}$]')
    axs[1].set_ylabel('$\Sigma_{gas}$ [M$_{\odot}$ kpc$^{-2}$]')
    axs[2].set_ylabel('$\sigma_R$, $\sigma_z$ [km/s]')
    axs[3].set_ylabel('$v_{circ}$ [km/s]')

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
def get_color(i,n,cmap=plt.cm.gist_ncar) : 
    return cmap(int(i*256./n))

def make_j_jmax_single(slist,titles) : 
    f, ax = plt.subplots()

    sph = pynbody.filt.Sphere('50 kpc')

    for i,s in enumerate(slist) : 
        ax.hist(s.s[sph]['jz']/s.s[sph]['jzmaxr'],
                range=[-3,3],color=get_color(i,len(slist)),histtype='step',
                bins=100, normed = True, label = titles[i])
    
    ax.legend(loc = 'upper left', frameon=False, prop = dict(size=12))
    ax.set_xlabel('$j_z/j_c(R)$')

    
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
    axs.flatten()[2].set_xlabel('$j_z/j_c(R)$')

def make_image_figure(slist, names) : 
    import matplotlib.image as mpimg
    
    plt.ioff()

    f,axs = plt.subplots(len(slist)/2,4,figsize=(14,1.5*len(slist)))

    axs = axs.flatten()

    for i,s in enumerate(slist): 
        s['pos'].convert_units('kpc')
        s['vel'].convert_units('km s^-1')
        s.g['rho'].convert_units('Msol kpc^-3')

        sph = s[pynbody.filt.Sphere('100 kpc')]

        pynbody.plot.image(sph.g, width=60, qty='rho',av_z='rho',subplot=axs[i*2],cmap=plt.cm.Greys_r,
                           show_cbar=False, approximate_fast=False, vmin=2.5,vmax=9,threaded=10)
        s.rotate_x(90)

        pynbody.plot.image(sph.g, width=60, qty='rho',av_z='rho',subplot=axs[i*2+1],cmap=plt.cm.Greys_r,
                           show_cbar=False, approximate_fast=False,vmin=2.5,vmax=9,threaded=10)
        axs[i*2].annotate(names[i],(0.1,.87),xycoords='axes fraction', color = 'white')
        s.rotate_x(-90)


    # set the colorbar
    bb1 = axs[3].get_position()
    bb2 = axs[-1].get_position()
    cbax = f.add_axes([bb1.x1+.01,bb2.y0,0.02,bb1.y1-bb2.y0])
    cb1 = f.colorbar(axs[-1].get_images()[0],cax=cbax)
    cb1.set_label(r'log($\rho$) [M$_{\odot}/$kpc$^3$]',fontsize='smaller', fontweight='bold')

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

def make_sfh_figure_singlepanel(slist,names) : 
    from scipy.interpolate import interp1d

    f,ax = plt.subplots()

    sph = pynbody.filt.Sphere('20 kpc')

    # read observational data

    tt,aa,sfr,high,low = np.genfromtxt('/home/itp/roskar/rad_fbk/sfr_obs_12.txt').T

#    ax.plot(tt[::10],sfr[::10],color='k')
    ax.fill_between(tt[::10],sfr[::10]+high[::10],sfr[::10]-low[::10],alpha=.2,color='k',label='Moster')
#    ax.plot(tt[::10],sfr[::10],'--k')
    
    # leitner data

    t,z,sfr = get_leitner_data('10.40')
    t2,z2,sfr2 = get_leitner_data('10.60')

    ax.fill_between(pynbody.analysis.cosmology.age(slist[0],z),sfr,sfr2,alpha=.2,color='b',label='Leitner')


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
        ax.plot(bins,sfh/1e9,color = get_color(i,len(slist)), label=names[i])


    ax.set_ylabel('SFR [M$_{\odot}$/yr]')
    ax.set_xlim(0,14)
    ax.legend(loc = 'upper right', frameon=False, prop = dict(size=12))
    ax.set_ylim(1e-3,20)
    ax.set_xlim(.3,14)
    ax.set_xlabel('$t$ [Gyr]')
    
    add_redshift_axis(slist[0],ax)


    
def make_sfh_figure(slist, names) : 
    from scipy.interpolate import interp1d

    f,axs = plt.subplots(3,1,figsize=(8,10))

    sph = pynbody.filt.Sphere('20 kpc')

    # read observational data

    tt,aa,sfr,high,low = np.genfromtxt('/home/itp/roskar/rad_fbk/sfr_obs_12.txt').T

    

    for ax in axs.flatten(): 
        ax.plot(tt[::10],sfr[::10],color='k')
        ax.fill_between(tt[::10],sfr[::10]+high[::10],sfr[::10]-low[::10],alpha=.2,color='k')


    for i, s in enumerate(slist) : 
        sub = s[sph]

        masses = sub.s['mass'].in_units('Msol')
        ind = np.where(pynbody.analysis.cosmology.age(sub)-sub.s['tform'] > .2)[0]
        masses[ind] *= 1.2
        
        sfh,bins = np.histogram(sub.s['tform'].in_units('Gyr'),weights=masses.in_units('Msol'),
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
        ax.set_ylim(1e-3,20)
        ax.set_xlim(.3,14)

    for ax in axs.flatten()[:2] : ax.set_xticklabels('')
    axs.flatten()[-1].set_xlabel('$t$ [Gyr]')
    
def make_abundance_matching_figure(slist, names) : 
    from utils import get_r200
    xmasses = np.logspace(11.5,12.,20)
    ystarmasses, errors = pynbody.plot.stars.moster(xmasses,0.0)

    f,ax = plt.subplots()

    ax.fill_between(xmasses,np.array(ystarmasses)/np.array(errors)/xmasses, 
                    y2 = np.array(ystarmasses)*np.array(errors)/xmasses, facecolor='#BBBBBB',color='#BBBBBB')

    ax.plot(xmasses,ystarmasses/xmasses,'--k')

#    ax.plot(xmasses,np.ones(len(xmasses))*(slist[0].g['mass'].sum()+slist[0].s['mass'].sum())/slist[0].d['mass'].sum(),color='red',linewidth=2)

    for i,s in enumerate(slist) :
        r200 = get_r200(s,pynbody.analysis.profile.Profile(s,ndim=3,min=.4,max=200,type = 'log'))
        print r200
        sph = s[pynbody.filt.Sphere(r200/10.0)]
        sph2 = s[pynbody.filt.Sphere(r200)]

        smass = sph.s['mass'].sum().in_units('Msol')
        hmass = sph2['mass'].sum().in_units('Msol')
        
        print '%s %e %e'%(s.filename,np.log10(smass),hmass)

#        if i==0 : 
 #           ax.plot(np.log10(hmass),smass/hmass,'rx',label=names[i])
 #       else:
        ax.plot(hmass,smass/hmass,'o',color=get_color(i,len(slist)),label=names[i])
        
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


def get_leitner_data(mass) : 
    data=np.genfromtxt('/home/itp/roskar/rad_fbk/sfrs/sfr%s'%mass)
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

def make_all_plots(sl,sl2,names) : 
    
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
    
    make_abundance_matching_figure(sl,names)
    savefig('abundance_matching')

    
    # highz maps
    for s in sl2 : 
        ram.make_rgb_image(s.g[pynbody.filt.Sphere('300 kpc')],400,filename='%s.png'%(s.filename.replace('/','_')))
    make_rgb_figure(paper_runs,paper_names,34,400)
    savefig('rgb_highz')

    # lowz maps
    for s in sl : 
        s.rotate_x(90)
        ram.make_rgb_image(s.g[pynbody.filt.Sphere('350 kpc')],500,filename='%s.png'%(s.filename.replace('/','_')))
        s.rotate_x(-90)
    make_rgb_figure(paper_runs,paper_names,101,500)
    savefig('rgb_lowz')

    
