



import pynbody
import pynbody.analysis.profile as profile
import numpy as np
import glob
import scipy as sp
import matplotlib.pylab as plt
from isolated import get_rform

def percent_outside_break() : 
    breaks = [8.0,9.3,10.0]
    sims = ['r61e11M_lam0.025','r61e11M_lam0.0355_bf0.1_UV','12M_hr']

    for i,sim in enumerate(sims) : 
        s = pynbody.load(sim)
        pynbody.analysis.angmom.faceon(s)
        ind = np.where(s.s['rxy'] > breaks[i])
        

def save_profile(dir,name,rbr) :
    s = pynbody.load(dir+'/10/'+dir+'.01000')
    s.physical_units()
    pynbody.analysis.angmom.faceon(s,disk_size='3 kpc')

    get_rform(s.s)

    s.s['age'].convert_units('Gyr')

    old = np.where((s.s['age'] > 1.0) & (s.s['rform'] < rbr))
    mid = np.where((s.s['age'] < 1.0) & (s.s['age'] > 0.1) & (s.s['rform'] < rbr))
    yng = np.where((s.s['age'] < 0.1) & (s.s['rform'] < rbr))

    p_old = profile.Profile(s.s[old], max = 15, nbins = 30)
    p_mid = profile.Profile(s.s[mid], max = 15, nbins = 30)
    p_yng = profile.Profile(s.s[yng], max = 15, nbins = 30)

    s.rotate_x(53.0)

    p_old_i = profile.InclinedProfile(s.s[old], 53.0, max = 15, nbins = 30)
    p_mid_i = profile.InclinedProfile(s.s[mid], 53.0, max = 15, nbins = 30)
    p_yng_i = profile.InclinedProfile(s.s[yng], 53.0, max = 15, nbins = 30)

    plt.plot(p_old['rbins'], p_old['density'].in_units('Msol kpc**-2'))
    plt.plot(p_mid['rbins'], p_mid['density'].in_units('Msol kpc**-2'))
    plt.plot(p_yng['rbins'], p_yng['density'].in_units('Msol kpc**-2'))

    plt.plot(p_old_i['rbins'], p_old_i['density'].in_units('Msol kpc**-2'),'--')
    plt.plot(p_mid_i['rbins'], p_mid_i['density'].in_units('Msol kpc**-2'),'--')
    plt.plot(p_yng_i['rbins'], p_yng_i['density'].in_units('Msol kpc**-2'),'--')


    plt.semilogy()

    plt.savefig(name+'_dens_plots_inner.pdf', format='pdf')

    old_x = np.array(p_old['rbins'])
    mid_x = np.array(p_mid['rbins'])
    yng_x = np.array(p_yng['rbins'])

    prof_old = np.array(p_old['density'].in_units('Msol kpc**-2'))
    prof_mid = np.array(p_mid['density'].in_units('Msol kpc**-2'))
    prof_yng = np.array(p_yng['density'].in_units('Msol kpc**-2'))

    prof_old_i = np.array(p_old_i['density'].in_units('Msol kpc**-2'))
    prof_mid_i = np.array(p_mid_i['density'].in_units('Msol kpc**-2'))
    prof_yng_i = np.array(p_yng_i['density'].in_units('Msol kpc**-2'))


    np.savez(name+'_profiles_inner', old_x=old_x, mid_x=mid_x, yng_x=yng_x, prof_old=prof_old, prof_mid=prof_mid, prof_yng=prof_yng, prof_old_i=prof_old_i, prof_mid_i = prof_mid_i, prof_yng_i = prof_yng_i)


    
    sfh_all, t_all = np.histogram(s.s['tform'],weights=s.s['massform'],bins=100)
    sfh_all_c = sfh_all.cumsum()
    t_all = (t_all[:-1]+t_all[1:])/2.0

    # calculate ellipsoidal radius
    
    s.s['r_ell'] = np.sqrt(s.s['x']**2 + (s.s['y']/np.cos(np.radians(53.0)))**2)

    bp = pynbody.filt.BandPass('r_ell',.7*10,10)
    
    sfh,t = np.histogram(s.s[bp]['tform'],weights=s.s[bp]['massform'],bins=100)
    sfhc = sfh.cumsum()
    t = (t[:-1]+t[1:])/2.0


    plt.figure()
    plt.plot(t,sfhc/sfhc[-1],label='$0.7 < R/R_{break} < 1.0$')
    plt.plot(t_all,sfh_all_c/sfh_all_c[-1],label='$\\mathrm{all}$')

    plt.legend(loc='upper left')

    plt.xlabel('$t/\\mathrm{Gyr}$')
    plt.ylabel('$M_{SF}/M_{tot}$')

    plt.savefig('sfh_cumu.pdf',format='pdf')

    np.savez('sfh', t_all = t_all, sfh_all = sfh_all, sfh_all_c = sfh_all_c, 
             t = t, sfh = sfh, sfhc = sfhc)

    
               


def plot_in_out_profiles(dir,rbr,title) :

    s = pynbody.load(dir+'/10/'+dir+'.01000')
    s.physical_units()
    pynbody.analysis.angmom.faceon(s,disk_size='3 kpc')


    get_rform(s.s)


    s.rotate_x(53.0)

    
    s.s['age'].convert_units('Gyr')
    
    old = pynbody.filt.HighPass('age', 1.0)
    mid = pynbody.filt.BandPass('age',.1,1.0)
    yng = pynbody.filt.BandPass('age', .01,.1)
    yngst = pynbody.filt.LowPass('age', .01)

#    old = pynbody.filt.LowPass('tform', 9.0)
#    mid = pynbody.filt.BandPass('tform',9.0,9.9)
#    yng = pynbody.filt.BandPass('tform', 9.9,9.99)
#    yngst = pynbody.filt.HighPass('tform', 9.99)

    # complete profiles
    p_old = profile.InclinedProfile(s.s[old],53.0, max=20.0, nbins=30)
    p_mid = profile.InclinedProfile(s.s[mid],53.0, max=20.0, nbins=30)
    p_yng = profile.InclinedProfile(s.s[yng],53.0, max=20.0, nbins=30)
    p_yngst = profile.InclinedProfile(s.s[yngst],53.0, max=20.0, nbins=30)

    # profiles for stars forming inside R_br
    
    inner = pynbody.filt.LowPass('rform', rbr)
    outer = pynbody.filt.HighPass('rform', rbr)

    p_old_in = profile.InclinedProfile(s.s[old&inner], 53.0, max = 20, nbins = 30)
    p_mid_in = profile.InclinedProfile(s.s[mid&inner], 53.0, max = 20, nbins = 30)
    p_yng_in = profile.InclinedProfile(s.s[yng&inner], 53.0, max = 20, nbins = 30)
    p_yngst_in = profile.InclinedProfile(s.s[yngst&inner], 53.0, max = 20, nbins = 30)
    
    # profiles for stars formed outside R_br

    p_old_out = profile.InclinedProfile(s.s[old&outer], 53, max = 20, nbins = 30)
    p_mid_out = profile.InclinedProfile(s.s[mid&outer], 53, max = 20, nbins = 30)
    p_yng_out = profile.InclinedProfile(s.s[yng&outer], 53, max = 20, nbins = 30)
    p_yngst_out = profile.InclinedProfile(s.s[yngst&outer], 53, max = 20, nbins = 30)
    
    
    # profiles of rform

    p_old_rf = profile.Profile(s.s[old],max=20,nbins=30,calc_x=lambda x: s.s[old]['rform'])
    p_mid_rf = profile.Profile(s.s[mid],max=20,nbins=30,calc_x=lambda x: s.s[mid]['rform'])
    p_yng_rf = profile.Profile(s.s[yng],max=20,nbins=30,calc_x=lambda x: s.s[yng]['rform'])


    # gas profile
    
    gfilt = pynbody.filt.LowPass('temp',1e5)&pynbody.filt.BandPass('z', -1, 1)
    pg = profile.Profile(s.g[gfilt],max=20,nbins=30)
    pg['density'].convert_units('Msol kpc^-2')
    
    plt.figure()
    
    colors = ['k-','k-','k-','r-','g-','b-','m-','r--','g--','b--','m--']

    for i,p in enumerate([p_old,p_mid,p_yng,p_old_in, p_mid_in, p_yng_in,p_yngst_in,
                         p_old_out,p_mid_out,p_yng_out,p_yngst_out]) : 
        p['density'].convert_units('Msol kpc^-2')
        plt.plot(p['rbins'],p['density'],colors[i])

    
    plt.semilogy()
    plt.xlabel(r'$R$ [kpc]')
    plt.ylabel(r'$\Sigma_{\star}$ [M$_{\odot}$ kpc$^{-2}$]')
    plt.title(title)

    plt.figure()

    colors = ['r-','g-','b-','r--','g--','b--']
    
    for i,p in enumerate([p_old,p_mid,p_yng,p_old_rf,p_mid_rf,p_yng_rf]):
        p['density'].convert_units('Msol kpc^-2')
        plt.plot(p['rbins'],p['density'],colors[i])
    
    plt.semilogy()
    plt.xlabel(r'$R$ [kpc]')
    plt.ylabel(r'$\Sigma_{\star}$ [M$_{\odot}$ kpc$^{-2}$]')
    plt.title(title)


    # save the profiles

    np.savez(title+'profiles', 
             old_r   = p_old['rbins'], # complete
             old_den = p_old['density'],
             mid_r   = p_mid['rbins'],
             mid_den = p_mid['density'],
             yng_r   = p_yng['rbins'],
             yng_den = p_yng['density'],
             yngst_r = p_yngst['rbins'],
             yngst_den = p_yngst['density'],
             old_out_r   = p_old_out['rbins'], # forming beyond Rbr
             old_out_den = p_old_out['density'],
             mid_out_r   = p_mid_out['rbins'],
             mid_out_den = p_mid_out['density'],
             yng_out_r   = p_yng_out['rbins'],
             yng_out_den = p_yng_out['density'],
             yngst_out_r = p_yngst_out['rbins'],
             yngst_out_den = p_yngst_out['density'],
             old_in_r   = p_old_in['rbins'], # forming inside Rbr
             old_in_den = p_old_in['density'],
             mid_in_r   = p_mid_in['rbins'],
             mid_in_den = p_mid_in['density'],
             yng_in_r   = p_yng_in['rbins'],
             yng_in_den = p_yng_in['density'],
             yngst_in_r = p_yngst_in['rbins'],
             yngst_in_den = p_yngst_in['density'],
             old_rf_r   = p_old_rf['rbins'],
             old_rf_den = p_old_rf['density'],
             mid_rf_r   = p_mid_rf['rbins'],
             mid_rf_den = p_mid_rf['density'],
             yng_rf_r   = p_yng_rf['rbins'],
             yng_rf_den = p_yng_rf['density'],
             pg_r = pg['rbins'],
             pg_den = pg['density'])


