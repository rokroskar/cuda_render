



import pynbody
import pynbody.analysis.profile as profile
import numpy as np
import glob
import scipy as sp
import matplotlib.pylab as plt


if __name__ == '__main__':

#    s = pynbody.load('10/r61e11M_lam0.025.01000')
    s = pynbody.load('10/12M_hr.01000')
    s.physical_units()
    pynbody.analysis.angmom.faceon(s,disk_size='3 kpc')

    s.s['age'].convert_units('Gyr')

    old = np.where(s.s['age'] > 1.0)
    mid = np.where((s.s['age'] < 1.0) & (s.s['age'] > 0.1))
    yng = np.where(s.s['age'] < 0.1)

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

    plt.savefig('dens_plots.pdf', format='pdf')

    old_x = np.array(p_old['rbins'])
    mid_x = np.array(p_mid['rbins'])
    yng_x = np.array(p_yng['rbins'])

    prof_old = np.array(p_old['density'].in_units('Msol kpc**-2'))
    prof_mid = np.array(p_mid['density'].in_units('Msol kpc**-2'))
    prof_yng = np.array(p_yng['density'].in_units('Msol kpc**-2'))

    prof_old_i = np.array(p_old_i['density'].in_units('Msol kpc**-2'))
    prof_mid_i = np.array(p_mid_i['density'].in_units('Msol kpc**-2'))
    prof_yng_i = np.array(p_yng_i['density'].in_units('Msol kpc**-2'))


    np.savez('profiles', old_x=old_x, mid_x=mid_x, yng_x=yng_x, prof_old=prof_old, prof_mid=prof_mid, prof_yng=prof_yng, prof_old_i=prof_old_i, prof_mid_i = prof_mid_i, prof_yng_i = prof_yng_i)


    
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

    
               
