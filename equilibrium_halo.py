#
#
# generate a temperature profile for the gas to put it in hydrostatic
# equilibrium with the underlying mass distribution
#
# partially based on stablehalo by Mastropierto, Kaufmann and Stinson
# -- see Kaufmann et al. 2006
#
#

import pynbody
import numpy as np
from pynbody import units

def get_mu(T, elecPres) : 
    HIIionE =13.5984*1.60217733e-12
    HeIIionE =24.5874*1.60217733e-12
    HeIIIionE =54.417760*1.60217733e-12
    Htot =0.9
    Hetot =0.1

    fracHIIHI = calc_saha(T, 1., 2., elecPres, HIIionE);
    HII = fracHIIHI/(fracHIIHI+1.);
    HI = 1./(fracHIIHI+1.);
    fracHe2n3HeI = calc_saha(T, 3., 4., elecPres, HeIIionE);
    He2n3 = fracHe2n3HeI/(fracHe2n3HeI+1.);
    HeI = 1./(fracHe2n3HeI+1.);
    fracHeIIIHeII = calc_saha(T, 2., 3., elecPres, HeIIIionE);
    HeIII = He2n3*fracHeIIIHeII/(fracHeIIIHeII+1.);
    HeII = He2n3/(fracHeIIIHeII+1.);
    return HII*Htot/2.+HI*Htot+4.*Hetot*(HeIII/3.+HeII/2.+HeI);


def calc_saha(T, partTop, partBottom, elecPres, IonEnergy) : 
    kB = 1.38066e-16 # erg kelvin^-1 
    TwoPimekBonhh = 17998807946.6
    return 2.*kB*T/elecPres*partTop/partBottom*(TwoPimekBonhh*T)**1.5*np.exp(-IonEnergy/kB/T)


def get_temp_profile(s, ngrid) : 
    # generate profiles
    p  = pynbody.analysis.profile.Profile(s,ndim=3,type = 'log',nbins=ngrid)
    pg = pynbody.analysis.profile.Profile(s.g,ndim=3,type = 'log', nbins=ngrid)
    
    # get the first round of temperatures
    Tgrid = pynbody.array.SimArray(np.zeros(len(p)), 'K')
    Pgrid = pynbody.array.SimArray(np.zeros(len(p)), 'g cm^-1 s^-2')
    integrand = ((-pg['vt']/10.)**2/pg['rbins'] +
                 units.G*p['mass_enc']/p['rbins']**2)*pg['density']
    
    mu = 1.0
    for i in range(len(p)) : 
        Pgrid[i] = (np.trapz(integrand[i:],x=p['rbins'][i:])).in_units('g cm^-1 s^-2') 
                    
        
    Tgrid = (Pgrid/pg['density']/units.k*mu*units.m_p).in_units('K')

    return p['rbins'], Tgrid, Pgrid

def iterate_temp(temp,pres) :
    
    Tdiff = 1.0
    Torig = temp
    Told = temp
    mu = 1.0

    while(Tdiff > 1e-3) : 
        munew = get_mu(Told,pres)
        Tnew = munew*Torig
        Thalf = (Told + Tnew)/2.0
        munew = get_mu(Thalf, pres)
        Tguess = munew*Torig
        if(Tguess > Thalf) : 
            if (Told < Tnew): 
                Told = Thalf
            else : 
                Tnew = Thalf
        else : 
            if (Told > Tnew) : 
                Told = Thalf
            else : 
                Tnew = Thalf
        
        Tdiff = np.abs(Told - Tnew)/Tnew

        print 'old temp = %e, new temp = %e, mu = %e, diff = %e'%(Told,Tnew,munew,Tdiff)
    return Tnew

def set_temperatures(s, ngrid = 100) : 
    from smooth import smooth

    r, t, p = get_temp_profile(s,ngrid)


    tcorr = pynbody.array.SimArray(np.zeros(len(t)),'K')
    tcorr = smooth(t,10,'hanning')

    good = ~(np.isinf(tcorr) | np.isnan(tcorr))

    r=r[good]
    tcorr=tcorr[good]
    p=p[good]

    for i in range(len(tcorr)) : 
        tcorr[i] = iterate_temp(tcorr[i],p[i])

    good = ~(np.isinf(tcorr) | np.isnan(tcorr))
    
    s.g['temp'] = np.interp(s.g['r'],r[good],tcorr[good])
    
    return r[good],tcorr[good]

    
    
