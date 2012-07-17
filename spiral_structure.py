"""
A set of routines for studying the properties of spiral structure


Rok Roskar 12/2010
Institute for Theoretical Physics, University of Zurich

"""

import pynbody
import pynbody.analysis.profile as profile
import numpy as np
import glob
import scipy as sp
import matplotlib.pylab as plt
import warnings
from scipy.stats import gaussian_kde as kde
from parallel_util import run_parallel, interruptible

#try: 
#    from IPython.parallel import Client
#    lv = Client().load_balanced_view()
#except : 
#    warnings.warn("Parallel interface not able to start -- using serial functions", RuntimeWarning)

def phi_bin(p):

    phi = np.arctan2(py,px)
    hist, binphi = np.histogram(phi, weights = pm, bins = 100)
    

def construct_R_phi_t_matrix(rmin, rmax, nrbins = 20, nphibins = 100, parent_dir = '.'):

    #flist = glob.glob(parent_dir+'/[1-9]/*.00??0.gz')
    #flist.sort()
    flist = glob.glob(parent_dir+'/fake*')
    
    
    nfiles = len(flist)

    R_phi_t_matrix_star = np.zeros((nfiles,nrbins,nphibins))
    R_phi_t_matrix_gas  = np.zeros((nfiles,nrbins,nphibins))

    times = np.zeros(nfiles)
    
    for i, file in enumerate(flist):
        print 'processing ' + file
        s = pynbody.load(file)
        times[i] = s.properties['a']
        pynbody.analysis.centre(s)
        ps = profile.Profile(s.star, max = rmax, min = rmin, nbins = nrbins)
        # select the interesting gas particles
        g = s.g[np.where((s.g['z'] < 1.0) & (s.g['temp'] < 1e5))]        
        pg = profile.Profile(g, max = rmax, min = rmin, nbins = nrbins)

        for rbin in range(nrbins):
            if ps.n[rbin] > 0:
                R_phi_t_matrix_star[i,rbin,:], binphi = np.histogram(np.arctan2(s.s['y'][ps.binind[rbin]],
                                                                                s.s['x'][ps.binind[rbin]]),
                                                                     weights = s.s['mass'][ps.binind[rbin]],
                                                                     bins = nphibins)
                
            if pg.n[rbin] > 0:
                R_phi_t_matrix_gas[i,rbin,:],  binphi = np.histogram(np.arctan2(g['y'][pg.binind[rbin]],
                                                                                g['x'][pg.binind[rbin]]),
                                                                     weights = g['mass'][pg.binind[rbin]],
                                                                     bins = nphibins)
                

    return R_phi_t_matrix_star, R_phi_t_matrix_gas, times, ps.r, 0.5*(binphi[1:]+binphi[:-1])
                                                                      

def calculate_fourier_coefficients(R_phi_t_matrix, phi):
    shape = R_phi_t_matrix.shape
    c = np.zeros((shape[0],shape[1],7),dtype=complex)

    for t in range(c.shape[0]):
        for r in range(c.shape[1]):
            for m in range(c.shape[2]):
                c[t,r,m] = np.sum(R_phi_t_matrix[t,r,:]*np.exp(1j*m*phi))/np.sum(R_phi_t_matrix[t,r,:])
                

    return c


def fourier_sequence_parallel(parent_dir='./', file_pattern = '/[1-9]/*.00???', block = True,
                              output=True, overwrite=False, nbins=50, max=15, min=0, cutoff_age=0.5, cutoff_mass = 0.1, ind = None, 
                              procs = int(pynbody.config['number_of_threads']), test = False):


    flist = glob.glob(parent_dir + file_pattern)
    flist.sort()

    res = run_parallel(fourier_single_file, flist, [output,overwrite,nbins, max, min, cutoff_age,cutoff_mass,ind],processes=procs, test = test)
    
    c_array = np.empty((len(flist),7,nbins),dtype=complex)
    mass = np.zeros([len(flist),nbins])
    times = np.zeros(len(flist))
    
    if block :
        bins = res[0][1]
        for i in range(len(flist)) : 
            c_array[i] = res[i][0]
            mass[i] = res[i][2]
            times[i] = res[i][3]

        np.savez(parent_dir+"/complete_fourier_fulldisk", c = c_array, t = times, r = bins, nbins = nbins, max = max, min = min,
                 file_pattern = file_pattern, cutoff_age = cutoff_age, mass = mass)
        
        return c_array, times, bins, res
    
    else : 
        return res, flist

def fourier_sequence(parent_dir='./', file_pattern = '/[1-9]/*.00???', nbins=50, **kwargs):

    cutoff_age = kwargs.get('cutoff_age', 0.5)
        
    flist = glob.glob(parent_dir+file_pattern)
    flist.sort()
    
    nfiles = len(flist)

    c_array = np.zeros((nfiles,7,nbins),dtype=complex)
    times = np.zeros(nfiles)
    mass = np.zeros([nfiles, nbins])
        
    for i, name  in enumerate(flist):
        print 'processing ' + name

        c_array[i], bins,  mass[i], times[i] = fourier_single_file(name, nbins = nbins, **kwargs)
        
        

    np.savez(parent_dir+"/complete_fourier_fulldisk", c = c_array, t = times, r = bins, nbins = nbins, max = max, min = min,
             file_pattern = file_pattern, cutoff_age = cutoff_age, mass = mass)

    #assert(len(np.unique(np.diff(times))) == 1)
    return c_array, times, bins

@interruptible
def fourier_single_file(a):
    import os 
    
    name, output, overwrite, nbins, max, min, cutoff_age, cutoff_mass, ind = a

    new = not os.path.isfile(name+".fourier.npz")
    
    if overwrite or (output and new):
        s = pynbody.load(name)
        pynbody.analysis.halo.center(s)
        if ind is None : 
            ind = np.where((s.s['age'] > cutoff_age) & (s.s['mass'] > cutoff_mass))[0]

        p = pynbody.analysis.profile.Profile(s.s[ind],nbins=nbins,max=max,min=min)
        c = p['fourier']['c']
        bins = p['rbins']
        mass = p['mass']
        time = s.properties['a']
    else:
        s = pynbody.load(name, only_header=True)
        fourier_file = np.load(name+'.fourier.npz')
        c = fourier_file['c']
        bins = fourier_file['bins']
        mass = fourier_file['mass']
        time = s.properties['a']
                
    if overwrite or (new and output):
        np.savez(name+".fourier", c = c,
                 mass=mass, den=p['density'], bins = p['rbins'], nbins = nbins, max = max, min = min,
                 cutoff_age = cutoff_age)

    return c, bins, mass, s.properties['a']


def get_fft(fourier_data,t1,t2,r,m=2, window=False) : 
    
    data = np.load(fourier_data)

    rbin = np.digitize([r],data['r'])

    ind = np.where((data['t'] >= t1) & (data['t'] <= t2))[0]

    # the optimal N for the FFT
    nfft = 2**np.ceil(np.log(len(ind))/np.log(2))

    # this is the sample we are interested in
    sample = data['c'][ind,m,rbin]

    # window the sample

    if window: 
        # Hanning
        x = np.arange(0,len(ind))
        win = 0.5*(1-np.cos(2*np.pi*x/(len(ind)-1)))
        sample *= win
    else : 
        win = np.ones(len(ind))

    ft = np.fft.fft(sample, n = nfft)
    fqs = np.fft.fftfreq(len(ft),data['t'][1]-data['t'][0])*2.0*np.pi/m

    # compute window normalization
    win_norm = len(ft)*np.sum(win**2)

    # compute the power spectrum

    psp = np.zeros(len(ft)/2)
    
    psp[0] = 1./win_norm*np.abs(ft[0])**2
    psp[-1] = 1./win_norm*np.abs(ft[len(ft)/2-1])**2
    for i in np.arange(1,len(ft)/2-1) : 
        psp[i] = 1./win_norm*(np.abs(ft[i])**2 + np.abs(ft[-i])**2)

    return ft, fqs, psp#, gauss, data['c'][ind,m,rbin]
    
def get_full_disk_fft(fourier_data, t1, t2, m=2) : 
    
    data = np.load(fourier_data)
    ind = np.where((data['t'] >= t1) & (data['t'] <= t2))[0]


    nfft = int(2**np.ceil(np.log(len(ind))/np.log(2)))
    pwr = np.zeros(nfft/2)
    ft = np.zeros((data['nbins'],nfft))

    #pwr_all = np.zeros(nfft/2, dtype='complex')
    
    x = np.arange(len(ind))
    gauss = 1.0*np.exp(-(x-nfft/2.0)**2/(nfft/4.0)**2)

    win = gauss

    ft = np.fft.fft(data['c'][ind,m,0]*gauss, n=nfft)
    fqs = np.fft.fftfreq(nfft,data['t'][1]-data['t'][0])*2.0*np.pi/m
    
    #for k in range(1,nfft/2) :
    #    pwr[k] = (abs(ft[k])**2 + abs(ft[-k])**2)/(nfft*gauss.sum()**2)
    
    # compute window normalization
    win_norm = len(ft)*np.sum(win**2)

    # compute the power spectrum

    psp = np.zeros(len(ft)/2)
    
    psp[0] = 1./win_norm*np.abs(ft[0])**2
    psp[-1] = 1./win_norm*np.abs(ft[len(ft)/2-1])**2
    for i in np.arange(1,len(ft)/2-1) : 
        psp[i] = 1./win_norm*(np.abs(ft[i])**2 + np.abs(ft[-i])**2)

    return ft, fqs, psp

def get_band_amplitude(fourier_data, t1, t2, f1, f2, r, m=2, window=True) :
    from scipy import signal

    ft, fqs, pwr = get_fft(fourier_data,t1,t2,r,window=window)
    
    # make a bandpass filter
    ind = np.where((fqs>=f1)&(fqs<=f2))[0]
    
    win = np.zeros(len(ft))
    win[ind] = 1.0

    t = np.load(fourier_data)['t']

    return np.fft.ifft(ft*win), t[np.digitize([t1],t)-1:np.digitize([t1],t)-1+len(ft)], ft, fqs, win

def get_fulldisk_band_amplitude(fourier_data,t1,t2,f1,f2,m=2,window=True):
    from scipy import signal

    ft, fqs, pwr = get_full_disk_fft(fourier_data,t1,t2)
    
    # make a bandpass filter
    ind = np.where((fqs>=f1)&(fqs<=f2))[0]
    
    win = np.zeros(len(ft))
    win[ind] = 1.0

    t = np.load(fourier_data)['t']

    return np.fft.ifft(ft*win), t[np.digitize([t1],t)-1:np.digitize([t1],t)-1+len(ft)], ft, fqs, win
    
def plot_mean_A2(dir='./',rmin=0,rmax=15): 
    
    data = np.load(dir+'/complete_fourier.npz')
    A2 = np.squeeze(np.abs(data['c'][:,2,:]))
    mass = data['mass']

    A2_mean = np.zeros(A2.shape[0])

    for i,t in enumerate(data['t']):
        rcut = np.where((data['r'] >= rmin) & (data['r'] <= rmax))
        A2_mean[i] = np.sum(A2[i,rcut]*mass[i,rcut],axis=1)/np.sum(mass[i,rcut])

    plt.plot(data['t'], A2_mean)


def rms_A2(dir, tmin, tmax, rmin, m):
    import scipy.interpolate as interp
    
    data = np.load(dir+'/complete_fourier.npz')

    tcut = np.where((data['t'] >= tmin) & (data['t'] <= tmax))

    breaks = np.genfromtxt(dir+'/fits.dat',unpack=True)
    break_fit = interp.interp1d(breaks[0],breaks[1])
    rmax = break_fit(data['t'][tcut])


    A2 = np.squeeze(np.abs(data['c'][tcut,2,:]))
    mass = np.squeeze(data['mass'][tcut,:])

    A2_mean = np.zeros(A2.shape[0])

    for i,t in enumerate(data['t'][tcut]):
        rcut = np.where((data['r'] >= rmin) & (data['r'] <= rmax[i]))
        A2_mean[i] = np.sum(A2[i,rcut]*mass[i,rcut],axis=1)/np.sum(mass[i,rcut])

    return data['t'][tcut], A2_mean, np.sqrt(np.mean(A2_mean)**2 + np.std(A2_mean)**2)

def rms_A2_allruns(rmin):
    dirs = ['12M_hr', 'stoch_test1', 'stoch_test2',
            '12M_hr_25pc_soft', '12M_hr_100pc_soft', '12M_hr_500pc_soft',
            '12M_hr_x0.5N', '12M_hr_x2N', '12M_hr_x4N',
            '12M_hr_sdm']

    labels = ['fid.', 'T1', 'T2',
              'S1', 'S3', 'S4',
              'R1', 'R3', 'R4',
              'SDM']

    labels2 = ['stoch', 'soft', 'N']

    rms_all = np.zeros(12)
    mean_all = np.zeros(12)
    
    #plt.figure()
    for i,dir in enumerate(dirs):
        t, A2_mean, A2_rms = rms_A2(dir, 2.0, 10.0, rmin, 2.0)
       # plt.plot(t,A2_mean,label=labels[i])
        mean_all[i] = np.mean(A2_mean)
        rms_all[i] = np.std(A2_mean)
    #plt.legend()

    plt.figure()

    color = ['ro', 'go', 'bo']

    for i in np.arange(0,3):
        for j in np.arange(0,3):
            #plt.plot(mean_all.reshape([4,3]).transpose()[:,i], 'o', label = labels2[i])
            plt.errorbar(i+j*.1-0.12,mean_all.reshape([4,3]).transpose()[j,i],
                         yerr = rms_all.reshape([4,3]).transpose()[j,i], fmt=color[i],lw=2)
            plt.annotate(labels[j+i*3], [i+0.1*j-0.15,0.005],fontsize=14)
        
    plt.xlim([-.2,2.3])
    plt.ylabel(r'$<A_2>$',fontsize=20,fontweight='bold')
    plt.xticks([0,1,2],['stochasticity', 'softening', r'$N_{part}$'])
    
    ax = plt.gca()
    fontsize=20

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    return rms_all.reshape([4,3])

def dedj(s1, s2, patspeeds, filelist = None):
    import scipy.interpolate as interpol
    import mpl_toolkits.axes_grid.parasite_axes as pa
    import matplotlib.transforms as transforms
    from pynbody.array import SimArray
    from pynbody.analysis.profile import Profile

    # center and align
    for s in [s1, s2] :
        cen = pynbody.analysis.halo.center(s, retcen=True)
        if cen.any() > 1e-5 :
            pynbody.analysis.angmom.faceon(s, cen = cen, disk_size='3 kpc')
        s.s['jz'] = s.s['j'][:,2]
    #    s.physical_units()

    s1.s['dj'] = s2.s['jz'][:len(s1.s)] - s1.s['jz']
    s1.s['de'] = s2.s['te'][:len(s1.s)] - s1.s['te']

    
    # determine corotation resonance locations
    p1 = Profile(s1, max=15, nbins=20, load_from_file=True)
    p2 = Profile(s2, max=15, nbins=20, load_from_file=True)
    cr_r  = SimArray(interpol.interp1d(p1['omega'][::-1], 
                                       p1['rbins'][::-1])(patspeeds),p1['rbins'].units)
    cr_jz = SimArray(interpol.interp1d(p1['rbins'], p1['j_circ'])(cr_r),p1['j_circ'].units)
    cr_e  = SimArray(interpol.interp1d(p1['rbins'], p1['E_circ'])(cr_r),p1['E_circ'].units)

   # cr_r.convert_units(s1['pos'].units)
   # cr_jz.convert_units(s1['j'].units)
   # cr_e.convert_untis(s1['te'].units)

    colors1 = ['bo', 'go', 'ro', 'yo']
    colors2 = ['bx', 'gx', 'rx', 'yx']
    
    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(221)
    ax = [pa.SubplotHost(fig,222), pa.SubplotHost(fig,223), pa.SubplotHost(fig,224)]

         # 43 km/s/kpc pattern
    one = (pynbody.filt.BandPass('jz', 1250, 1800) & pynbody.filt.LowPass('dj', -300)) | \
        (pynbody.filt.BandPass('jz', 750, 1250) & pynbody.filt.HighPass('dj', 300))
    
    # 68 km/s/kpc pattern
    two = (pynbody.filt.BandPass('jz', 950, 1100) & pynbody.filt.LowPass('dj', -250)) | \
        (pynbody.filt.BandPass('jz', 700, 850) & pynbody.filt.HighPass('dj', 250))

    # 20 km/s/kpc pattern
    three = (pynbody.filt.HighPass('jz', 2200) & pynbody.filt.HighPass('dj', 300)) | \
        (pynbody.filt.HighPass('jz',2300) & pynbody.filt.LowPass('dj',-300))

    filts = [one, two, three]

    for i, ps in enumerate(patspeeds) :
        cr_in_jz = cr_jz[i] - 100
        cr_out_jz = cr_jz[i] + 100
        cr_in_e = interpol.interp1d(p1['j_circ'],p1['E_circ'])(cr_in_jz)
        cr_out_e = interpol.interp1d(p1['j_circ'],p1['E_circ'])(cr_out_jz)
        
        #ind = np.squeeze(np.where((np.abs(s1.s['j'][:,2] - cr_jz[i]) < 0.01*cr_jz[i]) &
        #                          (np.abs(s1.s['te'] -     cr_e[i])  < np.abs(0.01*cr_e[i])) &
        #                          (np.abs(s2.s['j'][:len(s1.s),2] - s1.s['j'][:,2]) > 200)))

        ind = np.squeeze(np.where((((np.abs(s1.s['j'][:,2] - cr_in_jz) < 0.01*cr_in_jz) &
                                  (np.abs(s1.s['te'] -     cr_in_e)  < np.abs(0.01*cr_in_e))) |
                                 ((np.abs(s1.s['j'][:,2] - cr_out_jz) < 0.01*cr_out_jz) &
                                 (np.abs(s1.s['te'] -     cr_out_e)  < np.abs(0.01*cr_out_e)))) &
                                 (np.abs(s2.s['j'][:len(s1.s),2] - s1.s['j'][:,2]) > 200)))

        #ind = filts[i].where(s1.s)[0]

        rand_ind = np.round(np.random.rand(np.min([len(ind),100]))*(len(ind)-1))
        dE = s2.s[ind]['te']-s1.s[ind]['te']
        dJ = s2.s[ind]['j'][:,2] - s1.s[ind]['j'][:,2]

#        n, bins,patches = ax1.hist(dE/dJ, #weights=np.abs(dJ),
#                                   bins=100, range=[0,100], alpha = .5, normed=True) 
        omega = np.linspace(-40,40,1000)
        kd = kde(dE/dJ-ps)

        ax1.plot(omega,kd(omega))

        for j, k in enumerate(rand_ind) :
            k = np.int(k)
            ax[i].plot([s2.s['j'][ind[k],2], s1.s['j'][ind[k],2]],
                       [s2.s['te'][ind[k]],  s1.s['te'][ind[k]]],'black', alpha = 0.4)
#            ax[i].plot([s2.s['j'][ind[k],2], s1.s['j'][ind[k],2]],
#                       [s2.s['te'][ind[k]],  s1.s['te'][ind[k]]], colors[i], alpha = 0.7)
            
            ax[i].plot(s1.s['j'][ind[k],2], s1.s['te'][ind[k]], colors1[i], alpha = 0.4)
            ax[i].plot(s2.s['j'][ind[k],2], s2.s['te'][ind[k]], colors2[i], alpha = 0.7)

            if filelist is not None : 
                xs = np.zeros(len(filelist))
                ys = np.zeros(len(filelist))
                for n,s in enumerate(filelist) : 
                    xs[n] = s.s['j'][ind[k],2]
                    ys[n] = s.s['te'][ind[k]]

                ax[i].plot(xs,ys,colors[i],alpha=.3)

            #ax[i].arrow(s1.s['j'][ind[k],2], s1.s['te'][ind[k]], s1.s['dj'][ind[k]], s1.s['de'][ind[k]], alpha=.4, edgecolor='black', width=5)
            

        xlim = ax[i].get_xlim()
        ylim = ax[i].get_ylim()
        ax[i].plot(p1['j_circ'], p1['E_circ'], scalex=False, scaley=False, linewidth=2, color = 'orange')
        ax[i].plot(cr_jz[i], cr_e[i], 'ro')
        ax[i].plot(p2['j_circ'], p2['E_circ'], '--', scalex=False, scaley=False, linewidth=2, color = 'orange')
        #ax[i].set_xlim(xlim)
        #ax[i].set_ylim(ylim)
        ax[i].set_xlabel('$J_z$')
        ax[i].set_ylabel('$E$')
        ax[i].ticklabel_format(style='sci', axis='both', scilimits=(0,0))

        # draw the E = Omega_p * J tangent lines
#        for j, op in enumerate(patspeeds) :
        ax[i].plot([xlim[0], xlim[1]],[cr_e[i]-ps*(cr_jz[i]-xlim[0]), 
                                       cr_e[i]-ps*(cr_jz[i]-xlim[1])], colors1[i][0],linewidth=2)
        ax[i].plot([xlim[0], xlim[1]],[cr_e[i]-(ps-5)*(cr_jz[i]-xlim[0]), 
                                       cr_e[i]-(ps-5)*(cr_jz[i]-xlim[1])], colors1[i][0],linewidth=2, linestyle='dashed')
        ax[i].plot([xlim[0], xlim[1]],[cr_e[i]-(ps+5)*(cr_jz[i]-xlim[0]), 
                                       cr_e[i]-(ps+5)*(cr_jz[i]-xlim[1])], colors1[i][0],linewidth=2, linestyle='dashed')

        ax[i].set_xlim(xlim)
        ax[i].set_ylim(ylim)

        fig.add_subplot(ax[i])

        # add the twin axis to display delta_E and delta_J from the initial selection
        trans = transforms.Affine2D().scale(cr_jz[i], cr_e[i])
        axt = ax[i].twin(trans)
        axt.ticklabel_format(style='sci', axis='both', scilimits=(-100,100))

    ax1.set_xlabel('$dE/dJ - \Omega_p$')
    
    # save the profile because v_circ and pot take forever to compute
    p1.write()
    p2.write()

                        
#    s1 = s1_t.s
#    s2 = s2_t.s[0:len(s1)]


#    dE = s2.s['E'] - s1.s['E']
#    dJ = s2.s['j'][:,2] - s1.s['j'][:,2]

#    colors = ['or', 'ob', 'og']

#    for i,omega in enumerate(patspeeds) :
#        ind=np.squeeze(np.where((np.abs(dE/dJ - omega) < 1.0)&(dJ > 200)))
#        rand_ind = np.random.rand(100)*len(ind)
#        for j, k in enumerate(rand_ind) :
#            plt.plot([s2.s['j'][ind[k],2], s1.s['j'][ind[k],2]],
#                     [s2.s['E'][ind[k]], s1.s['E'][ind[k]]],'black', alpha = 0.4)
#            plt.plot([s2.s['j'][ind[k],2], s1.s['j'][ind[k],2]],
#                     [s2.s['E'][ind[k]], s1.s['E'][ind[k]]], colors[i], alpha = 0.4)

        


def migrators_distr(s1,s2,s3, patspeed): 
    """ s1 and s2 are the endpoints of the interval, s3 is in the middle """
    from scipy import interpolate as interpol
    from pynbody.array import SimArray
    from pynbody.analysis.profile import Profile 

    for s in [s1,s2,s3] : 
        pynbody.analysis.angmom.faceon(s)

    s1.s['dj'] = s2.s['jz'][0:len(s1.s)] - s1.s['jz']

    dj = np.abs(s1.s['dj'])
    dj.sort()
    jcut = dj[len(dj)*.98]
    print jcut

    outward = np.where(s1.s['dj'] > jcut)[0]
    inward  = np.where(s1.s['dj'] < -jcut)[0]

    hout,xout,yout = pynbody.plot.generic.gauss_kde(s3.s['x'][outward],s3.s['y'][outward], 
                                                    make_plot=False, 
                                                    x_range = [-15,15], 
                                                    y_range = [-15,15])

    hin,xin,yin = pynbody.plot.generic.gauss_kde(s3.s['x'][inward],s3.s['y'][inward], 
                                                 make_plot=False, 
                                                 x_range = [-15,15],
                                                 y_range = [-15,15])

    
    
    old = pynbody.filt.HighPass('age', 0.5)
    disk = pynbody.filt.Disc(15, 0.5)
    
    # get the CR radius

    p = Profile(s3, max=15, nbins=20, load_from_file=True)
    cr_r  = SimArray(interpol.interp1d(p['omega'][::-1], 
                                       p['rbins'][::-1])(patspeed),p['rbins'].units)
    

    plt.figure()
    ax = plt.gca()
    #ax.spines['left'].set_position(('data', 0))
    #ax.spines['bottom'].set_position(('data', 0))
    #ax.spines['right'].set_position(('data', -1))
    #ax.spines['top'].set_position(('data',  -1))
    ax.add_patch(plt.Circle((0,0),radius=cr_r,color='green',
                            fill=False, linestyle='dashed', linewidth=2))

    plt.contour(xout,yout,hout,np.linspace(1,50,5),colors='blue',label='outward',linewidths=2)
    plt.contour(xin,yin,hin,np.linspace(1,50,5),colors='red',linestyles='dashed',label='inward',linewidths=2)

    #h,x,y = pynbody.plot.generic.gauss_kde(s3.s['x'][0:len(s1.s)],s3.s['y'][0:len(s1.s)], 
    #                                       weights = s1.s['dj'].in_units('kpc km s**-1'),
    #                                       x_range = [-15,15],
    #                                       y_range = [-15,15],
    #                                       make_plot=False)

#    im = plt.imshow(h,cmap=plt.cm.RdGy,origin='lower left', extent=[-15,15,-15,15],
#                    vmin=-5e4,vmax=5e4)
#    plt.colorbar(im,format="%.2e").set_label('$\\mathrm{kpc~km~s^{-1}}$')
 
    pynbody.plot.fourier_map(s3.s[old],mmin=2,mmax=5,rmax=15,nbins=50,nphi=1000,
                             linewidths=2, colors='black')
    

    
    ax.set_aspect('equal')

    fontsize=20

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')

#    for line in ax.xaxis.get_ticklines()+ax.yaxis.get_ticklines():
#        line.set_markeredgewidth(3)

    plt.xlabel('$x~\\mathrm{[kpc]}$',fontsize=fontsize,fontweight='bold')
    plt.ylabel('$y~\\mathrm{[kpc]}$',fontsize=fontsize,fontweight='bold')

    plt.xlim(-10,10)
    plt.ylim(-10,10)

    #plt.figure()

    #kde_out = kde(s3.s['az'][outward])
    #kde_in  = kde(s3.s['az'][inward])

    #phis = np.linspace(-np.pi,np.pi,1000)
    
    #plt.plot(phis,kde_out(phis), color='blue', label = 'outward')
    #plt.plot(phis,kde_in(phis),  color='red', label = 'inward')
    #plt.xlabel('$\phi$ [rad]')
    #plt.legend()



def make_fourier_map(flist,nrow,ncol,nbins=50,nmin=1000,nphi=1000,mmin=1,mmax=5,rmax=10,levels=[-.5,-.3,-.2,-.1,-.05,-.01,.01,.05,.1,.2,.3,.5]) : 

    from matplotlib import cm
    from mpl_toolkits.axes_grid1 import ImageGrid

    fig = plt.figure(figsize=(5*ncol,5*nrow))

    grid = ImageGrid(fig,111,nrows_ncols=(nrow,ncol),
                     axes_pad=.1,
                     label_mode='1')
    
    for i,f in enumerate(flist): 
        if type(f) is not str : 
            s = f
        else : 
            s = pynbody.load(f)
            pynbody.analysis.angmom.faceon(s)
        
        if f._filename[1] != '/' : 
            ann = f._filename[0:2]
        else : 
            ann = f._filename[0]

        ax = grid[i]
        pynbody.plot.image(s.s,threaded=True,width=20,vmin=6.5,vmax=10,cmap=cm.binary_r,
                           units='Msol kpc^-2',colorbar=False,subplot=ax)
        if levels is not None: 
            pynbody.plot.fourier_map(s.s,nbins,nmin,nphi,mmin,mmax,rmax,levels,
                                     subplot=ax,colors = 'red',linewidths=.5)
        ax.annotate('$\\mathrm{%s~Gyr}$'%ann,(-9.5,8),fontsize=20,fontweight='bold',color='white')

#    plt.colorbar(ax=fig,cax=ax)
        

def make_ddj_fig(s1,s2,patspeeds) :

    pynbody.analysis.angmom.faceon(s1)
    pynbody.analysis.angmom.faceon(s2)
    
    s1.s['dj'] = s2.s['jz'][0:len(s1.s)] - s1.s['jz']
    
    pynbody.plot.generic.gauss_kde(s1.s['jz'],s1.s['dj'],weights=s1.s['mass'],scalemin=1)
    
    plt.xlabel('$j_z$')
    plt.ylabel('$\Delta j_z$')

    p = pynbody.analysis.profile.Profile(s1,max=15,bins=30,type='log',min=1e-3)

    colors = ['r','y','g']

    for i,pat in enumerate(patspeeds) : 
        cr_jz = get_crs(p,pat)[1]
    
        plt.plot([cr_jz,cr_jz],[-1000,1000],color=colors[i],linewidth=2)

    

def get_crs(p,patspeed):
    import scipy.interpolate as interpol
    from pynbody.array import SimArray

    cr_r  = SimArray(interpol.interp1d(p['omega'][::-1], 
                                       p['rbins'][::-1])(patspeed),p['rbins'].units)
    cr_jz = SimArray(interpol.interp1d(p['rbins'], p['j_circ'])(cr_r),p['j_circ'].units)
    cr_e  = SimArray(interpol.interp1d(p['rbins'], p['E_circ'])(cr_r),p['E_circ'].units)

    return cr_r, cr_jz, cr_e


if __name__ == '__main__':

    import getopt, sys, os

    try:
        opts, args = getopt.getopt(sys.argv[1:], "owf:a:p:m:", ["filepattern"])
    except:
        print 'bad options'
        sys.exit(2)

    annotate = None
    figure_name = None
    output = False
    overwrite = False
    file_pattern = "/[1-9]/*.00???"
    m = 2
    
    for opt, arg in opts :
        if opt in ("-f", "--filepattern"):
            file_pattern = arg
        elif opt == "-o" :
            output = True
        elif opt == "-w" :
            overwrite = True
        elif opt == '-a':
            annotate = arg
            figure_name = annotate
        elif opt == '-p' :
            figure_name = arg
        elif opt == '-m' :
            m = np.int(arg)

    
    
    if not os.path.isfile("complete_fourier.npz") :
        c, t, r = fourier_sequence(output=output, overwrite = overwrite, file_pattern = file_pattern)
    else :
        print "loading fourier data from \"complete_fourier.npz\""
        data = np.load("complete_fourier.npz")

        print "constructed with:"
        for field in ['min','max','nbins','file_pattern','cutoff_age'] :
            print field + ' = ' + np.str(data[field])
            
        c = data['c']
        t = data['t']
        r = data['r']
      

