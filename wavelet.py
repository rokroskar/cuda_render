import numpy as np
import cwt
import matplotlib.pylab as plt

def plot_wavelet_spectrum(c, times, file_pattern = '/[1-9]/*.00???.gz', rbin = 5, m = 2, pad = True):
    import cwt
    from scipy.special import erf
    
    if pad:
        cpad = np.append(c[:,m,rbin],np.zeros(2**np.ceil(np.log(len(times))/np.log(2))-len(times)))
    else:
        cpad = c[:,m,rbin]
        
    wav = cwt.Morlet(cpad, 1, 64, scaling='log')
    scales = wav.getscales()*wav.fourierwl*0.01
    freqs = 2*np.pi/scales/m
    pwr = np.sqrt(wav.getpower()[:,0:len(times)])

    # normalize the power (from Nener et al. 1999, "infrared physics & technology")
    cn = (4*np.pi**.5/(1+erf(5)))**.5
    for i in range(len(times)):
        pwr[:,i] *= 2/cn/np.sqrt(wav.getscales())

    # make the figure

    plt.contourf(times,freqs,pwr,np.linspace(0.05,0.6,10))
    plt.xlim(times[0],times[-1])
    plt.ylim(freqs[-1],freqs[0])

    
    #plt.colorbar()
    
    return c, times, wav, scales, freqs, pwr

def cwt_signal(s, x, largestscale = 1, notes = 0, order = 2, scaling = 'linear', pad = True) : 
    from scipy.special import erf

    if pad : 
        spad = np.append(s,np.zeros(2**np.ceil(np.log(len(x))/np.log(2))-len(x)))
    else :
        spad = s
        
    wav = cwt.Morlet(spad, largestscale, notes, order, scaling)
    scales = wav.getscales()*wav.fourierwl*0.01
    #freqs = 2*np.pi/scales/m
    pwr = np.sqrt(wav.getpower()[:,0:len(x)])

    # normalize the power (from Nener et al. 1999, "infrared physics & technology")
    cn = (4*np.pi**.5/(1+erf(5)))**.5
    for i in range(len(x)):
        pwr[:,i] *= 2/cn/np.sqrt(wav.getscales())

    return wav, pwr, scales

def plot_fulldisk_wavelet_spectrum(c, times, rbins, m = 2, pad = True, annotate = '', figure_name = ''):
    import cwt
    from scipy.special import erf


    if figure_name == '': figure_name = annotate

    

    freqs, pwr = get_fulldisk_wavelet_spectrum(c, len(rbins), len(times), m = m)
    
    # make the figure

    
    fig = plt.figure(1, (3.3, 3.3))
    ax = plt.axes([0.21,0.15,0.75,0.8])
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(15)
            
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(15)

    for line in ax.yaxis.get_ticklines():
        line.set_color('white')
        line.set_markeredgewidth(1)

    for line in ax.xaxis.get_ticklines():
        line.set_color('white')
        line.set_markeredgewidth(1)

    plt.ylabel(r'$\Omega$ [km/s/kpc]', fontsize = 15)
    plt.xlabel('time [Gyr]', fontsize = 15)

    fig.canvas.draw()
    
    plt.contourf(times,freqs,pwr,np.logspace(-1,np.log10(20),20))
    plt.contour(times,freqs,pwr,np.logspace(-1,np.log10(20),20),linewidth=1)
    plt.xlim(times[0],times[-1])
    plt.xlabel('time [Gyr]')
    #plt.xticks(size=15)
    plt.ylim(freqs[-1],100)
    plt.ylabel(r'$\Omega$ [km/s/kpc]')
    #plt.xticks(size=15)
    
    
    if annotate is not None:
        plt.annotate(annotate, [1.5,90], color = 'white', size = 40, )

    fig = plt.gcf()

    plt.savefig(figure_name+'fourier_fulldisk.eps', format='eps')
    
    return freqs, pwr

#def inv_fourier(sim, nbins=100, rmax=15, ncell=100,mmin=2,mmax=2,fourrmax=15) : 

def get_fulldisk_wavelet_spectrum(c, nrbins, ntimes, m=2, pad = True) :
    import cwt
    from scipy.special import erf

    for i in np.arange(nrbins) :
        if pad:
            cpad = np.append(c[:,m,i],np.zeros(2**np.ceil(np.log(ntimes)/np.log(2))-ntimes))
        else:
            cpad = c[:,m,i]


        wav = cwt.Morlet(cpad, 1, 64, scaling='log')
        scales = wav.getscales()*wav.fourierwl*0.01
        freqs = 2*np.pi/scales/m
        
        temp_pwr = np.sqrt(wav.getpower()[:,0:ntimes])

        # normalize the power (from Nener et al. 1999, "infrared physics & technology")
        cn = (4*np.pi**.5/(1+erf(5)))**.5
        for j in range(ntimes):
            temp_pwr[:,j] *= 2/cn/np.sqrt(wav.getscales())

        if i == 0 :
            pwr = temp_pwr.copy()
        else :
            pwr += temp_pwr

    return freqs, pwr

def plot_multiple_radii(c, t):
    # make the figure
    
    fig = plt.figure(figsize=(10,10))
    plt.subplots_adjust(wspace=0.05,hspace=0.05)    
    rbins = np.linspace(0,15,21)
    rbins = .5*(rbins[:-1]+rbins[1:])

    for i, j in enumerate([3,5,7,9,11,13,15,17,19]):
        ax = plt.subplot(3,3,i+1)
        plot_wavelet_spectrum(c,t,rbin=j)

        if (i+1) not in [1,4,7]:
            ax.yaxis.set_ticklabels("")
        else:
            plt.ylabel(r'$\Omega$ [km/s/kpc]')
        if i+1 < 7:
            ax.xaxis.set_ticklabels("")
        else:
            plt.xlabel('time [Gyr]')
            
        plt.text(2,5,'r=%.1f'% rbins[j] + ' kpc',
                 bbox=dict(facecolor='white',alpha=0.7))

    #plt.colorbar(format="%.1f")


def plot_multiple_systems(dirlist, names):
    import os
    
    fig = plt.figure(figsize=(10,10))
    plt.subplots_adjust(wspace=0.01,hspace=0.01)

    for i,dir in enumerate(dirlist):
        if not os.path.isfile(dir+"/complete_fourier.npz") :
            "generating fourier data for " + dir 
            c, t, r = fourier_sequence(output=True, overwrite = False)
        else :
            print "loading fourier data from " + dir + "/complete_fourier.npz"
            data = np.load(dir+"/complete_fourier.npz")
                
            print "constructed with:"
            for field in ['min','max','nbins','file_pattern','cutoff_age'] :
                print field + ' = ' + np.str(data[field])
                    
            c = data['c']
            t = data['t']
            r = data['r']
        freqs, pwr = get_fulldisk_wavelet_spectrum(c, len(r), len(t))
        ax = plt.subplot(3,3,i+1)

        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(15)
            
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(15)

        for line in ax.yaxis.get_ticklines():
            line.set_color('white')
            line.set_markeredgewidth(1)

        for line in ax.xaxis.get_ticklines():
            line.set_color('white')
            line.set_markeredgewidth(1)
                

        if (i+1) not in [1,4,7]:
            ax.yaxis.set_ticklabels("")
        else:
            plt.ylabel(r'$\Omega$ [km/s/kpc]', fontsize = 15)
        if i+1 < 7:
            ax.xaxis.set_ticklabels("")
        else:
            plt.xlabel('time [Gyr]', fontsize = 15)

        fig.canvas.draw()
            
        plt.contourf(t,freqs,pwr,np.logspace(-1,np.log10(20),20))
        plt.contour(t,freqs,pwr,np.logspace(-1,np.log10(20),20),linewidth=1)
        plt.xlim(t[0],t[-1])
        plt.ylim(freqs[-1],120)
        plt.annotate(names[i], [1.5,100], color = 'white', size = 20, )
        
        

        
def get_band_vs_radius(c,t,r,m,band):
    import cwt
    from scipy.special import erf

    spec = np.zeros((len(r),len(t)))
    for rbin in range(len(r)):
        cpad = np.append(c[:,m,rbin],np.zeros(2**np.ceil(np.log(len(t))/np.log(2))-len(t)))
        wav = cwt.Morlet(cpad,1,64,scaling='log')
        
        scales = wav.getscales()*wav.fourierwl*0.01
        freqs = 2*np.pi/scales/m
        pwr = np.sqrt(wav.getpower()[:,0:len(t)])
        
        # normalize the power (from Nener et al. 1999, "infrared physics & technology")
        cn = (4*np.pi**.5/(1+erf(5)))**.5
        for i in range(len(t)):
            pwr[:,i] *= 2/cn/np.sqrt(wav.getscales())


        band_ind = np.array(np.where((freqs >= band[0]) & (freqs <= band[1]))).flatten()
        spec[rbin,:] = pwr[band_ind,:].sum(axis=0)

    return spec

def plot_band_vs_radius(c,t,r,m,band):
    spec = get_band_vs_radius(c,t,r,m,band)
    
    plt.figure()
    plt.contourf(t,r,spec,10)
    plt.title(str(band[0]) + ' - ' + str(band[1]) + ' km/s/kpc')
    plt.xlabel('time [Gyr]')
    plt.ylabel('R [kpc]')
    plt.ylim(r.min(),r.max())
    plt.colorbar()
    

