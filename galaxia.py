import ebf
import pynbody
import os
import numpy as np
import matplotlib.pylab as plt

def make_galaxia_input(sim, run_enbid=False) :

    # load and align the snapshot
    
    s = pynbody.load(sim)
    pynbody.analysis.angmom.faceon(s)
    
    filt = pynbody.filt.Disc(15,2.5)
#    s=s[filt]
    filename = s.filename

      
    # set low metalicities to some reasonable value

    s.s['mets'] = s.s['feh']
    s.s['alpha'] = 0.0
    bad = np.where(s.s['feh'] < -5)[0]
    s.s[bad]['mets'] = -5.0

    s.s['mets']+=.1
    
    # shift metallicities and ages around in a random way to avoid identical values
    s.s['mets'] += np.random.normal(0,.001,len(s.s))
    s.s['ages'] = s.s['age'] + np.random.normal(0,1e-5,len(s.s))
    
    try: 
        assert((len(np.unique(s.s['mets'])) == len(s.s)) & 
               (len(np.unique(s.s['ages'])) == len(s.s)) & 
               (len(np.unique(s.s['x'])) == len(s.s)) & 
               (len(np.unique(s.s['y'])) == len(s.s)) & 
               (len(np.unique(s.s['z'])) == len(s.s)) & 
               (len(np.unique(s.s['vx'])) == len(s.s)) & 
               (len(np.unique(s.s['vy'])) == len(s.s)) & 
               (len(np.unique(s.s['vz'])) == len(s.s))) 
           
    except:
        print "unique mets = %s and ages = %s"%(len(np.unique(s.s['mets'])), len(np.unique(s.s['ages'])))
        raise AssertionError("Need unique values")
    # make the pos array

    pos = np.array([s.s['x'],s.s['y'],s.s['z'],s.s['vx'],s.s['vy'],s.s['vz'],s.s['ages'],s.s['mets']]).T

    # make the enbid file
    
    ebf.write(filename+'_galaxia.ebf', '/pos', pos, 'w')
    ebf.write(filename+'_galaxia.ebf', '/pos3', s.s['pos'], 'a')
    ebf.write(filename+'_galaxia.ebf', '/vel3', s.s['vel'], 'a')
    ebf.write(filename+'_galaxia.ebf', '/Mass', s.s['mass'].in_units('Msol'), 'a')
    ebf.write(filename+'_galaxia.ebf', '/feh', s.s['mets'], 'a')
    ebf.write(filename+'_galaxia.ebf', '/alpha', s.s['alpha'], 'a')
    ebf.write(filename+'_galaxia.ebf', '/age', s.s['ages'], 'a')
    ebf.write(filename+'_galaxia.ebf', '/id', s.s['iord'], 'a')
    
    


    if run_enbid:

        # remove 's' to save space

        del(s)

        # run enbid

#        os.system('~/bin/enbid -dmc --dim=3 --ngb=64 --dsuffix=_d3n64 %s_galaxia.ebf'%filename)
        os.system('~/bin/enbid -dmc --dim=6 --ngb=64 --dsuffix=_d6n64 %s_galaxia.ebf'%filename)
#        os.system('~/bin/enbid -dm --gmetric=1 --dim=8 --ngb=64 --dsuffix=_d8n64 %s_galaxia.ebf'%filename)




    

def compare_run_to_model(sim_gal, mod_gal) : 
    from scipy.stats import gaussian_kde as kde

    sim = ebf.read(sim_gal)
    mod = ebf.read(mod_gal)

    # two columns -- simulation on left, galaxia model on right

    hists = ['ubv_v','age','feh','rad']
    ranges = [[-6,10],[6,10.5],[-2,.5],[0,2]]
    labels = ['sim','model']

    f, axs = plt.subplots(len(hists),1,figsize=(7,len(hists)*2.5))
        
    # histograms

    for i,h in enumerate(hists) : 
        for j,x in enumerate([sim,mod]) :
            ax = axs[i]
            ax.hist(x[h],bins=50,normed=True,histtype='step',range=ranges[i],label=labels[j])
#            k = kde(x[h])
#            xs = np.linspace(ranges[i][0],ranges[i][1],200)
#            ax.plot(xs,k(xs),label=labels[j])
            ax.set_xlabel(r'%s'%h)
    

    axs[1].set_ylim(0,2.0)
    plt.subplots_adjust(hspace=.5)
    axs[0].legend(prop=dict(size=10))


    # CMDs

    f, axs = plt.subplots(1,2,figsize=(13,6))

    axs = axs.flatten()

    for i,s in enumerate([sim,mod]) : 
        axs[i].plot(s['ubv_b']-s['ubv_v'],s['ubv_v'],',',label=labels[i])
        axs[i].set_ylim(10,-5)
        axs[i].set_xlim(-.5,2)
        axs[i].set_ylabel('$V$')
        axs[i].set_xlabel('$B - V$')
        axs[i].set_title(labels[i])
    
