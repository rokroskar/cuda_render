import ebf
import pynbody
import os
import numpy as np
import matplotlib.pylab as plt

def make_galaxia_input(sim, run_enbid=False) :

    # load and align the snapshot
    
    s = pynbody.load(sim)
    pynbody.analysis.angmom.faceon(s)

    filename = s.filename

    # make the pos array

    pos = np.array([s.s['x'],s.s['y'],s.s['z'],s.s['vx'],s.s['vy'],s.s['vz']]).T
    
    # set low metalicities to some reasonable value

    s.s['mets'] = s.s['feh']
    s.s['alpha'] = 0.0
    bad = np.where(s.s['feh'] < -5)[0]
    s.s[bad]['mets'] = -5.0

    s.s['mets']+=.1
    # make the enbid file
    
    ebf.write(filename+'_galaxia.ebf', '/pos', pos, 'w')
    ebf.write(filename+'_galaxia.ebf', '/pos3', s.s['pos'], 'a')
    ebf.write(filename+'_galaxia.ebf', '/vel3', s.s['vel'], 'a')
    ebf.write(filename+'_galaxia.ebf', '/Mass', s.s['mass'].in_units('Msol'), 'a')
    ebf.write(filename+'_galaxia.ebf', '/feh', s.s['mets'], 'a')
    ebf.write(filename+'_galaxia.ebf', '/alpha', s.s['alpha'], 'a')
    ebf.write(filename+'_galaxia.ebf', '/age', s.s['age'], 'a')
    ebf.write(filename+'_galaxia.ebf', '/id', s.s['iord'], 'a')
    
    


    if run_enbid:

        # remove 's' to save space

        del(s)

        # run enbid

        os.system('~/bin/enbid -dmc --dim=3 --ngb=64 --dsuffix=_d3n64 %s_galaxia.ebf'%filename)
        os.system('~/bin/enbid -dmc --dim=6 --ngb=64 --dsuffix=_d6n64 %s_galaxia.ebf'%filename)




    

def compare_run_to_model(sim_gal, mod_gal) : 

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
            ax.hist(x[h],bins=100,normed=True,histtype='step',range=ranges[i],label=labels[j])
            ax.set_xlabel(r'%s'%h)
    

    axs[1].set_ylim(0,1.5)
    plt.subplots_adjust(hspace=.5)
    axs[0].legend(prop=dict(size=10))


    # CMDs

    f, axs = plt.subplots(1,2)

    axs = axs.flatten()

    for i,s in enumerate([sim,mod]) : 
        axs[i].plot(s['ubv_b']-s['ubv_v'],s['ubv_v'],',',label=labels[i])
        axs[i].set_ylim(10,-5)
        axs[i].set_xlim(-.5,2)
        axs[i].set_ylabel('$V$')
        axs[i].set_xlabel('$B - V$')
        
