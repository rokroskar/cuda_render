import ebf
import pynbody
import os
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import gaussian_kde as kde
import utils 

home = os.environ['HOME']
galdir = '%s/GalaxiaData'%home
galbin = '%s/bin/galaxia'%home
galinput = '{0}/nbody1/mw/mw1.ebf'.format(galdir)
dims = ['d3','d6','d8']
names = ['3D','6D','8D']

def make_galaxia_input(sim, run_enbid=False) :

    # load and align the snapshot
    
    s = pynbody.load(sim)
    pynbody.analysis.angmom.faceon(s)
    
    filt = pynbody.filt.Disc(15,2.5)
#    s=s[filt]
    filename = s.filename

      
    # set low metalicities to some reasonable value

    s.s['mets'] = s.s['feh']
    s.s['alpha'] = s.s['ofe']
    bad = np.where(s.s['feh'] < -5)[0]
    s.s[bad]['mets'] = -5.0

    s.s['mets']+=.1
    
    # shift metallicities and ages around in a random way to avoid identical values
    s.s['mets'] += np.random.normal(0,.001,len(s.s))
    s.s['ages'] = s.s['age'] + np.random.normal(0,1e-5,len(s.s))
    s.s['alpha'] += np.random.normal(0,.001,len(s.s))
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

        os.system('~/bin/enbid -dmcl --dim=3 --ngb=64 --dsuffix=_d3n64_c %s_galaxia.ebf'%filename)
        os.system('~/bin/enbid -dmsl --dim=3 --ngb=64 --dsuffix=_d3n64_s %s_galaxia.ebf'%filename)
        os.system('~/bin/enbid -dmcl --dim=6 --ngb=64 --dsuffix=_d6n64_c %s_galaxia.ebf'%filename)
        os.system('~/bin/enbid -dmsl --dim=6 --ngb=64 --dsuffix=_d6n64_s %s_galaxia.ebf'%filename)
        os.system('~/bin/enbid -dmsl --dim=8 --ngb=64 --dsuffix=_d8n64_s %s_galaxia.ebf'%filename)


def run_galaxia(ndim=3,l=0,b=90,mmin=0,mmax=10,geom=1,area=1000,frac=1.0,rmax=1000,photoerror=0) :
    
    
    # make the parameter file
    
    f = open('{0}/paramfiles/galrunparam'.format(galdir),'w')
    
    f.write('outputFile           gal_nbody_l{0}_b{1}_{2}_{3}\n'.format(l,b,mmin,mmax))
    f.write('outputDir            {0}/d{1}\n'.format(galdir,ndim))
    f.write('photoSys             UBV\n')
    f.write('magcolorNames        V,B-V\n')
    f.write('appMagLimits[0]      {0}\n'.format(mmin))
    f.write('appMagLimits[1]      {0}\n'.format(mmax))
    f.write('absMagLimits[0]      -1000\n')
    f.write('absMagLimits[1]      1000\n')
    f.write('colorLimits[0]       -1000\n')
    f.write('colorLimits[1]       1000\n')
    f.write('geometryOption       {0}\n'.format(geom))
    f.write('longitude            {0}\n'.format(l))
    f.write('latitude             {0}\n'.format(b))
    f.write('surveyArea           {0}\n'.format(area))
    f.write('fSample              {0}\n'.format(frac))
    f.write('popID                -1\n')
    f.write('warpFlareOn           1\n')
    f.write('seed                  12\n')
    f.write('r_max                 {0}\n'.format(rmax))
    f.write('starType              0\n')
    f.write('photoError            {0}\n'.format(photoerror))

    f.close()
    
    # run the code
    import os

    os.system('{0} -r --nfile={1} --hdim={2} {3}/paramfiles/galrunparam'.format(galbin,galinput,ndim,galdir))


def compare_run_to_model(sim_gal, mod_gal) : 

    sim = ebf.read(sim_gal)
    mod = ebf.read(mod_gal)

    # two columns -- simulation on left, galaxia model on right

    hists = ['ubv_v','age','feh','rad']
    ranges = [[-2,15],[0,10],[-2,.5],[0,2]]
    labels = ['sim','model']

    sim['age'] = 10**sim['age']/1e9
    mod['age'] = 10**mod['age']/1e9
    for band in ['u','b','v','k','j'] :
        sim['app_%s'%band] = sim['ubv_%s'%band]+5*np.log10(100*sim['rad']) #+ sim['exbv_schlegel']
        mod['app_%s'%band] = mod['ubv_%s'%band]+5*np.log10(100*mod['rad']) #+ mod['exbv_schlegel']
    f, axs = plt.subplots(len(hists),1,figsize=(7,len(hists)*2.5))
        
    # histograms

    for i,h in enumerate(hists) : 
        for j,x in enumerate([sim,mod]) :
            ax = axs[i]
            #if i == 1 : 
            #    ax.hist(x['ubv_b']-x['ubv_v'],histtype='step',normed=True,range=[-.5,2],bins=50)
            #else : 
            ax.hist(x[h],bins=50,normed=True,histtype='step',range=ranges[i],label=labels[j],weights=x['smass'])
#            k = kde(x[h][np.where((x[h]>ranges[i][0]) & (x[h]<ranges[i][1]))[0]])
#            xs = np.linspace(ranges[i][0],ranges[i][1],200)
#            ax.plot(xs,k(xs),label=labels[j])
            ax.set_xlabel(r'%s'%h.replace('_','-'))
    

#    axs[1].set_ylim(0,)
    plt.subplots_adjust(hspace=.5)
    axs[0].legend(prop=dict(size=10))
    axs[0].set_xlabel('M$_V$')
    axs[1].set_xlabel('Age [Gyr]')
    axs[2].set_xlabel('[Fe/H]')
    axs[3].set_xlabel('$d$ [kpc]')
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
    


def load_smoothing_data(flist=None):
    if flist is None: 
        flist = ['nbody1/mw/mw1_d3n64_s_den.ebf',
                 'nbody1/mw/mw1_d6n64_s_den.ebf',
                 'nbody1/mw/mw1_d8n64_s_den.ebf']
    return map(ebf.read,flist)

def load_stellar_data(l='0',b='90',mmin='0',mmax='10'):
    flist = ['d3/galaxy1_nbody_l%s_b%s_%s_%s.ebf'%(l,b,mmin,mmax),
             'd6/galaxy1_nbody_l%s_b%s_%s_%s.ebf'%(l,b,mmin,mmax),
             'd8/galaxy1_nbody_l%s_b%s_%s_%s.ebf'%(l,b,mmin,mmax)]
    
    return map(ebf.read,flist)

def compare_smoothing_volumes(sms,s) : 
    names = ['3D','6D','8D']
    colors = ['blue','green','red']

    f,ax = plt.subplots()
    snfilt = pynbody.filt.SolarNeighborhood(7,9,1.0)
    
    ind = snfilt.where(s.s)[0]

    for sm, name,color in zip(sms,names,colors) : 
        vol = (sm['h_smooth'][:,0:3]).prod(axis=1)
        volsn = (sm['h_smooth'][ind,0:3]).prod(axis=1)
        #kd_full = kde(vol)
        #kd_sn = kde(volsn)
        #xs = np.logspace(-8,2,1000)
        htot,binstot = np.histogram(np.log10(vol),100,normed=True)
        hsn,binssn = np.histogram(np.log10(volsn),100,normed=True)

        binstot = .5*(binstot[:-1]+binstot[1:])
        binssn = .5*(binssn[:-1]+binssn[1:])
                
        ax.plot(binstot,htot,color=color,label=name)
        ax.plot(binssn,hsn,color=color,linestyle='--')
        

    plt.legend(prop=dict(size=12))
    plt.xlabel('log(smoothing volume [kpc$^{-3}$])')
    
        
def compare_plists(sms, s, l=0,b=90,mmin=0,mmax=10) : 
    plist_name = 'gal_nbody_l{0}_b{1}_{2}_{3}_plist'.format(l,b,mmin,mmax)
    f, axs = plt.subplots(2,3,figsize=(17.5,7.2))

    for i,d in enumerate(dims):
        ps = '{0}/{1}'.format(d,plist_name)
        inds = np.array(np.genfromtxt(ps),dtype='int')
        axs[0,i].plot(s.s[inds]['y'][::2],s.s[inds]['x'][::2], 'k,')#, alpha=.5)
        axs[0,i].plot(0,-8,'yo',markersize=10)
        axs[1,i].plot(s.s[inds]['y'][::2],s.s[inds]['z'][::2], 'k,')#, alpha=.5)
        axs[0,i].set_title(names[i])
        
    for ax in axs[0,:]:
        ax.set_ylim(-10,-4)
        ax.set_xlim(-5,5)
        #ax.set_aspect(1)
        ax.set_xlabel('$x$ [kpc]',fontsize=14)
        ax.set_ylabel('$y$ [kpc]',fontsize=14)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)
        for r in [2,4,6,8,10] : 
            circ = plt.Circle((0,0),radius=r,edgecolor='red',facecolor='none',linestyle='dashed',linewidth=2.0)
            ax.add_patch(circ)

    for ax in axs[1,:]:
        ax.set_ylim(-2,.5)
        ax.set_xlim(-2,2)
        ax.set_aspect(1)
        ax.set_xlabel('$x$ [kpc]',fontsize=14)
        ax.set_ylabel('$z$ [kpc]',fontsize=14)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)
        ax.plot((-10,10),(0,0),'r--',linewidth=2)
        ax.plot(0,.015,'yo',markersize=10)

def compare_stars(l=0,b=90,mmin=0,mmax=10) :
    plist_name = 'gal_nbody_l{0}_b{1}_{2}_{3}.ebf'.format(l,b,mmin,mmax)
    f, axs = plt.subplots(2,3,figsize=(17.5,7.2))
#    f, axs = plt.subplots()

    for i,d in enumerate(dims):
        gal = ebf.read('{0}/{1}'.format(d,plist_name))
        if i == 0:
            fact = 10
        else : 
            fact = 1
        axs[0,i].plot(gal['py'][::10*fact],gal['px'][::10*fact], 'k,',alpha=.2)
        axs[0,i].plot(0,0,'yo',markersize=10)
        axs[1,i].plot(gal['py'][::10*fact],gal['pz'][::10*fact], 'k,',alpha=.2)
        axs[1,i].plot(0,0,'yo',markersize=10)
        axs[0,i].set_title(names[i])
        print 'total mass = %e'%gal['smass'].sum()


    for ax in axs[0,:]:
        ax.set_ylim(-.6,.6)
        ax.set_xlim(-.6,.6)
        #ax.set_aspect(1)
        ax.set_xlabel('$x$ [kpc]',fontsize=14)
        ax.set_ylabel('$y$ [kpc]',fontsize=14)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)
        for r in [2,4,6,8,10] : 
            circ = plt.Circle((0,0),radius=r,edgecolor='red',facecolor='none',linestyle='dashed')
            ax.add_patch(circ)

    for ax in axs[1,:]:
        ax.set_ylim(-2.,.1)
        ax.set_xlim(-.6,.6)
        
        ax.set_xlabel('$x$ [kpc]',fontsize=14)
        ax.set_ylabel('$z$ [kpc]',fontsize=14)
        plt.setp(ax.get_xticklabels(), fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14)
#        ax.plot((-10,10),(0,0),'r--')


def compare_stellar_distributions(l=0,b=90,mmin=0,mmax=10) :
    plist_name = 'gal_nbody_l{0}_b{1}_{2}_{3}.ebf'.format(l,b,mmin,mmax)
    f, axs = plt.subplots(1,3,figsize=(17.5,4))

    for i,d in enumerate(dims):
        gal = ebf.read('{0}/{1}'.format(d,plist_name))
        gal['linage'] = 10**gal['age']/1e9
        for ax, prop, range in zip(axs.flatten(),
                                   ['feh','linage','rad'], 
                                   ([-1,.5],[1,10],[0,2.0])):
            ax.hist(gal[prop],weights=gal['smass'],histtype='step',label=names[i], bins = 20, range=range,  normed=True)
            
            #kd = kde(gal[prop])
            #x = np.linspace(range[0],range[1],100)
            #ax.plot(x,kd(x),label=names[i])
        print 'total mass = %e'%gal['smass'].sum()
        
    axs[0].legend()
    axs[0].set_xlabel('[Fe/H]')
    axs[1].set_ylim(0,.2)
    axs[1].set_xlabel('log(Age [Gyr])')
    axs[2].set_xlabel('$r$ [kpc]')
    axs[0].annotate('$l={0}$, $b={1}$'.format(l,b),(.05,.9),textcoords='axes fraction')


def make_df_example(): 
    x=np.random.multivariate_normal([0,0],[[4,0],[0,1]],50)
    xx,yy = np.meshgrid(np.arange(-6,6,.1),np.arange(-6,6,.1))
    zz = np.exp(-xx**2/4-yy**2)
    f,ax=plt.subplots()
    ax.imshow(np.log10(zz),extent=(-6,6,-6,6),cmap=plt.cm.gist_heat,aspect=0.7538461538461538)
    ax.set_frame_on(False)
    utils.clear_labels(ax,True)    
    plt.savefig('smooth_df.pdf',format='pdf',bbox_inches='tight',transparent=True)
    
    ax.plot(x[:,0],x[:,1],'wo')
    ax.set_xlim(-6,6)
    ax.set_ylim(-6,6)
    ax.set_frame_on(False)
    utils.clear_labels(ax,True)
    plt.savefig('sampled_df.pdf',format='pdf',bbox_inches='tight',transparent=True)
    
    f,ax=plt.subplots()
    ax.plot(x[:,0],x[:,1],'wo')
    ax.set_xlim(-6,6)
    ax.set_ylim(-6,6)
    ax.set_frame_on(False)
    utils.clear_labels(ax,True)
    plt.savefig('sample_points.pdf',format='pdf',bbox_inches='tight',transparent=True)
    
    xs = np.arange(-1,1,.01)
    ys = np.arange(-1,1,.01)
    xx,yy=np.meshgrid(xs,ys)
    zz = np.exp(-xx**2/.16-yy**2/.04)
    for point in x : 
        plt.contour(xs+point[0],ys+point[1],zz,3,colors='k')
    ax.set_frame_on(False)
    utils.clear_labels(ax,True)
    plt.savefig('kde_points.pdf',format='pdf',bbox_inches='tight',transparent=True)

    f,ax=plt.subplots()
    xs = np.arange(-6,6,.1)
    ys = np.arange(-6,6,.1)
    kd = pynbody.plot.util.fast_kde(x[:,0],x[:,1],gridsize=(len(xs),len(ys)),extents=[-6,6,-6,6])
    plt.contourf(xs,ys,kd,np.linspace(1e-3,kd.max(),5),cmap=plt.cm.Greys)
    ax.set_frame_on(False)
    utils.clear_labels(ax,True)
    plt.savefig('kde.pdf',format='pdf',bbox_inches='tight',transparent=True)
    

    #x=np.random.multivariate_normal([0,0],[[4,0],[0,1]],5000)
    x = kde(x.T).resample(1000).T
    f,ax=plt.subplots()
    ax.plot(x[:,0],x[:,1],'ko', alpha=.3)
    ax.set_xlim(-6,6)
    ax.set_ylim(-6,6)
    ax.set_frame_on(False)
    utils.clear_labels(ax,True)
    plt.savefig('resample_points.pdf',format='pdf',bbox_inches='tight',alpha=.2,transparent=True)

    


def savefig(name):
    plt.savefig('plots/%s.pdf'%name,format='pdf',bbox_inches='tight')
    plt.savefig('plots/%s.eps'%name,format='eps',bbox_inches='tight')
    
