import pynbody, os, pickle, numpy as np, matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def get_rform_cosmo(s,nbins=250,plots=True):
    cenfilt = pynbody.filt.LowPass('r','5 kpc')
    if plots:
        pp = PdfPages('alignments.pdf')
        plt.ioff()
    bin_edges = np.linspace(np.min(s.s['tform'].in_units('Gyr')),
                            np.max(s.s['tform'].in_units('Gyr')),nbins+1)
    bins = .5*(bin_edges[1:]+bin_edges[:-1])
    cens  = np.zeros((nbins,3))*np.nan
    vcens = np.zeros((nbins,3))*np.nan
    js = np.zeros((nbins,3))*np.nan

    # make positions and velocities physical coordinates
    expfacfile = "expfacs.pickle"
    if os.path.exists(expfacfile):
        dat = pickle.load(open(expfacfile))
        expfac = dat
    else:
        c = pynbody.analysis.pkdgrav_cosmo.Cosmology(s)
        uniqtforms, tforminds = np.unique(s.s['tform'],return_index=True)
        print "starting expfac calculation"
        import time
        start = time.clock()
        uniqexpfac = [c.Time2Exp(tform) for tform in uniqtforms]
        print "done calculating expfac %5.3g s"%(time.clock()-start)
        expfac = np.zeros(len(s.s))
        print "assigning expacs"
        for itf, tfind in enumerate(tforminds):
            if itf==(len(tforminds)-1):
                expfac[tfind:]=uniqexpfac[itf]
            else:
                expfac[tfind:tforminds[itf+1]]=uniqexpfac[itf]
        print "done assigning expacs"
        pickle.dump(expfac,open(expfacfile,'w'))
    news = pynbody.new(stars=len(s.s))
    news.s['mass'] = s.s['massform']
    for i, arr in enumerate(['x','y','z']): 
        news.s[arr] = s.s['posform'][:,i]*expfac
        news.s['v'+arr] = s.s['velform'][:,i]*expfac
    news.s['tform']=s.s['tform']
    news.properties={'z':0,'a':1.0}

    # get center of mass for each bin using shrinking spheres
    print "Finding center positions"
    for ibe,below in enumerate(bin_edges[:-1]):
        tstars = news.s[pynbody.filt.BandPass('tform',str(below)+' Gyr',str(bin_edges[ibe+1])+' Gyr')]
        if len(tstars) > 0:
            cens[ibe]=pynbody.analysis.halo.center(tstars,mode='ssc',retcen=True)

    print "Setting positions according to new centers"
    for i, arr in enumerate(['x','y','z']) : 
        news.s[arr] -= np.interp(news.s['tform'].in_units('Gyr'),bins,cens[:,i])

    plt.plot(news.s['tform'].in_units('Gyr'),np.interp(news.s['tform'].in_units('Gyr'),bins,cens[:,0]),'.')
    plt.plot(bins,cens[:,0])

    #import pdb; pdb.set_trace()

    print "Finding centers of mass velocities"
    for ibe,below in enumerate(bin_edges[:-1]):
        tstars = news.s[pynbody.filt.BandPass('tform',str(below)+' Gyr',str(bin_edges[ibe+1])+' Gyr')]
        if len(tstars[cenfilt]) > 5:
            vcens[ibe]=pynbody.analysis.halo.vel_center(tstars[cenfilt],retcen=True,cen_size='5 kpc')
        #else:
            #import pdb; pdb.set_trace()

    goodvcen = np.isfinite(vcens).all(axis=1)
    print "Setting velocities"
    for i, arr in enumerate(['x','y','z']) : 
        news.s['v'+arr] -= np.interp(news.s['tform'],bins[goodvcen],vcens[goodvcen,i])

    r_array = news.s['r']
    print "Rotating angular momenta"
    for ibe,below in enumerate(bin_edges[:-1]):
        tstars = news.s[pynbody.filt.BandPass('tform',str(below)+' Gyr',str(bin_edges[ibe+1])+' Gyr')]
        if ((len(tstars[cenfilt]) > 5) & np.isfinite(vcens[ibe]).all()):
            js[ibe] = pynbody.analysis.angmom.ang_mom_vec(tstars[cenfilt])
            trans = pynbody.analysis.angmom.calc_faceon_matrix(js[ibe])

            jstars = news.s[pynbody.filt.BandPass('tform',str(below)+' Gyr',str(bin_edges[ibe+1])+' Gyr')]
            jstars.transform(trans)
            if plots:
                f,ax = plt.subplots(1,3)
                ax[0].plot(jstars['x'].in_units('kpc'),jstars['y'].in_units('kpc'),'.')
                ax[1].plot(jstars['x'].in_units('kpc'),jstars['z'].in_units('kpc'),'.')
                ax[2].plot(jstars['y'].in_units('kpc'),jstars['z'].in_units('kpc'),'.')
                for a in ax:
                    a.set_aspect('equal')
                    a.set_xlim(-25,25)
                    a.set_ylim(-25,25)
                pp.savefig()
                plt.clf()
                plt.close()

    s.s['rxyform'] = news.s['rxy']
    s.s['rxyform'].write()
    
    if plots:
        pp.close()    
    news.write(filename='new.starlog.tipsy',fmt=pynbody.tipsy.TipsySnap)
    pickle.dump({'cen':cens,'j':js},open('jcen','w'))
