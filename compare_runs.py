import pynbody
from pynbody.analysis.profile import Profile
import pylab as pl


if __name__ == '__main__' : 
    import getopt, sys, os

    colors = ['b','g','r','y','c']

    pl.figure()
    
    labels=["x0.5N","sdm","200pc soft","samesoft","fiducial"]
    
    for i in range(len(sys.argv)-1) :
        
        sim = pynbody.load(sys.argv[i+1])

        pynbody.analysis.halo.center(sim)
        pynbody.analysis.angmom.faceon(sim, cen_size=5, disk_size=5)
        pdm = Profile(sim.d,min=0.001,max=10,type='log',ndim=3)
        pg  = Profile(sim.g,min=0.001,max=10,type='log',ndim=3)
        ps  = Profile(sim.s,min=0.001,max=10,type='log',ndim=3)
        
        linecolor = "-%s"%colors[i]
        pl.plot(pdm['rbins'],pdm['density'],linecolor,label = labels[i])
        linecolor = "--%s"%colors[i]
        pl.plot(pg['rbins'],pg['density'],linecolor)
        linecolor = "-.{0}".format(colors[i])
        pl.plot(ps['rbins'],ps['density'],linecolor)

        pl.loglog()
        pl.xlim(1e-2,10)

  
    #pl.title(sys.argv[len(sys.argv)-1])
    pl.legend()

    pl.savefig('plots.pdf',format='pdf')
  
  
