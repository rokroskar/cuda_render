import pynbody,sys
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
import isolated as iso


def write_paramfile(stepnum):
    
    done = False
    info = open("output_%05d"%stepnum + "/info_%05d.txt"%stepnum)
    while(not done) : 
        line = info.readline().split()
    
        if len(line) > 0 : 
            if line[0] == 'aexp' : 
                aexp = float(line[2])
            elif line[0] == 'H0' : 
                H0 = float(line[2])
            elif line[0] == 'omega_l':
                omega_l = float(line[2])
            elif line[0] == 'omega_m' : 
                omega_m = float(line[2])
            elif line[0] == 'unit_l' : 
                unit_l = float(line[2])
            elif line[0] == 'unit_d' : 
                unit_d = float(line[2])
            elif line[0] == 'unit_t' : 
                unit_t = float(line[2])
                done = True

    pfile = open('run.param','w')
    g_to_msol = 5.0273993e-34
    cm_to_kpc = 3.2407793e-22
    cm_to_km = 1e-5
    s_to_gyr = 3.1688765e-17
    
    sysmass = unit_d*unit_l**3*g_to_msol # in Msol
    syslen = unit_l / aexp * cm_to_kpc # in kpc
    systime = np.sqrt(1.0/6.67e-8/(unit_d*aexp**3))*s_to_gyr # in Gyr
    sysvel = syslen/systime*0.97781311 # in km/s

    print "System mass = %e Msol"%sysmass
    print "System length = %e kpc"%syslen
    print "System time = %e"% systime

    pfile.write('dMsolUnit = %e\n'% sysmass)
    pfile.write('dKpcUnit = %e\n'% syslen)
    pfile.write('dHubble0 = %e\n'% (H0/100*syslen/sysvel/10.0))
    pfile.write('dLambda = %f\n'% omega_l)
    pfile.write('dOmega0 = %f\n'% omega_m)
    pfile.write('bComove = 1\n')
    pfile.close()

    

    

if __name__ == '__main__':
    make_gas_map(load_ramses(int(sys.argv[1])))


def hop_center(s):

    if s.filename[-1] == '/' : name = s.filename[-6:-1] 
    else: name = s.filename[-5:]

    filename = s.filename[:-12]+'hop/grp%s.pos'%name
    
    try : 
        data = np.genfromtxt(filename,unpack=True)
    except IOError : 
        import os
        os.system('cd %s;/home/itp/roskar/ramses/galaxy_formation/script_hop.sh %d;cd ..'%(s.filename[:-12],int(name)))
        data = np.genfromtxt(filename,unpack=True)

    cen = data.T[0][4:7]
    vcen = data.T[0][7:10]
    
    s['pos'] -= cen
    s['vel'] -= vcen
    


def make_pretty_picture(outputname, s = None):
    from cosmo_plots import make_multiple_snapshot_images
    if s is None: 
        s = load_center(outputname)
        
    st = s[pynbody.filt.Sphere('120 kpc')]
    make_multiple_snapshot_images([st],20)#,vgmin=21,vgmax=25.5,vsmin=-15,vsmax=-10.)
    
    return s, st


def compare_outputs(runlist,outnum) : 
    if isinstance(runlist[0], str) : 
        for i,run in enumerate(runlist) : 
            s = pynbody.load(run)
            hop_center(s)
            s.physical_units()
            pynbody.analysis.halo.sideon(s.g[pynbody.filt.Sphere(100)],mode='ssc')
            runlist[i] = s


def make_comparison_figure(dirlist,names):
    import matplotlib.image as mpimg
    f,axs = plt.subplots(1,3,figsize=(15,5))
    
    for i,run in enumerate(dirlist): 
        im = mpimg.imread(run+'/composites/composite13.76Gyr.png')
        axs[i].imshow(im)
        axs[i].annotate(names[i],(0.1,.87),xycoords='axes fraction', color = 'white')
        axs[i].yaxis.set_ticklabels("")
        axs[i].xaxis.set_ticklabels("")

    
def load_center(output):
    s = pynbody.load(output)
    hop_center(s)
#    s.physical_units()
    st = s[pynbody.filt.Sphere('100 kpc')]
    
    cen = pynbody.analysis.halo.center(st,retcen=True,mode='ssc')
    pynbody.analysis.angmom.faceon(st.g,disk_size='5 kpc',cen=cen,mode='ssc')
   # s.s['age'] = s._info['time'] - s.s['age']
   # s.s['age']*=s._info['unit_t']
   # s.s['age'].units = 's'
   # old = np.where(s.s['age'].in_units('Myr') > 10)[0]
   # s.s['oldmass'] = s.s['mass']
   # s.s['mass'][old] *= s.s['age'].in_units('Myr')[old]**(-.7)

    s['pos'].convert_units('kpc')
    s['vel'].convert_units('km s^-1')

    return s

@pynbody.ramses.RamsesSnap.derived_quantity
def tform(self) : 
    from scipy.io.numpyio import fread

    top = self
    while hasattr(top,'base') : top = self.base

    ncpu = top._info['ncpu']
    nstar = len(top.s)

    top.s['tform'] = -1.0
    done = 0
    for i in range(ncpu) : 
        f = open('%s/birth/birth_%s.out%05d'%(top.filename[:-12],top._timestep_id,i+1))
        n = fread(f,1,'i')
        n /= 8
        ages = fread(f,n,'d')
        new = np.where(ages > 0)[0]
        top.s['tform'][done:done+len(new)] = ages[new]
        done += len(new)
        f.close()
    top.s['tform'].units = 'Gyr'

    return self.s['tform']



    

    
    
