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


def load_hop(s): 
    
    if s.filename[-1] == '/' : 
        name = s.filename[-6:-1] 
        filename = s.filename[:-13]+'hop/grp%s.pos'%name
    else: 
        name = s.filename[-5:]
        filename = s.filename[:-12]+'hop/grp%s.pos'%name
    
    try : 
        data = np.genfromtxt(filename,unpack=True)
    except IOError : 
        import os
        os.system('cd %s;/home/itp/roskar/ramses/galaxy_formation/script_hop.sh %d;cd ..'%(s.filename[:-12],int(name)))
        data = np.genfromtxt(filename,unpack=True)

    return data

def hop_center(s):
    data = load_hop(s)

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

    
def load_center(output, align=True):
    s = pynbody.load(output)
    hop_center(s)
#    s.physical_units()
    st = s[pynbody.filt.Sphere('100 kpc')]
    
    cen = pynbody.analysis.halo.center(st,retcen=True,mode='ssc',verbose=True)
    
    if align: 
        pynbody.analysis.angmom.faceon(st.s,disk_size='10 kpc',cen=cen,mode='ssc')
    else :
        s['pos'] -= cen

    s['pos'].convert_units('kpc')
    s['vel'].convert_units('km s^-1')

    return s

def luminosity_weighted_mass(s):
    del(s.s['age'])
    s.s['age'] = s.properties['time'].in_units('Gyr') - s.s['tform']
    s.s['age'].convert_units('Gyr')
    old = np.where(s.s['age'].in_units('Myr') > 10)[0]
    s.s['oldmass'] = s.s['mass']
    s.s['mass'][old] *= s.s['age'].in_units('Myr')[old]**(-.7)




def convert_to_tipsy(output) : 
    s = load_center(output)

    for key in ['pos','vel','mass','iord','metal'] : 
        try: 
            s[key]
        except:
            pass

    s['eps'] = s.g['smooth'].min()

    for key in ['rho','temp','p']:
        s.g[key]

    print s.g['temp']

    s.s['tform']
    
    massunit = 2.222286e5  # in Msol
    dunit = 1.0 # in kpc
    denunit = massunit/dunit**3
    velunit = 8.0285 * np.sqrt(6.67384e-8*denunit) * dunit
    timeunit = dunit / velunit * 0.97781311

    s['pos'].convert_units('kpc')
    s['vel'].convert_units('%e km s^-1'%velunit)
    s['mass'].convert_units('%e Msol'%massunit)
    s['eps'].convert_units('kpc')
    s.g['rho'].convert_units('%e Msol kpc^-3'%denunit)
    
    s.s['tform'].convert_units('Gyr')    
    del(s.g['smooth'])
    s.s['metals'] = s.s['metal']
    s.g['metals'] = s.g['metal']
    del(s['metal'])
    s.g['temp']
    s.properties['a'] = pynbody.analysis.cosmology.age(s)
    s[pynbody.filt.Sphere('200 kpc')].write(pynbody.tipsy.TipsySnap,'%s.tipsy'%output[-12:])


def make_rgb_image(s,width,xsize=500,ysize=500,filename='test.png') : 
    from PIL import Image
    from matplotlib.colors import Normalize

    rgbArray = np.zeros((xsize,ysize,3),'uint8')

    tem = pynbody.plot.image(s,qty='temp',av_z='rho',width=width,resolution=xsize,noplot=True,threaded=10,approximate_fast=False)
    rho = pynbody.plot.image(s,qty='rho',av_z='rho',width=width,resolution=xsize,noplot=True,threaded=10,approximate_fast=False)
    met = pynbody.plot.image(s,qty='metal',av_z='rho',width=width,resolution=xsize,noplot=True,threaded=10,log=False,approximate_fast=False)
    
    rgbArray[...,0] = Normalize(vmin=3.5,vmax=6.5,clip=True)(tem)*256
    rgbArray[...,1] = Normalize()(rho)*256
    rgbArray[...,2] = Normalize(vmin=-3,vmax=0,clip=True)(np.log10(met/0.02))*256

    img = Image.fromarray(rgbArray)

    img.save(filename)
    
    return tem,rho,met

def make_composite_image(s,xsize=500,width='100 kpc',vmin=-1,vmax=6.5) :
    from PIL import Image
    from matplotlib.colors import Normalize

    rgbArray = np.zeros((xsize,xsize,3),'uint8')

    bband = pynbody.plot.image(s.s,qty='b_lum_den',av_z='rho',width=width,resolution=xsize,noplot=True,threaded=10,approximate_fast=False)
    rband = pynbody.plot.image(s.s,qty='r_lum_den',av_z='rho',width=width,resolution=xsize,noplot=True,threaded=10,approximate_fast=False)
    kband = pynbody.plot.image(s.s,qty='k_lum_den',av_z='rho',width=width,resolution=xsize,noplot=True,threaded=10,approximate_fast=False)
    
    norm = Normalize(vmin=vmin,vmax=vmax,clip=True)
    rgbArray[...,0] = norm(kband)*256
    rgbArray[...,1] = norm(rband)*256
    rgbArray[...,2] = norm(bband)*256

    img = Image.fromarray(rgbArray)

    img.show()
    
    return bband,rband,kband

@pynbody.ramses.RamsesSnap.derived_quantity
def temp(self) : 
    return (self['p']*pynbody.units.m_p/pynbody.units.k/self['rho']).in_units('K')


@pynbody.ramses.RamsesSnap.derived_quantity
def tform(self) : 
    from scipy.io.numpyio import fread

    top = self
    while hasattr(top,'base') : top = self.base

    ncpu = top._info['ncpu']
    nstar = len(top.s)

    top.s['tform'] = -1.0
    done = 0

    if len(top.filename.split('/')) > 1 : 
        parent_dir = top.filename[:-12]
    else : 
        parent_dir = './'

    for i in range(ncpu) : 
        try : 
            f = open('%s/birth/birth_%s.out%05d'%(parent_dir,top._timestep_id,i+1))
        except IOError : 
            import os
            
            os.system("cd %s; mkdir birth; /home/itp/roskar/ramses/galaxy_formation/part2birth -inp output_%s; cd .."%(parent_dir,top._timestep_id))
            f = open('%s/birth/birth_%s.out%05d'%(parent_dir,top._timestep_id,i+1))

        n = fread(f,1,'i')
        n /= 8
        ages = fread(f,n,'d')
        new = np.where(ages > 0)[0]
        top.s['tform'][done:done+len(new)] = ages[new]
        done += len(new)
        f.close()
    top.s['tform'].units = 'Gyr'

    return self.s['tform']
