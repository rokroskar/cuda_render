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
        dir = s.filename[:-12] if len(s.filename[:-12]) else './'
        
        os.system('cd %s;/home/itp/roskar/ramses/galaxy_formation/script_hop.sh %d;cd ..'%(dir,int(name)))
        data = np.genfromtxt(filename,unpack=True)

    return data

def hop_center(s,halo=0):
    data = load_hop(s)

    cen = data.T[halo][4:7]
    vcen = data.T[halo][7:10]
    
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

    
def load_center(output, align=True, halo=0):
    s = pynbody.load(output)
    hop_center(s,halo)
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


def prepare_for_amiga(outname, write = False, run_pkdgrav = False, run_amiga=False, zbox=False) :
    import os 
    import isolated as iso
    from pynbody.units import Unit

    s = pynbody.load(outname)
    
    #massunit = (1.0/pynbody.units.G*
    #            pynbody.units.Unit('%f cm'%s._info['unit_l'])**3/
    #            pynbody.units.Unit('%f s'%s._info['unit_t'])**2).in_units('Msol')
    #print massunit
    
    # figure out the units starting with mass

    cmtokpc = 3.2407793e-22
    lenunit  = s._info['unit_l']/s.properties['a']*cmtokpc
    massunit = pynbody.analysis.cosmology.rho_crit(s,z=0,unit='Msol kpc^-3')*lenunit**3
    G_u = 4.4998712e-6 # G in kpc^3 / Msol / Gyr^2
    timeunit = np.sqrt(1/G_u * lenunit**3/massunit)
    
    l_unit = Unit('%f kpc'%lenunit)
    t_unit = Unit('%f Gyr'%timeunit)
    v_unit = l_unit/t_unit
   
    print massunit, timeunit

    newfile = "%s_tipsy/%s_fullbox.tipsy"%(s.filename,outname)

    if write:
        s['mass'].convert_units('%f Msol'%massunit)
        s.g['temp']
        print s['mass']
        s.s['tform'].convert_units(t_unit)
        s.g['metals'] = s.g['metal']
        s['pos'].convert_units(l_unit)
        s['vel'].convert_units(v_unit)
        s['eps'] = s.g['smooth'].min()
        s['eps'].units = s['pos'].units
        del(s.g['metal'])
        del(s['smooth'])
        print s['vel']
        print s['pos']
        
        s.write(filename='%s'%newfile, fmt=pynbody.tipsy.TipsySnap, binary_aux_arrays = True)

    if run_pkdgrav: spawn_pkdgrav(s,newfile,lenunit,massunit,timeunit,zbox)
    if run_amiga : spawn_amiga(s,newfile,lenunit, massunit, timeunit, zbox)

def spawn_pkdgrav(s, newfile, lenunit, massunit, timeunit, zbox = False) : 
    from pynbody.units import Unit, G
    import os

    l_unit = Unit('%f kpc'%lenunit)
    t_unit = Unit('%f Gyr'%timeunit)
    v_unit = l_unit/t_unit
    
    f = open('%s.param'%newfile,'w')
        # determine units
    f.write('dKpcUnit = %f\n'%lenunit)
    f.write('dMsolUnit = %e\n'%massunit)
    f.write('dOmega0 = %f\n'%s.properties['omegaM0'])
    f.write('dLambda = %f\n'%s.properties['omegaL0'])
    h = Unit('%f km s^-1 Mpc^-1'%(s.properties['h']*100))
    f.write('dHubble0 = %f\n'%h.in_units(v_unit/l_unit))
    f.write('bComove = 1\n')
    f.close()

    
        

    if zbox : 
        f = open('%s.pkdgrav.zbox.sh'%newfile,'w')
        f.write('#!/bin/sh\n')
        f.write('#SBATCH -J zerosteps\n')
        f.write('#SBATCH --ntasks=32 \n')
        f.write('export PATH=/opt/mpi/mvapich2/1.9b/gcc/4.7.2/bin:/opt/gcc/4.7.2/bin:$PATH\n')
        f.write('export LD_LIBRARY_PATH=/opt/mpi/mvapich2/1.9b/gcc/4.7.2/lib:/opt/gcc/4.7.2/lib64:/opt/gcc/4.7.2/lib\n')
        f.write('srun /home/itp/roskar/bin/pkdgrav2_mpi +potout +accout +vstart +std +vdetails -n 0 -o %s -gas +overwrite -I %s %s.param\n'%(newfile,newfile,newfile))
        f.write('python finish.py\n')
        f.close()
        f = open('finish.py','w')
        f.write('#!/usr/bin/python\n')
        f.write('import ramses_pynbody as ram\n')
        f.write('ram.organize("%s")\n'%newfile)
        os.system('sbatch %s.pkdgrav.zbox.sh'%newfile)

    else : 
        command = "~/bin/pkdgrav2_pthread -sz 16 +overwrite +potout +accout +vstart +std +vdetails -n 0 -o %s -I %s %s.param"%(newfile,newfile,newfile)
        print command
        os.system('rm .lockfile')
        os.system(command)
        organize(newfile)

def organize(filename) : 
    import os

    os.system('mv %s.00000.pot %s.pot'%(filename,filename))
    os.system('mv %s.00000.acc %s.acc'%(filename,filename))
    os.system('rm %s.00000.*'%filename)
    st = pynbody.load(filename)
    st['phi'] = st['pot']
    st['phi'].write(overwrite=True)

def spawn_amiga(s, newfile, lenunit, massunit, timeunit, zbox = False) :
    from pynbody.units import Unit, G
    import os
    
    l_unit = Unit('%f kpc'%lenunit)
    t_unit = Unit('%f Gyr'%timeunit)
    v_unit = l_unit/t_unit
   
    f = open('%s.AHF.input'%newfile,'w')
    f.write('[AHF]\n')
    f.write('ic_filename = %s\n'%newfile)
    f.write('ic_filetype = 90\n')
    f.write('outfile_prefix = %s\n'%newfile)
    f.write('LgridDomain = 256\n')
    f.write('LgridMax = 2097152\n')
    f.write('NperDomCell = 5\n')
    f.write('NperRefCell = 5\n')
    f.write('VescTune = 1.0\n')
    f.write('NminPerHalo = 50\n')
    f.write('RhoVir = 0\n')
    f.write('Dvir = 200\n')
    f.write('MaxGatherRad = 1.0\n')
    f.write('[TIPSY]\n')
    f.write('TIPSY_BOXSIZE = %e\n'%(s.properties['boxsize'].in_units('Mpc')*s.properties['h']/s.properties['a']))
    f.write('TIPSY_MUNIT   = %e\n'%(massunit*s.properties['h']))
    f.write('TIPSY_OMEGA0  = %f\n'%s.properties['omegaM0'])
    f.write('TIPSY_LAMBDA0 = %f\n'%s.properties['omegaL0'])
    
 #   velunit = Unit('%f cm'%s._info['unit_l'])/Unit('%f s'%s._info['unit_t'])
    
    f.write('TIPSY_VUNIT   = %e\n'%v_unit.ratio('km s^-1 a', **s.conversion_context()))
    

    # the thermal energy in K -> km^2/s^2

    f.write('TIPSY_EUNIT   = %e\n'%((pynbody.units.k/pynbody.units.m_p).in_units('km^2 s^-2 K^-1')*5./3.))
    f.close()
    if zbox : 
        f = open('%s.zbox.sh'%newfile,'w')
        f.write('#!/bin/sh\n')
        f.write('#SBATCH -J amiga\n')
        f.write('#SBATCH -N 1 -n 1 --cpus-per-task=16\n')
        f.write('export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n')
        f.write('srun /home/itp/roskar/bin/amiga_pthread_for_tipsyramses %s.AHF.input\n'%newfile)
        f.close()
        os.system('sbatch %s.zbox.sh'%newfile)
    else : 
        os.environ['OMP_NUM_THREADS'] = '16'
        os.system("~/bin/amiga_pthread_for_tipsyramses %s.AHF.input"%newfile)

def make_rgb_image(s,width,xsize=500,ysize=500,filename='test.png') : 
    from PIL import Image
    from matplotlib.colors import Normalize

    rgbArray = np.zeros((xsize,ysize,3),'uint8')

    tem = pynbody.plot.image(s,qty='temp',width=width,resolution=xsize,noplot=True,threaded=10,approximate_fast=False, denoise=True)
    rho = pynbody.plot.image(s,qty='rho',width=width,resolution=xsize,noplot=True,threaded=10,approximate_fast=False, denoise=True)
    met = pynbody.plot.image(s,qty='metal',width=width,resolution=xsize,noplot=True,threaded=10,log=False,approximate_fast=False, denoise=True)
    
    rgbArray[...,0] = Normalize()(tem)*256#Normalize(vmin=3.5,vmax=6.5,clip=True)(tem)*256
    rgbArray[...,1] = Normalize()(rho)*256
    rgbArray[...,2] = Normalize()(np.log10(met/0.02))*256#Normalize(vmin=-3,vmax=0,clip=True)(np.log10(met/0.02))*256

    img = Image.fromarray(rgbArray)

    img.save(filename)
    
    return tem,rho,met


def make_rgb_stellar_image(s,width,xsize=500,ysize=500,filename='test.png') : 
    from PIL import Image
    from matplotlib.colors import Normalize

    rgbArray = np.zeros((xsize,ysize,3),'uint8')

    R = pynbody.plot.image(s.s,qty='k_lum_den',width=width,resolution=xsize,noplot=True,threaded=10,approximate_fast=False, av_z=True)
    G = pynbody.plot.image(s.s,qty='b_lum_den',width=width,resolution=xsize,noplot=True,threaded=10,approximate_fast=False, av_z=True)
    B = pynbody.plot.image(s.s,qty='u_lum_den',width=width,resolution=xsize,noplot=True,threaded=10,approximate_fast=False,av_z=True)
    
    rgbArray[...,0] = Normalize(vmin=2.,vmax=7.5,clip=True)(R)*256#Normalize(vmin=3.5,vmax=6.5,clip=True)(tem)*256
    rgbArray[...,1] = Normalize(vmin=2.,vmax=7,clip=True)(G)*256
    rgbArray[...,2] = Normalize(vmin=2.,vmax=6,clip=True)(B)*256#Normalize(vmin=-3,vmax=0,clip=True)(np.log10(met/0.02))*256

    img = Image.fromarray(rgbArray)

    img.save(filename)
    
    return rgbArray, R, G, B


@pynbody.ramses.RamsesSnap.derived_quantity
def temp(self) : 
    return (self['p']*pynbody.units.m_p/pynbody.units.k/self['rho']).in_units('K')

@pynbody.ramses.RamsesSnap.derived_quantity
def rhoz(self):
    res = self['rho']*self['metal']
    res.units = self['rho'].units
    return res

@pynbody.ramses.RamsesSnap.derived_quantity
def rho_ovi(self) : 
    return self['rhoz']*0.2*0.3/12.

@pynbody.ramses.RamsesSnap.derived_quantity
def tform(self) : 
    from numpy import fromfile
    
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

        n = fromfile(f,'i',1)
        if n > 0: 
            n /= 8
            ages = fromfile(f,'d',n)
            new = np.where(ages > 0)[0]
            top.s['tform'][done:done+len(new)] = ages[new]
            done += len(new)

        f.close()
    top.s['tform'].units = 'Gyr'

    return self.s['tform']


