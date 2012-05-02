import cosmo_plots as cp
import pynbody
import glob


list = glob.glob('00[1-9]??/*.0????')

for output in list : 
    s = pynbody.load(output)
    #cp.make_single_output_maps(s, s.halos())
    cp.make_single_output_profiles(s,s.halos())
