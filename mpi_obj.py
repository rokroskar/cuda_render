import numpy as np
from mpi4py import MPI
from pynbody.snapshot import SimSnap
from pynbody.tipsy import TipsySnap
import pynbody.family as family
from pynbody.array import SimArray
from pynbody import config

MPI_ROOT = 0

class ParallelTipsySnap(TipsySnap) :

    def __init__(self, filename) :
        import pynbody.family as family

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        self.mpi_comm = comm
        self.mpi_size = size
        self.mpi_rank = rank

        if rank == 0: 
            TipsySnap.__init__(self, filename)
            # save the original data arrays
            self._orig_arrays = self._arrays
            self._orig_family_arrays = self._family_arrays
            self._orig_family_slice = self._family_slice
            
            self._family_arrays = {}
            self._arrays = {}

        else : 
            TipsySnap.__init__(self, filename, only_header=True)
            del(self['pos'])
            del(self['vel'])

        totg = len(self.g)
        totd = len(self.d)
        tots = len(self.s)
        
        self._global_totg = totg
        self._global_totd = totd
        self._global_tots = tots
        self._global_tot = totg+totd+tots

        extra_gas = totg%size
        extra_dm = totd%size
        extra_star = tots%size

        ng = totg/size
        nd = totd/size
        ns = tots/size

        my_ng = ng + int(rank<extra_gas)
        my_nd = nd + int(rank<extra_dm)
        my_ns = ns + int(rank<extra_star)

        start_g = ng*rank + min(extra_gas, rank)
        start_d = nd*rank + min(extra_dm, rank)
        start_s = ns*rank + min(extra_star, rank) 

        self._global_family_slice = {family.gas.name:  slice(start_g, start_g+my_ng),
                                     family.dm.name:   slice(start_d, start_d+my_nd),
                                     family.star.name: slice(start_s, start_s+my_ns)}

        self._family_slice = {family.gas:  slice(0, my_ng),
                              family.dm:   slice(my_ng, my_ng+my_nd),
                              family.star: slice(my_ng+my_nd, my_ng+my_nd+my_ns)}

        # this is a poor hack because pickling of families doesn't work
        
        temp_family_slice = {}
        for fam in self._family_slice: temp_family_slice[fam.name] = self._family_slice[fam]
        
        self._num_particles = len(self.s) + len(self.g) + len(self.d)
        
        #
        # store all of the global and local slices in the root object for future use
        #

        self._all_global_family_slices = comm.gather(self._global_family_slice, root = MPI_ROOT)
        self._all_local_family_slices = comm.gather(temp_family_slice, root = MPI_ROOT)
        self._all_nparts = comm.gather({family.gas.name: len(self.g), 
                                        family.dm.name: len(self.dm), 
                                        family.star.name: len(self.s)}, root = MPI_ROOT)

        if rank == 0 : assert(self._all_global_family_slices[rank] == self._global_family_slice)
        
        for arr_name in ['phi', 'pos', 'vel', 'eps', 'mass', 
                         'tform', 'metals', 'temp', 'rho'] :
            self._distribute_array(arr_name)
            

        self._decorate()
        config['parallel'] = True


        if rank == 0 : 
            for a in self._orig_arrays.keys() : 
                del(self._orig_arrays[a])

            for a in self._orig_family_arrays.keys() : 
                for b in self._orig_family_arrays[a].keys() : 
                    del(self._orig_family_arrays[a][b])
                
    

    def _distribute_array(self, arr_name) : 
        import sys
        from pynbody.array import SimArray

        have_array = None
        dim     = None
        dtype   = None
        units   = None

        arr_families = []
        arr_family_names = []

        if self.mpi_rank == 0 :
            
            #
            # determine what type of an array we have
            #
            arr = {}
            if arr_name in self._orig_arrays.keys() : 
                # use the ordering gas, dark, star for the send/receive buffers
                arr_families = self.families()
                global_array = True
                have_array = True

            elif arr_name in self._orig_family_arrays.keys() : 
                for f in self._orig_family_arrays[arr_name].keys() : 
                    arr_families.append(f)
                global_array = False
                have_array = True
            
            else :
                have_array = False
#

            for fam in arr_families : 
                if global_array : 
                    arr[fam] = self._orig_arrays[arr_name][self._orig_family_slice[fam]]
                else : 
                    arr[fam] = self._orig_family_arrays[arr_name][fam]

                arr_family_names.append(fam.name)
                
            if len(arr[fam].shape) > 1 : dim = 3
            else                       : dim = 1
            dtype = arr[fam].dtype
            units = arr[fam].units

        self.mpi_comm.Barrier()
        have_array = self.mpi_comm.bcast(have_array, root = MPI_ROOT)
        print have_array
        if not have_array :
            return
           
        # broadcast the dimensionality
        dim = self.mpi_comm.bcast(dim, root=MPI_ROOT)
            
        # broadcast the type
        dtype = self.mpi_comm.bcast(dtype, root=MPI_ROOT)

        # broadcast the units
        units = self.mpi_comm.bcast(units, root=MPI_ROOT)

        # broadcast the array families that will be sent
        arr_family_names = self.mpi_comm.bcast(arr_family_names, root=MPI_ROOT)
        


        # set the families
        arr_families = []
        for name in arr_family_names: 
            arr_families.append(family.get_family(name))

        # create the MPI.Status object for checking the request state
        #mpi_stat = MPI.Status()
        
        for fam in arr_families : 
            self._create_family_array(arr_name,fam,ndim=dim,dtype=dtype)

        if self.mpi_rank == 0:
                
            # distribute the array to all nodes
            for f in arr_families : 
                # first put the data into our own array
                self[f][arr_name] = arr[f][self._global_family_slice[f.name]]

            # now send to the rest
            for i in range(1,self.mpi_size) :
                curr_slices = self._all_global_family_slices[i]
                    
                for f in arr_families: 
                    self.mpi_comm.Send(arr[f][curr_slices[f.name]], dest=i)
                    

        if self.mpi_rank > 0 :
            for f in arr_families :
                self.mpi_comm.Recv(self[f][arr_name],source=0)
        

    def _load_array(self, *args, **kwargs) :
        
        have_array = None

        if self.mpi_rank == 0: 
            # trick pynbody into thinking this is the full snapshot
            npart = self._num_particles
            self._num_particles = self._global_tot
            try: 
                super(ParallelTipsySnap,self)._load_array(*args,**kwargs)
                self._num_particles = npart
                self._orig_arrays = {}
                self._orig_arrays[args[0]] = self._arrays[args[0]]
                del(self._arrays[args[0]])
                have_array = True
            except : 
                have_array = False

        have_array = self.mpi_comm.bcast(have_array,root=MPI_ROOT)
        if not have_array : 
            raise IOError
        else:
            self._distribute_array(args[0])

            
def generate_empty(ngas,ndark,nstar) :
    import pynbody 

    s = pynbody.snapshot._new(gas=ngas,dark=ndark,star=nstar)
    s['pos'] = np.random.rand(len(s),3)
    s['vel'] = np.random.rand(len(s),3)
    s['mass'] = np.random.rand(len(s))
    if nstar > 0: 
        s.s['tform'] = np.random.rand(nstar)
        s.s['metals'] = np.random.rand(nstar)
    if ngas > 0:
        s.g['metals'] = np.random.rand(ngas)
        s.g['temp'] = np.random.rand(ngas)
    
    s.write(filename='test_file', fmt=pynbody.tipsy.TipsySnap)

