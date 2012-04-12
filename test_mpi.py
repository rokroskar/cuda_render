from mpi4py import MPI
import numpy as np

def my_reduction(aa,bb,mpi_datatype) :
#    a = np.frombuffer(aa,dtype='d')
 #   b = np.frombuffer(bb,dtype='d')
    #b += 2*a
    pass

def test_reduction() : 
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    myop = MPI.Op.Create(my_reduction)

    a = np.array(rank+1, dtype='d')
    b = np.array(0,dtype='d')

    comm.Reduce([a,MPI.DOUBLE],[b,MPI.DOUBLE],myop)

    print a, b, (2*np.arange(1,size+1)).sum()

    MPI.Op.Free(myop)

