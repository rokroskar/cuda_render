#!/usr/bin/env python

import struct
import numpy as np
import gzip


class TipsyFile:
    """

    Class for a tipsy file with some basic routines.

    Example:

    >>> import tipsyio
    >>> f1 = tipsyio.TipsyFile(<filename>)
    >>> f1.cofm('pot')

    """

    def __init__(self, filename):
        self.h, self.g, self.d, self.s = self.rtipsy(filename)

        self.filename = filename

        # set some pointers for convenience

        # set pointers to the gas properties
        self.gm = self.g[:,0]
        self.gx = self.g[:,1]
        self.gy = self.g[:,2]
        self.gz = self.g[:,3]
        self.gvx = self.g[:,4]
        self.gvy = self.g[:,5]
        self.gvz = self.g[:,6]
        self.grho = self.g[:,7]
        self.gtemp = self.g[:,8]
        self.geps = self.g[:,9]
        self.gmetals = self.g[:,10]
        self.gpot = self.g[:,11]

        # set pointers to the dark properties
        self.dm = self.d[:,0]
        self.dx = self.d[:,1]
        self.dy = self.d[:,2]
        self.dz = self.d[:,3]
        self.dvx = self.d[:,4]
        self.dvy = self.d[:,5]
        self.dvz = self.d[:,6]
        self.deps = self.d[:,7]
        self.dpot = self.d[:,8]

        # set pointers to the star properties
        self.sm = self.s[:,0]
        self.sx = self.s[:,1]
        self.sy = self.s[:,2]
        self.sz = self.s[:,3]
        self.svx = self.s[:,4]
        self.svy = self.s[:,5]
        self.svz = self.s[:,6]
        self.smetals = self.s[:,7]
        self.stform = self.s[:,8]
        self.seps = self.s[:,9]
        self.spot = self.s[:,10]

     
        
    def rtipsy(self, filename):
        """

        Read a tipsy file and output numpy arrays of various particle attributes.
        
        Usage: h, g, d, s = rtipsy(filename)
        
        return values are:
        
        h - a key:value pair array with the tipsy header values
        g, d, s are numpy arrays of dimensions:
        
        Note the structure of these arrays:
        g.shape() = [h.ng,12] indices = [mass,x,y,z,vx,vy,vz,rho,tempg,eps,metals,pot]
        d.shape() = [h.nd, 9] indices = [mass,x,y,z,vx,vy,vz,eps,pot]
        s.shape() = [h.ns,11] indices = [mass,x,y,z,vx,vy,vz,metals,tform,eps,pot]

    
        """
        
    
        if filename[-2:] == 'gz':
            f = gzip.open(filename, "rb")
        else:
            f = open(filename, "rb")
            
        t, n, ndim, ng, nd, ns = struct.unpack(">dlllll", f.read(28))
            
        h = {'time':t,'n':n,'ndim':ndim,'ng':ng,'nd':nd,'ns':ns}
            
        dum = f.read(4)  # tipsy files are padded
            
            
        g = fread(f,12*ng,'f').reshape((ng,12)).byteswap()
        d = fread(f,9*nd,'f').reshape((nd,9)).byteswap()
        s = fread(f,11*ns,'f').reshape((ns,11)).byteswap()
        
            
        f.close()
            
        return h, g, d, s


    
    def cofm(self, mode):
        """

        Determine the center of mass according to <mode> and shift particles.
        <mode> can only be 'pot' at the moment
        """

        if mode != 'pot':
            print "Only centering on potential minimum possible at the moment."
            return
        
        x = np.concatenate((self.g[:,1], self.d[:,1], self.s[:,1]))
        y = np.concatenate((self.g[:,2], self.d[:,2], self.s[:,2]))
        z = np.concatenate((self.g[:,3], self.d[:,3], self.s[:,3]))
        pot = np.concatenate((self.g[:,11], self.d[:,8], self.s[:,10]))

        pot_min = np.where(pot == min(pot))

        xc, yc, zc = x[pot_min].item(), y[pot_min].item(), z[pot_min].item()

        self.g[:,1:4] -= xc, yc, zc
        self.d[:,1:4] -= xc, yc, zc
        self.s[:,1:4] -= xc, yc, zc

        print xc, yc, zc
        

#------------------------
# End of TipsyFile class
#------------------------


#
# fread/fwrite are taken from http://projects.scipy.org/scipy/attachment/ticket/14/numpyIO.py
# beacause the version in scipy.io doesn't work with gzipped files
#
#=======================
def fread(fid, num=None, readType='f', mem_type=None, byteswap=0):
#=======================

#=== how many bytes per number
  if readType=='1' or readType=='b' or readType=='c' or readType=='B':
    byteSize = 1
  elif readType=='s' or readType=='w' or readType=='h' or readType=='H':
    byteSize = 2
  elif readType=='f' or readType=='i' or readType=='I'\
    or readType=='l' or readType=='u':
    byteSize = 4
  elif readType=='d' or readType=='F':
    byteSize = 8
  elif readType=='D':
    byteSize = 16
  else:
    print readType
    raise TypeError

#=== figure out num
  if not num:
    import os
    fName = fid.name
    fileSize = os.path.getsize(fName)
    num = fileSize / byteSize

#=== figure out mem_type
  if not mem_type:
    mem_type = readType

#=== read in
  nByte = num * byteSize

  if byteswap==1:
    a1 = np.fromstring(fid.read(nByte),readType).byteswapped()
  else:
    a1 = np.fromstring(fid.read(nByte),readType)

#=== mem_type
  if readType!=mem_type:
    a1 = a1.astype(mem_type)

#=== return
  return a1

#=======================
def fwrite(fid, num, myArray, write_type=None, byteswap=0):
#=======================

#=== figure out write_type
  mem_type = myArray.typecode()
  if write_type==0 or write_type==None:
    write_type = mem_type

#=== figure out total size of myArray
  temp1 = myArray.shape
  totalSize = 1
  for i in temp1: 
    totalSize = totalSize * i

  if num==None:
    num = totalSize

#=== if not all elements of myArray are written
  if num!=totalSize:
    myArray1 = np.ravel(myArray)[:num]
  else:
    myArray1 = myArray

#=== write
  if mem_type!=write_type:
    if byteswap: 
      fid.write( myArray1.astype(write_type).byteswapped().tostring() )
    else:
      fid.write( myArray1.astype(write_type).tostring() )
  else:
    if byteswap: 
      fid.write( myArray1.byteswapped().tostring() )
    else:
      fid.write( myArray1.tostring() )
