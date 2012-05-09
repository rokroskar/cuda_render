  def _load_ahf_particles_better(self,filename) : 
        from StringIO import StringIO

        f = util.open_(filename)
        data = f.readlines()
        f.close()
        
        if filename.split("z")[0][-1] is "." : self.isnew = True
        else : self.isnew = False
        # tried readlines, which is fast, but the time is spent in the
        # for loop below, so sticking with this (hopefully) more readable 
        
        if self.isnew:
            nhalos=int(data[0])
        else: 
            nhalos = self._nhalos
        ng = len(self.base.gas)
        nds = len(self.base.dark) + len(self.base.star)
        l_start = 1
        nblock = 1024

        for h in xrange(nhalos) :
            nparts = int(data[l_start])
            nleft = nparts
            ndone = 0L
            # wow,  AHFstep has the audacity to switch the id order to dark,star,gas
            # switching back to gas, dark, star
            
            partids = np.empty(len(nparts),dtype='int')
            while nleft : 
                start = l_start + 1 + ndone
                ndo = min(nleft,nblock)
                end   = start + ndo
                partids[ndone:ndone+ndo] = np.loadtxt(StringIO('\n'.join(data[start:end])))
                ndone += ndo
                nleft -= ndo
                
    def _load_ahf_particles_single_block(x) : 
        self, data, startend = x
        from StringIO import StringIO
        return np.loadtxt(StringIO('\n'.join(data[startend[0]:startend[1]])))
