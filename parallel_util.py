import pynbody
import multiprocessing
import numpy as np
from functools import wraps

def run_parallel(func, single_args, repeat_args, 
                 processes=int(pynbody.config['number_of_threads'])) : 
    """
    
    Run a function in parallel using the python multiprocessing module. 

    **Input**

    *func* : the function you want to run in parallel; there are some
             restrictions on this, see Usage below

    *single_args* : a list of arguments; only one argument from the
                    list gets passed to a single execution instance of
                    func

    *repeat_args* : a list of arguments; all of the arguments in the
                    list get passed to each execution of func

    **Optional Keywords**

    *processes* : the number of processes to spawn; default is
                  pynbody.config['number of threads']. Set to 1 for
                  testing and using the serial version of map. 

    **Usage** 

    Note that the function must accept only a single argument as a
    list and then expand that argument into individual inputs. 

    For example:

    def f(a) : 
       x,y,z = a
       return x*y*z


    Also note that the function must have a try/except clause to look
    for a KeyboardInterrupt, otherwise it's impossible to stop
    execution with ctrl+c once the code is running in the Pool. To
    facilitate this, you can use the interruptible decorator.

    To use the example from above: 

    from parallel_util import interruptible 

    @interruptible
    def f(a) : 
        x,y,z = a
        return x*y*z
 
    'single_args' can be thought of as the argument you are
    parallelizing over -- for example, if you are running the same
    code over a number of different files, then 'single_args' might be
    the list of filenames. 

    Similarly, 'repeat_args' are the arguments that might modify the
    behavior of 'func', but are the same each time you run the code.

    """
    

    from multiprocessing import Pool 
    import itertools

    args = []

    if len(repeat_args) > 0:
        for arg in repeat_args: 
            args.append(itertools.repeat(arg))
        all_args = itertools.izip(single_args, *args)
    else : 
        all_args = single_args

    if processes==1 : 
        res = map(func, all_args)

    else : 
        pool = Pool(processes=processes)
        try : 
            res = pool.map(func, all_args)
            pool.close()

        except KeyboardInterrupt : 
            pool.terminate()

        finally: 
            pool.join()
    
    return res

class KeyboardInterruptError(Exception): pass

def interruptible(func) : 
    @wraps(func)
    def newfunc(*args, **kwargs):
        try : 
            return func(*args, **kwargs)
        except KeyboardInterrupt: 
            raise KeyboardInterruptError()
    return newfunc


from IPython.parallel import Client

class ParallelTipsySnap(pynbody.tipsy.TipsySnap) : 
    def __init__(self, filename, **kwargs) : 
        super(ParallelTipsySnap,self).__init__(filename,**kwargs)
        rc = Client()
        dview = rc[:]
        nengines = len(rc)
        
        self.rc,self.dview,self.nengines = [rc,dview,nengines]

        dview.execute('import pynbody')

        # set up particle slices
        
        for engine, particle_ids in zip(rc,self._get_particle_ids()) : 
            engine.push({'particle_ids':particle_ids, 'filename':filename})
            engine.execute('s = pynbody.load(filename,take=particle_ids)')
            
    def _get_particle_ids(self) :
        ng = len(self.g) / self.nengines
        nd = len(self.d) / self.nengines
        ns = len(self.s) / self.nengines
        g_start = 0
        d_start = 0
        s_start = 0
        
        while True:
            yield range(g_start,g_start+ng)+range(d_start,d_start+nd)+range(s_start,s_start+ns)
            g_start+=ng
            d_start+=nd
            s_start+=ns

            if (g_start > len(self.g)) & (d_start > len(self.d)) & (s_start > len(self.s)) : 
                raise StopIteration

            

        
    def __getitem__(self,i) : 
        if isinstance(i,str) :
            self.dview.execute("arr = s['%s']"%i)
            res = pynbody.array.SimArray(self.dview['arr'])
            if len(res.shape) == 3 : shape = (res.shape[0]*res.shape[1],res.shape[2])
            else : shape = (res.shape[0]*res.shape[1])
            return res.reshape(shape)
        return super(ParallelTipsySnap,self).__getitem__(i)
