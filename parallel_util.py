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

