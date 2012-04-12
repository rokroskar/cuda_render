#
#
# a set of routines for doing frequency analysis on fourier components
# of the particle distribution
#

import numpy as np


def read_fourier_data(list):

    fl = open(list)

    line = fl.readline()

    f = np.load(line[:-1])

    array_shape = f['amp'].shape
        
    fourier_dtype = np.dtype(
        [('time', np.double           ),
         ('c',    np.complex, array_shape   ),
         ('amp',  np.float,   array_shape   ),
         ('phi',  np.float,   array_shape   ),
         ('mass', np.float,   array_shape[1]),
         ('den',  np.float,   array_shape[1]),
         ('bins', np.float,   array_shape[1])])


    fourier_data_array = np.zeros(0,dtype=fourier_dtype)    

    fl.seek(0)

    for i,line in enumerate(fl):
        if line[-1:] == '\n': line = line[:-1]
        new_data = np.zeros(1,dtype=fourier_dtype)
        
        f = np.load(line)
        
        for field in f.files:
            new_data[field] = f[field]

        if 'time' not in f.files:
            new_data['time'] = 1.0 + 1e-3*(i+1)

        fourier_data_array = np.append(fourier_data_array,new_data)
        

    return fourier_data_array


def convert_fourier_data(list, name):
    data = read_fourier_data(list)
    np.savez(name,data=data)


def fft_coefficients(file, t1, t2, m):
    data = np.load(file)

    data = data['data']

    pwr_re = np.zeros((len(data),len(data[0]['bins'])), dtype = np.float)
    pwr_im = np.zeros((len(data),len(data[0]['bins'])), dtype = np.float)

    for i in np.arange(pwr_re.shape[1]):
        pwr_re[:,i] = np.abs(np.fft.fft(np.real(data['c'][:,2,i])))
        pwr_im[:,i] = np.abs(np.fft.fft(np.imag(data['c'][:,2,i])))

    pwr_tot = np.sqrt(pwr_re**2 + pwr_im**2)

    freq = np.fft.fftfreq(len(data), np.diff(data['time'])[0])*2*np.pi/m
    
    return pwr_tot, freq, data[0]['bins']

