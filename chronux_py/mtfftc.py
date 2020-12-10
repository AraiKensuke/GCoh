import numpy as _N

def mtfftc(data, tapers, nfft, Fs):
    # function J=mtfftc(data,tapers,nfft,Fs)
    # % Multi-taper fourier transform - continuous data
    # %
    # % Usage:
    # % J=mtfftc(data,tapers,nfft,Fs) - all arguments required
    # % Input: 
    # %       data (in form samples x channels/trials or a single vector) 
    # %       tapers (precalculated tapers from dpss) 
    # %       nfft (length of padded data)
    # %       Fs   (sampling frequency)
    # %                                   
    # % Output:
    # %       J (fft in form frequency index x taper index x channels/trials)
    
    #  data size nfft
    #  data_rep now size [
    data_rep = _N.repeat(data[_N.newaxis], tapers.shape[0], axis=0)  #  

    J=_N.fft.fft(data_rep * tapers, nfft)/Fs;   # fft of projected data
    return J
