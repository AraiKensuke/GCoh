import numpy as _N

# def dpsschk(tapers, N, Fs):
#     """
#     0001 function [tapers,eigs]=dpsschk(tapers,N,Fs)
#     0002 % Helper function to calculate tapers and, if precalculated tapers are supplied,
# 0003 % to check that they (the precalculated tapers) the same length in time as
#     0004 % the time series being studied. The length of the time series is specified
#     0005 % as the second input argument N. Thus if precalculated tapers have
#     0006 % dimensions [N1 K], we require that N1=N.
#     0007 % Usage: tapers=dpsschk(tapers,N,Fs)
#     0008 % Inputs:
#     0009 % tapers        (tapers in the form of:
#     0010 %                                   (i) precalculated tapers or,
#     0011 %                                   (ii) [NW K] - time-bandwidth product, number of tapers)
#     0012 %
#     0013 % N             (number of samples)
#     0014 % Fs            (sampling frequency - this is required for nomalization of
#     0015 %                                     tapers: we need tapers to be such
#     0016 %                                     that integral of the square of each taper equals 1
#     0017 %                                     dpss computes tapers such that the
#     0018 %                                     SUM of squares equals 1 - so we need
#     0019 %                                     to multiply the dpss computed tapers
#     0020 %                                     by sqrt(Fs) to get the right
#     0021 %                                     normalization)
#     0022 % Outputs:
#     0023 % tapers        (calculated or precalculated tapers)
#     0024 % eigs          (eigenvalues)
#     """
#     sz=tapers.shape;
#     if sz[0]==1 and sz(2)==2:
#         [tapers,eigs]=dpss(N,tapers(1),tapers(2));
#         tapers = tapers*sqrt(Fs);
#     elif N~=sz(1):
#         print('seems to be an error in your dpss calculation; the number of time points is different from the length of the tapers')

def getfgrid(Fs, nfft, fpass):
    """
    % Helper function that gets the frequency grid associated with a given fft based computation
    % Called by spectral estimation routines to generate the frequency axes 
    % Usage: [f,findx]=getfgrid(Fs,nfft,fpass)
    % Inputs:
    % Fs        (sampling frequency associated with the data)-required
    % nfft      (number of points in fft)-required
    % fpass     (band of frequencies at which the fft is being calculated [fmin fmax] in Hz)-required
    % Outputs:
    % f         (frequencies)
    % findx     (index of the frequencies in the full frequency grid). e.g.: If
    % Fs=1000, and nfft=1048, an fft calculation generates 512 frequencies
    % between 0 and 500 (i.e. Fs/2) Hz. Now if fpass=[0 100], findx will
    % contain the indices in the frequency grid corresponding to frequencies <
    % 100 Hz. In the case fpass=[0 500], findx=[1 512].
    """
    df=Fs/nfft
    _f = _N.arange(0, Fs, df) # all possible frequencies
    f = _f[0:nfft]

    if len(fpass) != 1:
        findx= _N.where((f>=fpass[0]) & (f<=fpass[-1]))[0]
    else:
        fmin,findx=min(abs(f-fpass));

    return f, findx

def getparams(params):
    """
    Helper function to convert structure params to variables used by the
    various routines - also performs checks to ensure that parameters are
    defined; returns default values if they are not defined.
    
    Usage: [tapers,pad,Fs,fpass,err,trialave,params]=getparams(params)
    
    Inputs:
    params: structure with fields tapers, pad, Fs, fpass, err, trialave
    - optional
    tapers : precalculated tapers from dpss or in the one of the following
forms:  
    (1) A numeric vector [TW K] where TW is the
    time-bandwidth product and K is the number of
    tapers to be used (less than or equal to
    2TW-1). 
    (2) A numeric vector [W T p] where W is the
    bandwidth, T is the duration of the data and p 
is an integer such that 2TW-p tapers are used. In
    this form there is no default i.e. to specify
    the bandwidth, you have to specify T and p as
    well. Note that the units of W and T have to be
    consistent: if W is in Hz, T must be in seconds
and vice versa. Note that these units must also
    be consistent with the units of params.Fs: W can
    be in Hz if and only if params.Fs is in Hz.
    The default is to use form 1 with TW=3 and K=5
    
    pad            (padding factor for the FFT) - optional (can take values -1,0,1,2...). 
    -1 corresponds to no padding, 0 corresponds to padding
    to the next highest power of 2 etc.
    e.g. For N = 500, if PAD = -1, we do not pad; if PAD = 0, we pad the FFT
    to 512 points, if pad=1, we pad to 1024 points etc.
    Defaults to 0.
    Fs   (sampling frequency) - optional. Default 1.
    fpass    (frequency band to be used in the calculation in the form
    [fmin fmax])- optional. 
Default all frequencies between 0 and Fs/2
    err  (error calculation [1 p] - Theoretical error bars; [2 p] - Jackknife error bars
    [0 p] or 0 - no error bars) - optional. Default 0.
    trialave (average over trials when 1, don't average when 0) - optional. Default 0
    Outputs: 
    The fields listed above as well as the struct params. The fields are used
    by some routines and the struct is used by others. Though returning both
    involves overhead, it is a safer, simpler thing to do.
    
    CROSS-REFERENCE INFORMATION ^
    This function calls:
    
    This function is called by:
    """
    keys = params.keys()
    print(keys)
    print(len(keys))

    print("---------------------")
    if (not 'tapers' in keys) or (params['tapers'] is None):
        print('tapers unspecified, defaulting to params.tapers=[3 5]')
        params["tapers"] = [3, 5]
    if (len(keys) > 0) and len(params["tapers"]) == 3:
        #0052     % Compute timebandwidth product
        TW = params["tapers"][1] * params["tapers"][0]
        #% Compute number of tapers
        K  = floor(2*TW - params["tapers"][2]);
        params["tapers"] = [TW,  K]
    if (not "pad" in keys) or (params["pad"] is None):
        params["pad"]  = 0
    if (not "Fs" in keys) or (params["Fs"] is None):
        params["Fs"] =1
    if (not "fpass" in keys) or (params["fpass"] is None):
        params["fpass"]  =[0, params["Fs"] / 2 ]
    if (not "err" in keys) or (params["err"] is None):
        params["err"] =0
    if (not "trialave" in keys) or (params["trialave"] is None):
        params["trialave"] =0

    return params["tapers"], params["pad"], params["Fs"], params["fpass"], params["err"], params["trialave"], params

#function [Sc,Cmat,Ctot,Cvec,Cent,f]=CrossSpecMatc(data,win,params)
def CrossSpecMatc(data,win,params=None):
    """
% Multi-taper cross-spectral matrix - another routine, allows for multiple trials and channels 
% Does not do confidence intervals. Also this routine always averages over trials - continuous process
%
% Usage:
%
% [Sc,Cmat,Ctot,Cvec,Cent,f]=CrossSpecMatc(data,win,params)
% Input: 
% Note units have to be consistent. See chronux.m for more information.
%       data (in form samples x channels x trials) 
%       win  (duration of non-overlapping window)
%       params: structure with fields tapers, pad, Fs, fpass
%       - optional
%           tapers : precalculated tapers from dpss or in the one of the following
%                    forms: 
%                   (1) A numeric vector [TW K] where TW is the
%                       time-bandwidth product and K is the number of
%                       tapers to be used (less than or equal to
%                       2TW-1). 
%                   (2) A numeric vector [W T p] where W is the
%                       bandwidth, T is the duration of the data and p 
%                       is an integer such that 2TW-p tapers are used. In
%                       this form there is no default i.e. to specify
%                       the bandwidth, you have to specify T and p as
%                       well. Note that the units of W and T have to be
%                       consistent: if W is in Hz, T must be in seconds
%                       and vice versa. Note that these units must also
%                       be consistent with the units of params.Fs: W can
%                       be in Hz if and only if params.Fs is in Hz.
%                       The default is to use form 1 with TW=3 and K=5
%
%	        pad		    (padding factor for the FFT) - optional. Defaults to 0.  
%			      	 e.g. For N = 500, if PAD = 0, we pad the FFT 
%			      	 to 512 points; if PAD = 2, we pad the FFT
%			      	 to 2048 points, etc.
%           Fs   (sampling frequency) - optional. Default 1.
%           fpass    (frequency band to be used in the calculation in the form
%                                   [fmin fmax])- optional. 
%                                   Default all frequencies between 0 and Fs/2
% Output:
%       Sc (cross spectral matrix frequency x channels x channels)
%       Cmat Coherence matrix frequency x channels x channels
%       Ctot Total coherence: SV(1)^2/sum(SV^2) (frequency)
%       Cvec leading Eigenvector (frequency x channels)
%       Cent A different measure of total coherence: GM/AM of SV^2s
%       f (frequencies)  
    """
    d = len(data.shape)
    if d==2:
        N,C = data.shape
    #if d==3:
    #    [N,C,Ntr]=size(data)

    if params is None:
        params=[]

    tapers,pad,Fs,fpass,err,trialave,params =getparams(params)

    #clear err trialave params

    nwin=int(_N.round(win*Fs))
    print(nwin)

    nfft=max(2**int(_N.ceil(_N.log2(nwin))+pad), nwin)
    f,findx = getfgrid(Fs,nfft,fpass)
    
    # tapers=dpsschk(tapers,nwin,Fs) #% check tapers
    # Sc=zeros(length(findx),C,C);

    # Nwins=floor(N/nwin);

    # if d==2:# % only one trial
    #     for iwin in range(Nwins):
    #         data1=squeeze(data(1+(iwin-1)*nwin:iwin*nwin,:));
    #         J1=mtfftc(data1,tapers,nfft,Fs);
    #         J1=J1(findx,:,:);
    #         for k=1:C,
    #         for l=1:C,
    #         Sc(:,k,l)=Sc(:,k,l)+squeeze(mean(conj(J1(:,:,k)).*J1(:,:,l),2));
    #         end
    #     end
    # end
    # Sc=Sc/Nwins;
    # end

    # Cmat=Sc;
    # Sdiag=zeros(length(findx),C);
    # for k=1:C:
    #     Sdiag(:,k)=squeeze(Sc(:,k,k));

    # for k=1:C:
    #     for l=1:C:
    #         Cmat(:,k,l)=Sc(:,k,l)./sqrt(abs(Sdiag(:,k).*Sdiag(:,l)));

    # Ctot=zeros(length(findx),1);
    # Cent=Ctot;
    # Cvec=zeros(length(findx),C);
    # for i=1:length(findx):
    #     [u s]=svd(squeeze(Sc(i,:,:)));s=diag(s);
    #     Ctot(i)=s(1).^2/sum(s.^2);
    #     Cent(i)=exp(mean(log(s.^2)))/mean(s.^2);             
    # Cvec(i,:)=transpose(u(:,1));
    # end
        
