import numpy as _N
import os
from sklearn import mixture
from sklearn.decomposition import PCA

from GCoh.datconfig import getResultFN
import GCoh.datconfig as datconf


#from AIiRPS.utils.dir_util import getResultFN

def cut_zero_ends(y):
    where1 = _N.where(y == 1)[0]
    if where1.shape[0] > 0:
        return where1[0], where1[-1]
    return -1, -1

def break_at_gaps_01(y, meanISImult=20):
    """
    For binary data, if there are lots of 0s, break 
    """
    meanISI = len(y) / _N.sum(y)

    in0 = False
    st0 = -1
    en0 = -1
    gaps= [0]
    for i in range(len(y)):
        if (y[i] == 0) and (not in0):
            in0 = True
            st0 = i
        if (y[i] == 1) and in0:
            in0 = False
            en0 = i
            if (en0 - st0 >= meanISImult*meanISI) and (st0 > 0):
                gaps.append((en0+st0)//2)
    gaps.append(len(y))
    print(gaps)
    return _N.array(gaps)
            

def unique_in_order_of_appearance(labs):
    """
    8 8 9 2 1 4 1 2 3 3 2 2
    --> returns
    8 9 2 1 4 3
    """
    unq_labs = _N.unique(labs)

    first_appear = _N.empty(len(unq_labs), dtype=_N.int)
    iu = -1
    for ul in unq_labs:
        iu += 1
        first_appear[iu] = _N.where(labs == iu)[0][0]
    return _N.array(sorted(range(len(first_appear)), key=lambda k: first_appear[k]))  #  unique in order or appearance
    

def increasing_labels_mapping(labs, startLab=0):
    """
    labs
    """
    
    #  return me a mapping
    labs_cpy = _N.array(labs)
    unq_labs_in_app_ord = unique_in_order_of_appearance(labs)

    maxLab           = _N.max(unq_labs_in_app_ord)

    iLab = maxLab
    for srtd_lab in unq_labs_in_app_ord:
        iLab += 1
        ths = _N.where(labs_cpy == srtd_lab)[0]
        labs_cpy[ths] = iLab

    labs_cpy -= maxLab + 1
    return labs_cpy

def remap_label_by_similarity(nStates, labs, orig_features):
    """
    labs
    """
    #  first find means
    mn = _N.zeros((nStates, orig_features.shape[1]))
    for lb in range(nStates):
        these = _N.where(labs == lb)[0]
        mn[lb] = _N.mean(orig_features[these], axis=0)
        mn[lb] /= _N.sum(mn[lb])

    lb0 = 0
    diff_mn = _N.zeros(nStates)
    labels_left = nStates
    
    mapped  = _N.zeros(nStates, dtype=_N.int)
    mapped[nStates - labels_left] = 0  #  12 10 

    labels_left = nStates - 1
    while labels_left > 0:
        for lb in range(nStates):
            diff_mn[lb] = _N.sum((mn[lb] - mn[lb0])*(mn[lb] - mn[lb0])) if lb != lb0 else 20000
        closest = _N.where(diff_mn == _N.min(diff_mn))[0][0]
        mn[closest] = _N.ones(orig_features.shape[1]) * 20000   # make it really far away

        mapped[nStates - labels_left] = closest  #  12 10 
        labels_left -= 1

    new_labs = _N.zeros(labs.shape[0], dtype=_N.int)

    #  mapped = [0, 4, 2, 5, 3, 1]
    #  original 0 -> 0
    #  original 4 -> 1
    for i in range(1, nStates):
        ths = _N.where(labs == mapped[i])[0]
        new_labs[ths] = i
    return new_labs
    

def rmpd_lab_trnsfrm(rmpd_lab, long):
    """
    """
    L       = rmpd_lab.shape[0]
    iCurrSt = rmpd_lab[0]
    inStateFor = 0
    trnsfrmd  = _N.zeros(L)
    iStart    = 0

    for i in range(1, L):
        if rmpd_lab[i] == iCurrSt:
            inStateFor += 1
        else:
            if inStateFor > long:
                trnsfrmd[iStart:i] = 1
            inStateFor = 0
            iStart     = i
            iCurrSt    = rmpd_lab[i]
    if inStateFor > long:
        trnsfrmd[iStart:L] = 1
        
    return trnsfrmd

def shift_correlated_shuffle(arr, low=1, high=5, local_shuffle=False, local_shuffle_pcs=10):
    #  cut into long pieces, then shuffle the pieces intact
    arr_len = arr.shape[0]

    piece_number = _N.empty(arr_len, dtype=_N.int)
    
    pc = 0

    i0 = 0
    i1 = _N.random.randint(low, high)
    begInd     = []


    while i0 < arr_len:
        i1 = i1 if i1 < arr_len else arr_len
        
        begInd.append(i0)
        piece_number[i0:i1] = pc
        i0 = i1
        i1 = i0+_N.random.randint(low, high)
        pc += 1

    use_pc = _N.arange(pc)
    L = len(use_pc) // local_shuffle_pcs

    if local_shuffle:
        for i in range(local_shuffle_pcs):
            _N.random.shuffle(use_pc[i*L:(i+1)*L])
    else:
        _N.random.shuffle(use_pc)
    arr_shuffled = _N.empty(arr_len, dtype=arr.dtype)   #  new shuffled version of arr

    ifilled = 0
    for ipc in range(pc):
        i = begInd[use_pc[ipc]]
        while (i < arr_len) and (piece_number[i] == use_pc[ipc]):
            arr_shuffled[ifilled] = arr[i]
            ifilled += 1
            i += 1
            
    return arr_shuffled

def shuffle_discrete_contiguous_regions(arr, local_shuffle_pcs=10, local_shuffle=False):
    #  cut into long pieces, then shuffle the pieces intact
    arr_len = arr.shape[0]

    piece_number = _N.empty(arr_len, dtype=_N.int)
    
    pc = 0

    i0 = 0
    i1 = 1
    while (i1 < arr_len) and (arr[i0] == arr[i1]):
        i1 += 1
    
    #i1 = _N.random.randint(low, high)
    begInd     = []

    while i0 < arr_len:
        i1 = i1 if i1 < arr_len else arr_len
        
        begInd.append(i0)
        piece_number[i0:i1] = pc
        i0 = i1
        while (i1 < arr_len) and (arr[i0] == arr[i1]):
            i1 += 1
        #i1 = i0+_N.random.randint(low, high)
        pc += 1

    use_pc = _N.arange(pc)
    L = len(use_pc) // local_shuffle_pcs

    if local_shuffle:
        for i in range(local_shuffle_pcs):
            _N.random.shuffle(use_pc[i*L:(i+1)*L])
    else:
        _N.random.shuffle(use_pc)
    arr_shuffled = _N.empty(arr_len, dtype=arr.dtype)   #  new shuffled version of arr

    ifilled = 0
    for ipc in range(pc):
        i = begInd[use_pc[ipc]]
        while (i < arr_len) and (piece_number[i] == use_pc[ipc]):
            arr_shuffled[ifilled] = arr[i]
            ifilled += 1
            i += 1
            
    return arr_shuffled

def find_or_retrieve_GMM_labels(dataset, eeg_date, eeg_gcoh_name, real_evs, iL, iH, fL, fH, armv_ver, gcoh_ver, which=0, try_K=[1, 2, 3, 4, 5, 6, 7], TRs=[1, 2, 4, 8, 16, 32, 64], manual_cluster=False, ignore_stored=False, do_pca=False, min_var_expld=0.95, dontsave=False, Bayesian=False):
    ###############
    if not dontsave:
        outdir = datconf.getResultFN(dataset, "%(rpsm)s/v%(av)d%(gv)d" % {"rpsm" : eeg_date, "w" : which, "av" : armv_ver, "gv" : gcoh_ver})

        if not os.access(outdir, os.F_OK):
            os.mkdir(outdir)

        fn = "%(od)s/%(eeg)s_%(fL)d-%(fH)d_GMM_labels%(w)d" % {"od" : outdir, "eeg" : eeg_gcoh_name, "w" : which, "fL" : fL, "fH" : fH}


    if (not ignore_stored) and os.access(fn, os.F_OK):
        rmpd_lab = _N.loadtxt(fn, dtype=_N.int)
        nStates  = len(_N.unique(rmpd_lab))
    else:
        print("doing GMM")
        minK    = _N.min(try_K)
        maxK    = _N.max(try_K)

        bics = _N.ones(((maxK-minK), _N.max(TRs)))*1000000
        labs = _N.empty((maxK-minK, _N.max(TRs), real_evs.shape[0]), dtype=_N.int)
        _features = _N.mean(real_evs[:, iL:iH], axis=1)

        if do_pca:
            pca = PCA()
            pca.fit(_features)
            
            proj = _N.einsum("ni,mi->nm", pca.components_, _features)
            print(pca.explained_variance_ratio_)
            maxC = _N.where(_N.cumsum(pca.explained_variance_ratio_) > min_var_expld)[0][0]
            features = proj[0:maxC].T
            print("Using %d features" % maxC)
        else:
            features = _features

        if not Bayesian:
            for K in range(minK, maxK):
                for tr in range(TRs[K]):
                    gmm = mixture.GaussianMixture(n_components=K, covariance_type="full")

                    gmm.fit(features)
                    bics[K-minK, tr] = gmm.bic(features)
                    labs[K-minK, tr] = gmm.predict(features)

            coords = _N.where(bics == _N.min(bics))
            bestLab = labs[coords[0][0], coords[1][0]]   #  indices in 2-D array
            nStates =  list(range(minK, maxK))[coords[0][0]]
            print("@!!!!! ")
            print(coords)
        else:
            gmm = mixture.BayesianGaussianMixture(n_components=15, weight_concentration_prior_type="dirichlet_process", covariance_type="full")
            gmm.fit(features)
            bestLab = gmm.predict(features)
            nStates = len(_N.unique(bestLab))

        rmpd_lab = remap_label_by_similarity(nStates, bestLab, _features) # raw features
        #rmpd_lab = increasing_labels_mapping(bestLab)

        if not dontsave:
            print("saving!!!!!!!!!!!!!!!!  %s" % fn)
            _N.savetxt(fn, rmpd_lab, fmt="%d")

    print("manual_clustering   %s" % str(manual_cluster))
    if manual_cluster:
        if os.access("%s_manual" % fn, os.F_OK):  #  open file of mapping
            print("found manual cluster for %s" % fn)
            mapping = _N.loadtxt("%s_manual" % fn, dtype=_N.int)
            #  col 0 is original cluster ID
            #  col 1 is remapped cluster ID
            where_origs = []
            for imaps in range(mapping.shape[0]):
                orig_ID = mapping[imaps, 0]
                new_ID  = mapping[imaps, 1]
                where_origs.append(_N.where(rmpd_lab == orig_ID)[0])
            for imaps in range(mapping.shape[0]):
                orig_ID = mapping[imaps, 0]
                new_ID  = mapping[imaps, 1]
                rmpd_lab[where_origs[imaps]] = new_ID
        else:
            print("!!  No manual cluster for %s.  Using original mapping" % fn)

    return nStates, rmpd_lab


def find_GMM_labels(_data, try_K=[1, 2, 3, 4, 5, 6, 7, 8, 9], TRs=[1, 2, 4, 8, 16, 32, 64, 64, 64], do_pca=False, min_var_expld=0.95):
    """
    data:  shape (# samples, # features)
    """
    if do_pca:
        pca = PCA()
        pca.fit(_data)

        proj = _N.einsum("ni,mi->nm", pca.components_, _data)
        maxC = _N.where(_N.cumsum(pca.explained_variance_ratio_) > min_var_expld)[0][0]
        data = proj[0:maxC].T
        print("Using %d features" % maxC)
    else:
        data = _data

    ###############
    minK    = _N.min(try_K)
    maxK    = _N.max(try_K)

    bics = _N.ones(((maxK-minK), _N.max(TRs)))*1000000
    labs = _N.empty((maxK-minK, _N.max(TRs), data.shape[0]), dtype=_N.int)

    for K in range(minK, maxK):
        for tr in range(TRs[K]):
            gmm = mixture.GaussianMixture(n_components=K, covariance_type="full")

            gmm.fit(data)
            bics[K-minK, tr] = gmm.bic(data)
            labs[K-minK, tr] = gmm.predict(data)

    coords = _N.where(bics == _N.min(bics))
    bestLab = labs[coords[0][0], coords[1][0]]   #  indices in 2-D array
    rmpd_lab = increasing_labels_mapping(bestLab)

    nStates =  list(range(minK, maxK))[coords[0][0]]

    if do_pca:
        return nStates, rmpd_lab, data
    else:
        return nStates, rmpd_lab


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

def autocorrelate_whatsthisfor(_signal, maxlag, pieces=1, overlap=0):
    """
    my attempt to get autocorrelation from very nonstationary signal, ie
    one where signal might disappear for long blocks of data.
    """
    AC = _N.empty((pieces, maxlag*2+1))

    _sigN = _signal.shape[0]
    sigN = _sigN if pieces == 1 else int((_sigN + (pieces-1) * overlap) / pieces)
    signal = _N.array(_signal, dtype=_N.float)

    l_use_these = []
    for pc in range(pieces):
        strtInd = (sigN - overlap)*pc
        endInd  = strtInd + sigN
        st1, st2 = cut_zero_ends(signal[strtInd:endInd])        
        if st2 - st1 > 3*maxlag:
            sigN_cz = st2+1-st1
            signal[strtInd+st1:strtInd+st2+1] = _signal[strtInd+st1:strtInd+st2+1] - _N.mean(_signal[strtInd+st1:strtInd+st2+1])
        #  
        #nz      = _N.random.randint(strtInd, high=endInd, size=int((endInd-strtInd)*0.01))
        #_signal[nz] = 1  #  
        #_signal[(strtInd+endInd)//2] = 1

            lag0mag = _N.dot(signal[strtInd+st1:strtInd+st2+1], signal[strtInd+st1:strtInd+st2+1]) / (sigN_cz*sigN_cz)

            for lg in range(1, maxlag+1):
                datlen = sigN_cz-lg
                AC[pc, maxlag-lg] = (_N.dot(signal[strtInd+st1:strtInd+st1+datlen], signal[strtInd+st1+lg:strtInd+st1+lg+datlen])/(datlen*datlen)) / lag0mag
                AC[pc, maxlag+lg] = AC[pc, maxlag-lg]
            AC[pc, maxlag] = 1
            l_use_these.append(pc)
    use_these = _N.array(l_use_these)
    print(use_these)
    if len(use_these) > 0:
        return _N.mean(AC[use_these], axis=0)
    return _N.zeros(maxlag*2+1)

def autocorrelate(_signal, maxlag):
    """
    my attempt to get autocorrelation from very nonstationary signal, ie
    one where signal might disappear for long blocks of data.
    """
    x  = _N.arange(-maxlag, maxlag+1)
    AC = _N.empty(maxlag*2+1)

    signal = _signal - _N.mean(_signal)
    sigN = signal.shape[0]

    lag0mag = _N.dot(signal, signal) / (sigN*sigN)

    for lg in range(1, maxlag+1):
        datlen = sigN-lg
        AC[maxlag-lg] = (_N.dot(signal[0:sigN-lg], signal[lg:])/(datlen*datlen)) / lag0mag
        AC[maxlag+lg] = AC[maxlag-lg]
    AC[maxlag] = 1
    return x, AC

def autocorrelate_chunky_01s(_signal, maxlag):
    """
    my attempt to get autocorrelation from very nonstationary signal, ie
    one where signal might disappear for long blocks of data.
    """
    #_signal

    breaks = break_at_gaps_01(_signal, meanISImult=10)
    pieces = len(breaks)-1

    AC = _N.empty((pieces, maxlag*2+1))
    signal = _N.array(_signal, dtype=_N.float)

    #weight by amount of data used to calculate autocorrelation
    weights= _N.ones((pieces, 1))
    for pc in range(pieces):
        strtInd = breaks[pc]
        endInd  = breaks[pc+1]
        
        st1, st2 = cut_zero_ends(_signal[strtInd:endInd])
        # if pieces == 1:    #  compromise
        #     if (st2 - st1 < 3*maxlag) and ((st1 >= 0) and (st2 >= 0)):
        #         midp = int(0.5*(st2+st1))
        #         st1 = strtInd + midp - int(1.6*maxlag)
        #         st2 = strtInd + midp + int(1.6*maxlag)
        #         if st1 < 0:
        #             st2 -= st1
        #             st1 = 0
        #         if st2 > len(_signal):
        #             st1 = st1 - (st2 - len(_signal))
        #             st2 = len(_signal)

        if (st2 - st1 > 3*maxlag) or ((pieces == 1) and (st2 - st1 > 2*maxlag)):
            weights[pc, 0] = st2 - st1
            sigN_cz = st2+1-st1
            signal[strtInd+st1:strtInd+st2+1] = _signal[strtInd+st1:strtInd+st2+1] - _N.mean(_signal[strtInd+st1:strtInd+st2+1])

            lag0mag = _N.dot(signal[strtInd+st1:strtInd+st2+1], signal[strtInd+st1:strtInd+st2+1]) / (sigN_cz*sigN_cz)
            #print("!!!!  %.3f" % lag0mag)
            #print(signal[strtInd+st1:strtInd+st2+1])

            for lg in range(1, maxlag+1):
                datlen = sigN_cz-lg
                AC[pc, maxlag-lg] = (_N.dot(signal[strtInd+st1:strtInd+st1+datlen], signal[strtInd+st1+lg:strtInd+st1+lg+datlen])/(datlen*datlen)) / lag0mag
                AC[pc, maxlag+lg] = AC[pc, maxlag-lg]
            AC[pc, maxlag] = 1
        else:
            weights[pc, 0] = 0
            AC[pc, :] = 0

    weights /= _N.sum(weights)
    return _N.mean(AC*weights, axis=0)

def crosscorrelate(signal1, signal2, maxlag):
    CC = _N.empty(maxlag*2+1)
    sigN= signal1.shape[0]

    s1  = _N.std(signal1)
    s2  = _N.std(signal2)
    u1  = _N.mean(signal1)
    u2  = _N.mean(signal2)

    for lg in range(-maxlag, maxlag+1):
        datlen = sigN-_N.abs(lg)
        if lg > 0:
            CC[maxlag-lg] = _N.dot(signal1[0:datlen], signal2[lg:lg+datlen])/sigN
        else:
            nlg = -lg
            CC[maxlag-lg] = _N.dot(signal1[nlg:nlg+datlen], signal2[0:datlen])/sigN
    return CC / (s1*s2)

#  Ref    =       t1    t2       t3    t4    (A defined at these times)
#  FindIn = i1 i2 i3 i4 i5 i6 i7 i8 i9 i10
#   Given Ref, FindIn, I want to be returned elements in FindIn that occur 
#   just before or just after when elements in Ref occur
def find_elements_before_or_after(tRef, tFindIn, before=True):
    return_inds = _N.empty(tRef.shape[0], dtype=_N.int)
    if before:
        for i in range(tRef.shape[0]):
            inds = _N.where((tRef[i] > tFindIn[0:-1]) & (tRef[i] <= tFindIn[1:]))[0]
            if len(inds) == 0:
                if tRef[i] <= tFindIn[0]:
                    inds = _N.array([0])
                elif tRef[i] >= tFindIn[-1]:
                    inds = _N.array([tFindIn.shape[0]-1])
            return_inds[i] = inds[0]
    else:
        for i in range(tRef.shape[0]):
            inds = _N.where((tRef[i] >= tFindIn[0:-1]) & (tRef[i] < tFindIn[1:]))[0]
            if len(inds) == 0:
                if tRef[i] <= tFindIn[0]:
                    inds = _N.array([0])-1
                elif tRef[i] >= tFindIn[-1]:
                    inds = _N.array([tFindIn.shape[0]-2])
            return_inds[i] = inds[0]+1

    return return_inds

#tRef   = _N.array([4, 9, 13])
#tFindIn= _N.array([1, 3, 4, 5, 8, 10, 11, 14, 15])
#_eu.find_elements_before_or_after(tRef, tFindIn)


    
