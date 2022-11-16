import numpy as _N
import scipy.signal as _ssig
import mne.time_frequency as mtf
import mne
import pickle
import os
import sys

import GCoh.chronux_py.chronux_funcs as _chrf
import preprocess_ver

import GCoh.datconfig as datconf

def windowed_gcoh(Fs, wnd, slideby, X_cm, ch_w_ref, ch_picks, info, dpss_bw=7, dpss=None, f=None, findx=None):
    N       = X_cm.shape[0]

    ###  drop reference channel
    hf_wnd  = wnd//2

    _X    =_N.array(X_cm[:, ch_w_ref])

    ws = _N.hamming(21)
    ws /= _N.sum(ws)
    for i in range(_X.shape[1]):
        tx = _X[:, i]
        tp = _ssig.filtfilt(ws, 1, tx)
        _X[:, i] = _X[:, i] - tp

    X = _N.empty((1, _X.shape[1], _X.shape[0]))
    X[0] = _X.T
    #  params

    ns      = int(_N.round(X.shape[2]/slideby)) - wnd//slideby

    if dpss is None:
        dpss, eigvals, adaptive = mtf.multitaper._compute_mt_params(wnd, Fs, dpss_bw, True, True)
    #epochs = mne.EpochsArray(X, info)
    epochs = mne.EpochsArray(X[:, ch_picks], info)

    fMin = 5
    fMax = 50
    if f is None:
        f, findx = _chrf.getfgrid(Fs, wnd, [fMin, fMax])

        if len(_N.where(f == fMax)[0]) == 1:  #  csd treats lims as [fMin fMax)?
            findx = _N.array(findx[0:-1])

    n_picked_chs = len(ch_picks)
    EVS  = _N.empty((ns, len(findx), n_picked_chs, n_picked_chs), dtype=_N.complex)

    csd_all_freq = _N.empty((len(findx), n_picked_chs), dtype=_N.complex)

    dSv = _N.zeros((n_picked_chs, n_picked_chs))

    n_bins = (N - wnd) // slideby + 1

    Cvec   = _N.zeros((n_bins, len(findx), 2, n_picked_chs), dtype=_N.complex)
    Ctot   = _N.zeros((n_bins, len(findx)))


    for ni in range(n_bins):
        t0 = ni*slideby
        t1 = t0 + wnd

        X[0, t0:t1]
        if ni % 20 == 0:
            print("win %(0)d of %(1)d" % {"0" : ni, "1" : n_bins})

        csd = mtf.csd_multitaper(epochs, tmin=(t0/Fs), tmax=(t1/Fs), fmin=fMin, fmax=fMax, n_fft=wnd, bandwidth=dpss_bw, adaptive=False, low_bias=True, projs=None, n_jobs=1, verbose=False)

        for fi in range(len(findx)):
            csd_dat = csd.get_data(index=fi)
            U, Sv, Vh = _N.linalg.svd(csd_dat)
            #_N.fill_diagonal(dSv, Sv);
            Ctot[ni, fi]=Sv[0]**2/_N.sum(Sv**2)
            Cvec[ni, fi]=U[:,0:2].T

    return f, findx, Ctot, Cvec

def trial_by_trial_windowed_gcoh(Fs, wnd, slideby, list_of_X_cm, ch_w_ref, ch_picks, info, dpss_bw=7, dpss=None):
    n_trials = len(list_of_X_cm)

    dpss, eigvals, adaptive = mtf.multitaper._compute_mt_params(wnd, Fs, dpss_bw, True, True)

    f_ret = None
    findx_ret = None

    Ctots = []
    Cvecs = []

    gcoh_L_tot = 0
    tr_bounds     = _N.zeros(n_trials + 1, dtype=_N.int)
    for n in range(n_trials):
        f_ret, findx_ret, Ctot, Cvec = windowed_gcoh(Fs, wnd, slideby, list_of_X_cm[n], ch_w_ref, ch_picks, info, dpss_bw=dpss_bw, f=f_ret, findx=findx_ret)
        Ctots.append(Ctot)
        Cvecs.append(Cvec)
        gcoh_L_tot += Cvec.shape[0]
        tr_bounds[n+1] = gcoh_L_tot

    Ctot_all = _N.empty((gcoh_L_tot, findx_ret.shape[0]))
    Cvec_all = _N.empty((gcoh_L_tot, findx_ret.shape[0], 2, len(ch_picks)), dtype=_N.complex)
    for n in range(n_trials):
        Ctot_all[tr_bounds[n]:tr_bounds[n+1]] = Ctots[n]
        Cvec_all[tr_bounds[n]:tr_bounds[n+1]] = Cvecs[n]

    return f_ret, findx_ret, Ctot_all, Cvec_all, tr_bounds
        
    
