import numpy as _N
import scipy.signal as _ssig
import mne.time_frequency as mtf
import mne
import pickle
import os
import sys

from sumojam.devscripts.cmdlineargs import process_keyval_args
import GCoh.chronux_py.chronux_funcs as _chrf
import preprocess_ver

import GCoh.datconfig as datconf
#import GCoh.utils.dir_util as _du

dataset =    datconf._SIM
c    = 2
if dataset == datconf._SIM:
    Fs   = 200
if dataset == datconf._RPS or dataset == datconf._STROOP:
    Fs   = 300
elif dataset == datconf._GONOGO:
    Fs   = 500
elif dataset == datconf._AAFFECT:
    Fs   = 500

#eeg_date_time = "Jan092020_15_05_39"
#eeg_date_time = "Jan092020_14_00_00"
eeg_date_time = "Jan082020_17_03_48"
# eeg_date_time = "Aug122020_13_17_39"
#eeg_date_time ="Aug122020_12_52_44"
eeg_date_time ="Aug182020_15_45_27"
eeg_date_time ="Aug182020_16_44_18"
eeg_date_time="Aug182020_16_25_28"
eeg_date_time="Dec102020_12_54_24"
eeg_date_time="Dec102020_13_48_05"
#eeg_date_time= "Aug122020_13_30_23"
#eeg_date_time = "gonogo_0519"
#eeg_date_time = "aaffect_062"

artrmv_ver = 1
gcoh_ver    = 1

process_keyval_args(globals(), sys.argv[1:])
wnd, wnd_mov      = preprocess_ver.get_win_slideby(gcoh_ver)
hf_wnd  = wnd//2

ch_w_CM, rm_chs, ch_names, ch_types = datconf.getConfig(dataset, sim_nchs=10)

print(dataset)
X_cm    = _N.loadtxt(datconf.getDataFN(dataset, "%(edt)s_artfctrmvd/v%(av)d/%(edt)s_artfctrmvd_v%(av)d.dat" % {"edt" : eeg_date_time, "av" : artrmv_ver}))

N       = X_cm.shape[0]

if os.access(datconf.getDataFN(dataset, "%(edt)s_artfctrmvd/v%(v)d/bad_chs.dat" % {"edt" : eeg_date_time, "v" : artrmv_ver}), os.F_OK):
    bad_chs = _N.loadtxt(datconf.getDataFN(dataset, "%(edt)s_artfctrmvd/v%(v)d/bad_chs.dat" % {"edt" : eeg_date_time, "v" : artrmv_ver}), dtype=_N.int, ndmin=1)
    rm_chs.extend(bad_chs)
ch_picks = _N.setdiff1d(ch_w_CM, _N.array(rm_chs))

###  drop CM
_X    =_N.array(X_cm[:, ch_w_CM])

ws = _N.hamming(21)
ws /= _N.sum(ws)
for i in range(_X.shape[1]):
    tx = _X[:, i]
    tp = _ssig.filtfilt(ws, 1, tx)
    _X[:, i] = _X[:, i] - tp

X = _N.empty((1, _X.shape[1], _X.shape[0]))
X[0] = _X.T
#  params

ns      = int(_N.round(X.shape[2]/wnd_mov)) - wnd//wnd_mov

#info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=Fs)
info = mne.create_info(ch_names=(_N.array(ch_names)[ch_picks]).tolist(), ch_types=(_N.array(ch_types)[ch_picks]).tolist(), sfreq=Fs)

datconf.set_montage(dataset, info)

dpss_bw   = 7
dpss, eigvals, adaptive = mtf.multitaper._compute_mt_params(wnd, Fs, dpss_bw, True, True)
#epochs = mne.EpochsArray(X, info)
epochs = mne.EpochsArray(X[:, ch_picks], info)

fMin = 5
fMax = 50
f, findx = _chrf.getfgrid(Fs, wnd, [fMin, fMax])

#if wnd == 384:
#    findx = _N.array(findx[0:-1])
if len(_N.where(f == fMax)[0]) == 1:  #  csd treats lims as [fMin fMax)?
    findx = _N.array(findx[0:-1])        

n_picked_chs = len(ch_picks)
EVS  = _N.empty((ns, len(findx), n_picked_chs, n_picked_chs), dtype=_N.complex)

#for i in range(ns-16):

csd_all_freq = _N.empty((len(findx), n_picked_chs), dtype=_N.complex)

params = {}
dSv = _N.zeros((n_picked_chs, n_picked_chs))

n_bins = (N - wnd) // wnd_mov + 1

Cvec   = _N.zeros((n_bins, len(findx), 2, n_picked_chs), dtype=_N.complex)
Ctot   = _N.zeros((n_bins, len(findx)))


for ni in range(n_bins):
    t0 = ni*wnd_mov
    t1 = t0 + wnd
    
    X[0, t0:t1]
    print("%(0)d  %(1)d" % {"0" : t0, "1" : t1})
    """
    params["tapers"]=[3, 4]
    params["Fs"] = Fs
    params["fpass"]=[5, 50]
    tapers        = [3, 4]
    """
    
    #t1, t2, Ctot, Cvec, Cent, fs = _chrf.CrossSpecMatc(X[0, t0:t1],sub_wnd, params=params);
    #_chrf.CrossSpecMatc(X[0, t0:t1],sub_wnd, params=params);

    #csd = mtf.csd_multitaper(epochs, tmin=(t0/Fs), tmax=(t1/Fs), fmin=5, fmax=50, n_fft=wnd, picks=ch_picks, bandwidth=dpss_bw, adaptive=False, low_bias=True, projs=None, n_jobs=1, verbose=False)
    csd = mtf.csd_multitaper(epochs, tmin=(t0/Fs), tmax=(t1/Fs), fmin=fMin, fmax=fMax, n_fft=wnd, bandwidth=dpss_bw, adaptive=False, low_bias=True, projs=None, n_jobs=1, verbose=False)

    max
#     #for fi in range(len(findx)):
    for fi in range(len(findx)):
        csd_dat = csd.get_data(index=fi)
        U, Sv, Vh = _N.linalg.svd(csd_dat)
        #_N.fill_diagonal(dSv, Sv);
        Ctot[ni, fi]=Sv[0]**2/_N.sum(Sv**2)
        Cvec[ni, fi]=U[:,0:2].T

arr_ch_picks = _N.array(ch_picks)
pkldat = {"VEC" : Cvec, "Cs" : Ctot, "fs" : f[findx], "chs_picks" : arr_ch_picks, "dpss_bw" : dpss_bw}

dmp = open(datconf.getDataFN(dataset, "%(edt)s_artfctrmvd/v%(av)d/%(edt)s_gcoh_%(w)d_%(wm)d_v%(av)d%(gv)d.dmp" % {"edt" : eeg_date_time, "w" : wnd, "wm" : wnd_mov, "av" : artrmv_ver, "gv" : gcoh_ver, "dd" : datdir}), "wb")
pickle.dump(pkldat, dmp, -1)
dmp.close()

