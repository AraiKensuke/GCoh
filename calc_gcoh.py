import numpy as _N
import scipy.signal as _ssig
import mne.time_frequency as mtf
import mne
import pickle
import os
import sys

from sumojam.devscripts.cmdlineargs import process_keyval_args
import chronux_funcs as _chrf
import preprocess_ver

_RPS        = 1
_STROOP     = 2
_GONOGO     = 3
_AAFFECT    = 4
dataset =    _GONOGO
c    = 2
if dataset == _RPS or dataset == _STROOP:
    Fs   = 300
elif dataset == _GONOGO:
    Fs   = 500
elif dataset == _AAFFECT:
    Fs   = 500

#eeg_date_time = "Apr112020_21_26_01"
#eeg_date_time = "Apr112020_13_00_00"
#eeg_date_time = "Apr242020_09_00_00"
#eeg_date_time = "Jan092020_15_05_39"
#eeg_date_time = "Jan092020_14_00_00"
#eeg_date_time = "Apr242020_16_53_03"
#eeg_date_time = "Apr242020_13_00_00"
#eeg_date_time = "Apr152020_13_00_00"   #  35 seconds    #  15:04:32
#eeg_date_time = "Apr182020_22_02_03"   #  35 seconds    #  15:04:32
#eeg_date_time = "Apr182020_13_00_00"   #  35 seconds    #  15:04:32
#eeg_date_time = "May042020_22_23_04"
#eeg_date_time = "May042020_14_00_00"
#eeg_date_time = "Apr242020_00_00_00"
#eeg_date_time = "Jan092020_14_55_38"
eeg_date_time = "Jan082020_17_03_48"
#eeg_date_time = "Jan082020_16_56_08"
# eeg_date_time = "Aug122020_13_17_39"
#eeg_date_time ="Aug122020_12_52_44"
eeg_date_time ="Aug182020_15_45_27"
eeg_date_time ="Aug182020_16_44_18"
eeg_date_time ="Aug182020_16_25_28"
#eeg_date_time= "Aug122020_13_30_23"
#eeg_date_time = "May052020_21_08_01"
#eeg_date_time = "May052020_13_00_00"
#eeg_date_time = "May052020_15_00_00"
#eeg_date_time = "May082020_23_34_58"
#eeg_date_time = "May142020_23_31_04"
#eeg_date_time  = "May142020_23_16_34"   #  35 seconds    #  15:04:32
#eeg_date_time  = "May142020_14_00_00"   #  35 seconds    #  15:04:32
eeg_date_time = "Jan012019_04_00_00"
eeg_date_time = "gonogo_0519"
#eeg_date_time = "aaffect_062"

artrmv_ver = 1
gcoh_ver    = 2

process_keyval_args(globals(), sys.argv[1:])
wnd, wnd_mov      = preprocess_ver.get_win_slideby(gcoh_ver)
hf_wnd  = wnd//2


if dataset == _RPS or dataset == _STROOP:
    ch_w_CM  =   _N.arange(21)
    rm_chs   =  [8]
    ch_names=["P3", "C3", "F3", "Fz", "F4",   
              "C4", "P4", "Cz", "Pz", "A1",    #  "Pz" is "CM"
              "Fp1","Fp2","T3", "T5", "O1",
              "O2", "F7", "F8", "A2", "T6",
              "T4"]
    ch_types=["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
              "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
              "eeg", "eeg", "eeg", "eeg", "eeg"]
    datdir = "DSi_dat"
if dataset == _AAFFECT:
    ch_w_CM  =   _N.arange(19)
    rm_chs   =  []
    ch_names = ["Fp1", "Fp2", "F7", "F3","Fz", "F4", 
                "F8", "T3", "C3", "Cz", "C4", "T4", 
                "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
    ch_types=["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
              "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
              "eeg", "eeg", "eeg"]
    datdir = "aaffect_dat"
elif dataset == _GONOGO:
    ch_w_CM  =   _N.arange(31)
    rm_chs   =  []
    ch_names = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "PZ", "T5'", "T6'", "O1'", "O2'", "P3\"", "P4\"", "PZ\"", "OZ", "I", "CB1\"", "CB2\"", "CB1", "CB2"]
    ch_types=["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
              "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
              "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg",
              "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg"]
    datdir = "gonogo_dat"
    mntg = mne.channels.read_custom_montage("gonogo_dat/channel_loc.loc")

X_cm    = _N.loadtxt("%(dd)s/%(edt)s_artfctrmvd/v%(av)d/%(edt)s_artfctrmvd_v%(av)d.dat" % {"edt" : eeg_date_time, "av" : artrmv_ver, "dd" : datdir})
#X_cm     = _N.loadtxt("DSi_dat/test.dat")
#X_cm    = _N.loadtxt("DSi_dat/test.dat")

N       = X_cm.shape[0]

if os.access("%(dd)s/%(edt)s_artfctrmvd/v%(v)d/bad_chs.dat" % {"edt" : eeg_date_time, "v" : artrmv_ver, "dd" : datdir}, os.F_OK):
    bad_chs = _N.loadtxt("%(dd)s/%(edt)s_artfctrmvd/v%(v)d/bad_chs.dat" % {"edt" : eeg_date_time, "v" : artrmv_ver, "dd" : datdir}, dtype=_N.int, ndmin=1)
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
if dataset == _RPS or dataset == _STROOP:
    info.set_montage("standard_1020")
elif dataset == _GONOGO:
    info.set_montage(mntg)
elif dataset == _AAFFECT:
    info.set_montage("standard_1020")

dpss_bw   = 7
dpss, eigvals, adaptive = mtf.multitaper._compute_mt_params(wnd, Fs, dpss_bw, True, True)
#epochs = mne.EpochsArray(X, info)
epochs = mne.EpochsArray(X[:, ch_picks], info)

f, findx = _chrf.getfgrid(Fs, wnd, [5, 50])
if wnd == 384:
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
    csd = mtf.csd_multitaper(epochs, tmin=(t0/Fs), tmax=(t1/Fs), fmin=5, fmax=50, n_fft=wnd, bandwidth=dpss_bw, adaptive=False, low_bias=True, projs=None, n_jobs=1, verbose=False)

    #for fi in range(len(findx)):
    for fi in range(len(findx)):
        csd_dat = csd.get_data(index=fi)
        U, Sv, Vh = _N.linalg.svd(csd_dat)
        #_N.fill_diagonal(dSv, Sv);
        Ctot[ni, fi]=Sv[0]**2/_N.sum(Sv**2)
        Cvec[ni, fi]=U[:,0:2].T

arr_ch_picks = _N.array(ch_picks)
pkldat = {"VEC" : Cvec, "Cs" : Ctot, "fs" : f[findx], "chs_picks" : arr_ch_picks, "dpss_bw" : dpss_bw}

dmp = open("%(dd)s/%(edt)s_artfctrmvd/v%(av)d/%(edt)s_gcoh_%(w)d_%(wm)d_v%(av)d%(gv)d.dmp" % {"edt" : eeg_date_time, "w" : wnd, "wm" : wnd_mov, "av" : artrmv_ver, "gv" : gcoh_ver, "dd" : datdir}, "wb")
pickle.dump(pkldat, dmp, -1)
dmp.close()

