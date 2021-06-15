import os
import numpy as _N
import matplotlib.pyplot as _plt
from scipy.signal import hilbert
#import read_taisen as _rt
import scipy.stats as _ss
from mne.preprocessing.ica import corrmap  # noqa
from mne.preprocessing import ICA

import mne   #  mne EEG data
from sklearn.decomposition import PCA

def exc_save(exc_list, put_back=False):
    global ica, raw_tmp, artRmvDir, armv_ver, filt_low, filt_high, bad_chs, orig_dat, bad_chs21, Fs, filt_low, filt_high

    ica.exclude = exc_list
    raw_corrected = raw_tmp.copy()
    ica.apply(raw_corrected)

    out_dat  = raw_corrected.get_data()
    if put_back:
        info_orig  = mne.create_info(bad_chs21, Fs, ch_types="eeg")#, montage="standard_1020")
        raw_orig   = mne.io.RawArray(orig_dat.T, info_orig)
        raw_orig.filter(filt_low, filt_high)

        raw_orig_dat = raw_orig.get_data()
        ich = -1
        for ch in exc_list:
            ich += 1
            out_dat[ch] = raw_orig_dat[ich]

    _N.savetxt("%(dir)s/%(fn)s_artfctrmvd_v%(v)d.dat" % {"dir" : artRmvDir, "fn" : fname, "v" : armv_ver}, out_dat.T, fmt=("%.3f " * 21))
    fp = open("%(dir)s/params.txt" % {"dir" : artRmvDir, "v" : armv_ver}, "w")
    fp.write("exclude: %s\n" % str(ica.exclude))
    fp.write("low  Hz: %s\n" % str(filt_low))
    fp.write("high Hz: %s\n" % str(filt_high))

    # if (bad_chs_inds is not None):
    #     _N.savetxt("%(dir)s/bad_chs.dat" % {"dir" : artRmvDir}, bad_chs_inds, fmt="%d")
    # else:
    #     if os.access("%(dir)s/bad_chs.dat" % {"dir" : artRmvDir}, os.F_OK):
    #        os.unlink("%(dir)s/bad_chs.dat" % {"dir" : artRmvDir})
        
    fp.close()

#fname= "Apr112020_21_26_01"   #  35 seconds    #  15:04:32
#fname= "Apr112020_09_00_00"   #  35 seconds    #  15:04:32
#fname= "Apr242020_13_00_00"   #  35 seconds    #  15:04:32
#fname= "Apr152020_13_00_00"   #  35 seconds    #  15:04:32
#fname= "Apr182020_13_00_00"   #  35 seconds    #  15:04:32
#fname= "Jan092020_14_00_00"   #  35 seconds    #  15:04:32
#
#fname=  "Jan082020_17_03_48"
#fname=   "Jan082020_16_56_08"
#fname= "Jan092020_14_55_38"   #  35 seconds    #  15:04:32
#fname= "May132020_22_52_25"
fname= "Apr242020_16_53_03"   #  35 seconds    #  15:04:32
#fname= "Apr242020_09_00_00"   #  35 seconds    #  15:04:32
#fname= "Apr242020_14_00_00"   #  35 seconds    #  15:04:32
#fname= "May042020_22_23_04"   #  35 seconds    #  15:04:32
#fname= "Aug122020_13_17_39"   #  35 seconds    #  15:04:32
#fname= "May042020_17_00_00"   #  35 seconds    #  15:04:32
#fname= "May052020_21_08_01"   #  35 seconds    #  15:04:32
#fname= "May052020_15_00_00"   #  35 seconds    #  15:04:32
#fname= "May082020_23_34_58"   #  35 seconds    #  15:04:32
#fname= "May102020_22_01_21"   #  35 seconds    #  15:04:32
#fname= "May142020_23_31_04"   #  35 seconds    #  15:04:32
#fname= "May142020_23_16_34"   #  35 seconds    #  15:04:32
#fname= "May142020_13_00_00"   #  35 seconds    #  15:04:32
#fname="Aug122020_12_52_44"
#fname="Aug182020_13_55_51"
#fname="Aug182020_16_44_18"
#fname= "Aug122020_13_30_23"
#fname= "Jul042020_12_00_00"
#fname= "Jan012019_15_00_00"
#fname= "May262021_13_18_41"
#fname= "May262021_13_18_43"
fname= "May262021_14_06_30"
fname= "May262021_14_37_19"
#fname= "May262021_14_37_22"
fname= "May262021_15_05_54"
#fname= "Jun092021_12_31_51"
#fname= "Jun092021_12_54_23"
# fname= "Jun092021_17_57_45"
# fname= "Jun092021_15_22_58"
# fname= "Jun092021_12_42_00"
# fname= "Jun092021_13_06_34"
# fname= "Jun092021_18_09_37"
# fname= "Jun092021_15_35_22"
# fname= "Jun092021_13_27_01"
fname="Apr052021_17_50_15"
fname="Apr052021_17_52_50"
fname="Apr052021_18_01_35"
fname="Apr052021_18_12_12"
#fname="Feb132021_22_16_45"
#fname="Feb132021_23_09_09"
fname="Feb132021_21_57_27"
fname="Feb132021_23_04_43"
fname= "Jan092020_15_05_39"   #  35 seconds    #  15:04:32

#fname= "TTtest"   #  35 seconds    #  15:04:32  

armv_ver = 1
artRmvDir = "../DSi_dat/%(fn)s_artfctrmvd" % {"fn" : fname, "av" : armv_ver}
if not os.access(artRmvDir, os.F_OK):
    os.mkdir(artRmvDir)
artRmvDir = "%(dir)s/v%(av)d" % {"dir" : artRmvDir, "av" : armv_ver}
if not os.access(artRmvDir, os.F_OK):
    os.mkdir(artRmvDir)

#######################################################################
#######################################################################
#edf  = mne.io.read_raw_edf("../DSi_dat/%s.edf" % fname)
#dAll = edf.get_data().T*10000
dAll  = _N.genfromtxt("../DSi_dat/%s.csv" % fname, delimiter=',')  # use 
#dAll  = _N.loadtxt("Q20_dat/%s.csv" % fname)
#dAll /= 100000

Fs  = 300

#####   All other days except Jul012020_12_00_00
chs = _N.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 21, 22, 23]) + 1   # .csv 
#chs = _N.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 21, 22, 23]) # edf
#chs = _N.arange(21)

d   = _N.array(dAll[:, chs])
#d[:, 9] = _N.random.randn(dAll.shape[0])

ch_names = ["P3", "C3", "F3", "Fz","F4", "C4", "P4", "Cz", "Pz", "M1", "Fp1", "Fp2", "T3", "T5", "O1", "O2", "F7", "F8", "M2", "T6", "T4"]
bad_chs21 = []

bad_chs_inds = None
orig_dat = None
if bad_chs21 is not None:
    orig_dat = _N.empty((d.shape[0], len(bad_chs21)))
    bad_chs_inds = _N.empty(len(bad_chs21), dtype=_N.int)

    ib = -1
    for bc in bad_chs21:
        ib += 1
        bad_chs_inds[ib] = ch_names.index(bc)
        orig_dat[:, ib] = d[:, bad_chs_inds[ib]]
        d[:, bad_chs_inds[ib]] = _N.random.randn(d.shape[0])

info  = mne.create_info(ch_names, Fs, ch_types="eeg")#, montage="standard_1020")
raw   = mne.io.RawArray(d.T, info)

raw.set_montage("standard_1020")

filt_low = 0.5
filt_high= 50
raw_tmp = raw.copy()
raw_tmp.filter(filt_low, filt_high)

#ica = mne.preprocessing.ICA(method="extended-infomax", random_state=1)
ica = mne.preprocessing.ICA(method="fastica", random_state=1)
ica.fit(raw_tmp)

#pyfig = ica.plot_components(inst=raw_tmp)
#pyfig[0].savefig("%(ard)s/%(fname)s_skull" % {"ard" : artRmvDir, "fname" : fname})
#_plt.close()

pyfig = ica.plot_sources(inst=raw_tmp, start=0, stop=d.shape[0]/Fs)
pyfig.savefig("%(ard)s/%(fname)s_All" % {"ard" : artRmvDir, "fname" : fname})

"""
#  plot 20 seconds every 100 seconds of data
#t0    = Fs*0
#t1    = Fs*20
t0    = 0
t1    = 20



while t1*Fs < d.shape[0]:
    pyfig = ica.plot_sources(inst=raw_tmp, start=t0, stop=t1)
    pyfig.savefig("%(ard)s/%(fname)s_%(t0)d_%(t1)d" % {"ard" : artRmvDir, "fname" : fname, "t0" : t0, "t1" : t1})
    _plt.close(pyfig)
    # t0z = t0+1
    # t1z = t0+2
    # pyfig = ica.plot_sources(inst=raw_tmp, start=t0z, stop=t1z)
    # pyfig.savefig("%(ard)s/%(fname)s_%(t0)d_%(t1)d" % {"ard" : artRmvDir, "fname" : fname, "t0" : t0z, "t1" : t1z})
    # _plt.close(pyfig)
    t0z = t0+11
    t1z = t0+12
    pyfig = ica.plot_sources(inst=raw_tmp, start=t0z, stop=t1z)
    pyfig.savefig("%(ard)s/%(fname)s_%(t0)d_%(t1)d" % {"ard" : artRmvDir, "fname" : fname, "t0" : t0z, "t1" : t1z})
    _plt.close(pyfig)    
    t0 += 100#*Fs
    t1 += 100#*Fs
"""
print("pick components to exclude and do something like \"exc_save([2, 4, 5])\"")

#exc_save([])
