import os
import numpy as _N
import matplotlib.pyplot as _plt
from scipy.signal import hilbert
import read_taisen as _rt
import scipy.stats as _ss
from mne.preprocessing.ica import corrmap  # noqa
from mne.preprocessing import ICA
import scipy.io as _scio

import mne   #  mne EEG data
from sklearn.decomposition import PCA
from pyedflib import highlevel


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

    _N.savetxt("%(dir)s/%(fn)s_artfctrmvd_v%(v)d.dat" % {"dir" : artRmvDir, "fn" : fname, "v" : armv_ver}, out_dat.T, fmt=("%.3f " * 19))
    fp = open("%(dir)s/params.txt" % {"dir" : artRmvDir, "v" : armv_ver}, "w")
    fp.write("exclude: %s\n" % str(ica.exclude))
    fp.write("low  Hz: %s\n" % str(filt_low))
    fp.write("high Hz: %s\n" % str(filt_high))

    if bad_chs_inds is not None:
        _N.savetxt("%(dir)s/bad_chs.dat" % {"dir" : artRmvDir}, bad_chs_inds, fmt="%d")
        
    fp.close()

fname= "aaffect_066"

#fname= "TTtest"   #  35 seconds    #  15:04:32  

armv_ver = 1
artRmvDir = "aaffect_dat/%(fn)s_artfctrmvd" % {"fn" : fname, "av" : armv_ver}
if not os.access(artRmvDir, os.F_OK):
    os.mkdir(artRmvDir)
artRmvDir = "%(dir)s/v%(av)d" % {"dir" : artRmvDir, "av" : armv_ver}
if not os.access(artRmvDir, os.F_OK):
    os.mkdir(artRmvDir)

#######################################################################
#######################################################################

#signals, signal_headers, header = highlevel.read_edf("aaffect_dat/%(fn)s/%(fn)s_eeg.edf" % {"fn" : fname})


lm = _scio.loadmat("aaffect_dat/%(fn)s/%(fn)s.mat" % {"fn" : fname})

# dAll  = _N.genfromtxt("DSi_dat/%s.csv" % fname, delimiter=',')

Fs  = 500

# #####   All other days except Jul012020_12_00_00
# chs = _N.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 21, 22, 23]) + 1
chs = _N.arange(19)

d   = _N.array(lm["rec500"].T[:, chs])

ch_names = ["Fp1", "Fp2", "F7", "F3","Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
bad_chs19 = None#["M1", "M2"]

bad_chs_inds = None
orig_dat = None
if bad_chs19 is not None:
    orig_dat = _N.empty((d.shape[0], len(bad_chs19)))
    bad_chs_inds = _N.empty(len(bad_chs19), dtype=_N.int)

    ib = -1
    for bc in bad_chs19:
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

pyfig = ica.plot_components(inst=raw_tmp)
pyfig[0].savefig("%(ard)s/%(fname)s_skull" % {"ard" : artRmvDir, "fname" : fname})
_plt.close()

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
