import os
import numpy as _N
import matplotlib.pyplot as _plt
from scipy.signal import hilbert
import read_taisen as _rt
import scipy.stats as _ss
from mne.preprocessing.ica import corrmap  # noqa
from mne.preprocessing import ICA

import mne   #  mne EEG data

def exc_save(exc_list, put_back=False):
    global ica, raw_tmp, artRmvDir, armv_ver, filt_low, filt_high, bad_chs, orig_dat, bad_chs_31, Fs, filt_low, filt_high

    ica.exclude = exc_list
    raw_corrected = raw_tmp.copy()
    ica.apply(raw_corrected)

    out_dat  = raw_corrected.get_data()
    if put_back:
        info_orig  = mne.create_info(bad_chs31, Fs, ch_types="eeg")#, montage="standard_1020")
        raw_orig   = mne.io.RawArray(orig_dat.T, info_orig)
        raw_orig.filter(filt_low, filt_high)

        raw_orig_dat = raw_orig.get_data()
        ich = -1
        for ch in exc_list:
            ich += 1
            out_dat[ch] = raw_orig_dat[ich]

    _N.savetxt("%(dir)s/%(fn)s_artfctrmvd_v%(v)d.dat" % {"dir" : artRmvDir, "fn" : fname, "v" : armv_ver}, out_dat.T, fmt=("%.3e " * 31))
    fp = open("%(dir)s/params.txt" % {"dir" : artRmvDir, "v" : armv_ver}, "w")
    fp.write("exclude: %s\n" % str(ica.exclude))
    fp.write("low  Hz: %s\n" % str(filt_low))
    fp.write("high Hz: %s\n" % str(filt_high))

    if bad_chs_inds is not None:
        _N.savetxt("%(dir)s/bad_chs.dat" % {"dir" : artRmvDir}, bad_chs_inds, fmt="%d")
        
    fp.close()

fname= "gonogo_0519"

#fname= "TTtest"   #  35 seconds    #  15:04:32  

armv_ver = 1
artRmvDir = "gonogo_dat/%(fn)s_artfctrmvd" % {"fn" : fname, "av" : armv_ver}
if not os.access(artRmvDir, os.F_OK):
    os.mkdir(artRmvDir)
artRmvDir = "%(dir)s/v%(av)d" % {"dir" : artRmvDir, "av" : armv_ver}
if not os.access(artRmvDir, os.F_OK):
    os.mkdir(artRmvDir)

raw = mne.io.read_raw_eeglab("gonogo_dat/%(fn)s/%(fn)s_eeg.set" % {"fn" : fname}, preload=True)
mntg = mne.channels.read_custom_montage("gonogo_dat/channel_loc.loc")

# #######################################################################
# #######################################################################
# dAll  = _N.genfromtxt("gonogo_dat/%s.csv" % fname, delimiter=',')

Fs_orig  = 1000
Fs_ds    = 500

# #####   All other days except Jul012020_12_00_00
chs = _N.arange(31)

ch_names = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "PZ", "T5'", "T6'", "O1'", "O2'", "P3\"", "P4\"", "PZ\"", "OZ\"", "I", "CB1\"", "CB2\"", "CB1", "CB2"]
bad_chs_31 = None
bad_chs_inds = None
#orig_dat = None
# if bad_chs_31 is not None:
#     orig_dat = _N.empty((d.shape[0] // 2, len(bad_chs_31)))
#     bad_chs_inds = _N.empty(len(bad_chs_31), dtype=_N.int)

#     ib = -1
#     for bc in bad_chs_31:
#         ib += 1
#         bad_chs_inds[ib] = ch_names.index(bc)
#         orig_dat[:, ib] = d[::2, bad_chs_inds[ib]]
#         d[:, bad_chs_inds[ib]] = _N.random.randn(d.shape[0]//2)

#info  = mne.create_info(ch_names, Fs, ch_types="eeg", montage=mntg)
raw.set_montage(mntg)

filt_low = 0.5
filt_high= 50
raw_tmp = raw.copy().resample(Fs_ds)
raw_tmp.filter(filt_low, filt_high)
d = raw_tmp.get_data()
#ica = mne.preprocessing.ICA(method="extended-infomax", random_state=1)
ica = mne.preprocessing.ICA(method="fastica", random_state=1)
ica.fit(raw_tmp)

pyfig = ica.plot_components(inst=raw_tmp)
pyfig[0].savefig("%(ard)s/%(fname)s_skull" % {"ard" : artRmvDir, "fname" : fname})
_plt.close()

pyfig = ica.plot_sources(inst=raw_tmp, start=0, stop=d.shape[1]/Fs_ds)
pyfig.savefig("%(ard)s/%(fname)s_All" % {"ard" : artRmvDir, "fname" : fname})

"""
#  plot 20 seconds every 100 seconds of data
#t0    = Fs*0
#t1    = Fs*20
t0    = 0
t1    = 20



# while t1*Fs < d.shape[0]:
#     pyfig = ica.plot_sources(inst=raw_tmp, start=t0, stop=t1)
#     pyfig.savefig("%(ard)s/%(fname)s_%(t0)d_%(t1)d" % {"ard" : artRmvDir, "fname" : fname, "t0" : t0, "t1" : t1})
#     _plt.close(pyfig)
#     # t0z = t0+1
#     # t1z = t0+2
#     # pyfig = ica.plot_sources(inst=raw_tmp, start=t0z, stop=t1z)
#     # pyfig.savefig("%(ard)s/%(fname)s_%(t0)d_%(t1)d" % {"ard" : artRmvDir, "fname" : fname, "t0" : t0z, "t1" : t1z})
#     # _plt.close(pyfig)
#     t0z = t0+11
#     t1z = t0+12
#     pyfig = ica.plot_sources(inst=raw_tmp, start=t0z, stop=t1z)
#     pyfig.savefig("%(ard)s/%(fname)s_%(t0)d_%(t1)d" % {"ard" : artRmvDir, "fname" : fname, "t0" : t0z, "t1" : t1z})
#     _plt.close(pyfig)    
#     t0 += 100#*Fs
#     t1 += 100#*Fs
# """
# print("pick components to exclude and do something like \"exc_save([2, 4, 5])\"")

# #exc_save([])
