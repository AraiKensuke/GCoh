import numpy as _N
import scipy.signal as _ssig
import mne.time_frequency as mtf
import mne
import pickle
import os
import sys

#from sumojam.devscripts.cmdlineargs import process_keyval_args
from ka_tools.cmdlineargs import process_keyval_args
import GCoh.chronux_py.chronux_funcs as _chrf
import preprocess_ver

import GCoh.datconfig as datconf
import GCoh.windowed_gcoh as _w_gcoh
#import GCoh.utils.dir_util as _du

dataset =    datconf._RPS
c    = 2
if dataset == datconf._SIM:
    Fs   = 200
if dataset == datconf._RPS or dataset == datconf._STROOP:
    Fs   = 300
elif dataset == datconf._GONOGO:
    Fs   = 500
elif dataset == datconf._AAFFECT:
    Fs   = 500

eeg_date_times = ["Jan092020_15_05_39"]
#eeg_date_time = "Jan092020_14_00_00"
#eeg_date_time = "Jan082020_17_03_48"
# eeg_date_time = "Aug122020_13_17_39"
#eeg_date_time ="Aug122020_12_52_44"
# eeg_date_time ="Aug182020_15_45_27"
#eeg_date_time ="Aug182020_16_44_18"
#eeg_date_time="Aug182020_16_25_28"
#eeg_date_time="Dec102020_12_54_24"
# eeg_date_time="Dec102020_13_48_05"
# #eeg_date_time= "Aug122020_13_30_23"
# #eeg_date_time = "gonogo_0519"
# #eeg_date_time = "aaffect_062"
# eeg_date_time="Dec102020_17_22_55"
# eeg_date_time="Dec102020_17_27_01"
# eeg_date_time="Dec102020_18_13_33"
#eeg_date_time="Dec162020_16_10_00"

# eeg_date_times  = ["Jan012019_00_00_00", "Jan012019_01_00_00", 
#                    "Jan012019_02_00_00", "Jan012019_03_00_00", 
#                    "Jan012019_04_00_00", "Jan012019_05_00_00", 
#                    "Jan012019_06_00_00", "Jan012019_07_00_00", 
#                    "Jan012019_08_00_00", "Jan012019_09_00_00", 
#                    "Jan012019_12_00_00", "Jan012019_13_00_00",
#                    "Jan012019_14_00_00", "Jan012019_15_00_00"]

#eeg_date_times = ["Jan092020_15_05_39"]
#eeg_date_times = ["May262021_13_18_41"]
#eeg_date_times = ["May262021_14_06_30"]
#eeg_date_times = ["May262021_14_06_31"]
#eeg_date_times = ["May262021_14_37_19"]
#eeg_date_times = ["May262021_14_37_19"]
#eeg_date_times = ["May262021_14_37_20"]
eeg_date_times = ["May262021_15_05_54"]
#eeg_date_times = ["Jun092021_12_31_51"]
#eeg_date_times  = ["Jun092021_12_54_23"]
#eeg_date_times = ["Jun092021_15_22_58"]
# eeg_date_times = ["Jun092021_12_42_00"]
# eeg_date_times = ["Jun092021_13_06_34"]
# eeg_date_times = ["Jun092021_18_09_37"]
# eeg_date_times = ["Jun092021_15_35_22"]
#eeg_date_times  = ["Jun092021_17_57_45"]
#eeg_date_times  = ["Jun092021_13_27_01"]
# eeg_date_times=["Apr052021_17_50_15"]
# eeg_date_times=["Apr052021_17_52_50"]
# eeg_date_times=["Feb132021_22_16_45"]
# eeg_date_times=["Feb132021_23_09_09"]
#eeg_date_times=["Feb132021_21_57_27"]
#eeg_date_times=["Feb132021_23_04_43"]
# eeg_date_times=["Apr052021_18_01_35"]
# eeg_date_times=["Apr052021_18_12_12"]
artrmv_ver = 1
gcoh_ver    = 3

process_keyval_args(globals(), sys.argv[1:])

wnd, slideby      = preprocess_ver.get_win_slideby(gcoh_ver)

#cm    = 9
cm    = None
ch_w_CM, rm_chs, ch_names, ch_types = datconf.getConfig(dataset, sim_nchs=10, cm=cm)

for eeg_date_time in eeg_date_times:
    ####  load EEG data
    X_cm    = _N.loadtxt(datconf.getDataFN(dataset, "%(edt)s_artfctrmvd/v%(av)d/%(edt)s_artfctrmvd_v%(av)d.dat" % {"edt" : eeg_date_time, "av" : artrmv_ver}))
    X_cm 

    if os.access(datconf.getDataFN(dataset, "%(edt)s_artfctrmvd/v%(v)d/bad_chs.dat" % {"edt" : eeg_date_time, "v" : artrmv_ver}), os.F_OK):
        ####  find channels identified as bad
        bad_chs = _N.loadtxt(datconf.getDataFN(dataset, "%(edt)s_artfctrmvd/v%(v)d/bad_chs.dat" % {"edt" : eeg_date_time, "v" : artrmv_ver}), dtype=_N.int, ndmin=1)
        rm_chs.extend(bad_chs)
    ch_picks = _N.setdiff1d(ch_w_CM, _N.array(rm_chs))
    arr_ch_picks = _N.array(ch_picks)

    info = mne.create_info(ch_names=(_N.array(ch_names)[ch_picks]).tolist(), ch_types=(_N.array(ch_types)[ch_picks]).tolist(), sfreq=Fs)

    datconf.set_montage(dataset, info)

    dpss_bw=7
    f, findx, Ctot, Cvec = _w_gcoh.windowed_gcoh(Fs, wnd, slideby, X_cm, ch_w_CM, ch_picks, info, dpss_bw=dpss_bw)

    pkldat = {"VEC" : Cvec, "Cs" : Ctot, "fs" : f[findx], "chs_picks" : arr_ch_picks, "dpss_bw" : dpss_bw}

    dmp = open(datconf.getDataFN(dataset, "%(edt)s_artfctrmvd/v%(av)d/%(edt)s_gcoh_%(w)d_%(wm)d_v%(av)d%(gv)d.dmp" % {"edt" : eeg_date_time, "w" : wnd, "wm" : slideby, "av" : artrmv_ver, "gv" : gcoh_ver}), "wb")
    pickle.dump(pkldat, dmp, -1)
    dmp.close()

