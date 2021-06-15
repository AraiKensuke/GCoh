import numpy as _N
import matplotlib.pyplot as _plt
import mne

tcksz  = 16
lblsz   = 18

#date = "Jan092020_15_05_39"
#date = "Jan102021_02_00_00"
#date = "Jan122021_02_00_00"
#date = "Jan202021_01_00_00"
#date = "Aug182020_16_44_18"
#date = "Aug182020_15_45_27"
#date = "Aug122020_12_52_44"
#date="Jan082020_17_03_48"
date = "Jan132021_04_00_00"
date = "Jan182021_01_00_00"
date = "Apr052021_06_00_00"
#date = "Mar162021_02_00_00"
#date = "Mar162021_02_00_00"
#date = "Mar262021_00_00_00"
##date = "Jan012021_06_00_00"
#date = "Jan022021_09_00_00"
#date = "Jan012021_10_00_00"
#date = "Jan012021_13_00_00"
#date = "Jan012021_12_00_00"
date="May262021_14_37_19"
date="Jun092021_17_57_45"
date="Jun092021_12_54_23"
date="Jun092021_13_27_01"
date="Jun092021_15_22_58"
date="May262021_15_05_54"
av   = 1

raw_cvs = True
edf_file     = False
show_all = True

zoom   = 1
Fs     = 300
#dT     = 10000

if date == "Mar162021_07_00_00":
    snap_ts = [[0, 100*Fs], [160*Fs, 220*Fs], [270*Fs, 410*Fs], [500*Fs, 600*Fs], [750*Fs, 820*Fs]]
elif date == "Mar162021_01_00_00":
    snap_ts = [[0, 100*Fs], [150*Fs, 250*Fs], [300*Fs, 700*Fs]]
elif date == "Mar162021_00_00_00":
    snap_ts = [[0, 100*Fs], [10*Fs, 250*Fs], [250*Fs, 750*Fs]]
elif date == "Mar162021_02_00_00":
    #snap_ts = [[50, 150*Fs], [250*Fs, 400*Fs], [460*Fs, 520*Fs], [740*Fs, 810*Fs], [810*Fs, 840*Fs]]
    snap_ts = [[700*Fs, 730*Fs]]
elif date == "Mar162021_03_00_00":
    snap_ts = [[360, 500*Fs], [540*Fs, 740*Fs]]
elif date == "Mar162021_04_00_00":
    snap_ts = [[420*Fs, 510*Fs], [600*Fs, 800*Fs], [920*Fs, 980*Fs], [1000*Fs, 1070*Fs]]
elif date == "Mar262021_00_00_00":
    snap_ts = [[0, 100*Fs], [100*Fs, 150*Fs], [200*Fs, 250*Fs], [300*Fs, 350*Fs], [500*Fs, 550*Fs], [700*Fs, 750*Fs]]

ch_names = ["P3", "C3", "F3", "Fz","F4", "C4", "P4", "Cz", "Pz", "M1", "Fp1", "Fp2", "T3", "T5", "O1", "O2", "F7", "F8", "M2", "T6", "T4"]

if raw_cvs:
    chs = _N.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 21, 22, 23]) 
    if edf_file:
        edf_dat  = mne.io.read_raw_edf("../DSi_dat/%s.edf" % date)
        _dat = edf_dat.get_data().T
    else:
        chs += 1
        _dat = _N.genfromtxt("../DSi_dat/%(date)s.csv" % {"date" : date}, delimiter=" ")
    d = _N.array((_dat[:, chs]))
    info  = mne.create_info(ch_names, Fs, ch_types="eeg")#, montage="standard_1020")
    raw   = mne.io.RawArray(d.T, info)

    raw.set_montage("standard_1020")
    filt_low = 0.5
    filt_high= 50
    datT = raw.filter(filt_low, filt_high, method="iir").get_data()
    dat  = datT.T
else:
    dat = _N.loadtxt("../DSi_dat/%(date)s_artfctrmvd/v%(av)d/%(date)s_artfctrmvd_v%(av)d.dat" % {"date" : date, "av" : av})

N      = dat.shape[0]

if show_all:
    dT     = N
    snap_ts = [[0, N]]


it = 0

clr    = ["#000000", "#666666"]
#for t1 in _N.arange(100000, 300000, 30000):
#for t1 in _N.arange(150000, 220000, 30000):
#for t1,t2 in : # 010000
for t1,t2 in snap_ts:

    it += 1
    dT = t2-t1
    srtdat = _N.sort(dat[t1:t2], axis=0)
    lo5s    = srtdat[int(0.05*dT)]
    hi5s    = srtdat[int(0.95*dT)]
    md5s    = srtdat[int(0.5*dT)]
    amps    = hi5s - lo5s

    ts = _N.linspace(t1/Fs, t2/Fs, dT, endpoint=False)
        
    fig = _plt.figure(figsize=(15, 10))
    if show_all:
        _plt.suptitle("%(dat)s pg all  zoom %(zm)d   %(pp)s" % {"dat" : date, "pg" : it, "zm" : zoom, "pp" : ("LP only" if raw_cvs else "LP+artifact rmv")}, fontsize=(lblsz+2))
    else:
        _plt.suptitle("%(dat)s pg%(pg)d  zoom %(zm)d   %(pp)s   pg%(pg)d" % {"dat" : date, "pg" : it, "zm" : zoom, "pp" : ("LP only" if raw_cvs else "LP+artifact rmv")}, fontsize=(lblsz+2))

    for i in range(21):
        _plt.plot(ts, (zoom*(dat[t1:t2, i] - md5s[i])) / amps[i] + 3*i, color=clr[i%2])
    _plt.xlim(t1/Fs, t2/Fs)
    _plt.ylim(-2, 20*3+2)
    _plt.yticks(_N.arange(21)*3, ch_names, fontsize=tcksz)
    _plt.xticks(fontsize=tcksz)
    _plt.xlabel("seconds", fontsize=lblsz)

    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.94)
    if show_all:
        _plt.savefig("EEG%(cvs)s%(dat)s_all" % {"dat" : date, "pg" : it, "cvs" : ("_cvs_" if raw_cvs else "")})
    else:
        _plt.savefig("EEG%(cvs)s%(dat)s_pg%(pg)d" % {"dat" : date, "pg" : it, "cvs" : ("_cvs_" if raw_cvs else "")})

