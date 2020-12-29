import mne
import numpy as _N

RPS_resultDir     = "/Users/arai/nctc/Workspace/AIiRPS_Results"
STROOP_resultDir     = "/Users/arai/nctc/Workspace/Stroop_Results"
SIM_resultDir     = "/Users/arai/nctc/Workspace/gcohsim_results"
dataDirBase   = "/Users/arai/nctc/Workspace"

_RPS        = 1
_STROOP     = 2
_GONOGO     = 3
_AAFFECT    = 4
_SIM        = 5

def getConfig(dataset, sim_nchs=None):
    """
    if we want to run this on simulated data, sim_nchs sets # of EEG channels
    """
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
        #datdir = "%s/DSi_dat"
    elif dataset == _AAFFECT:
        ch_w_CM  =   _N.arange(19)
        rm_chs   =  []
        ch_names = ["Fp1", "Fp2", "F7", "F3","Fz", "F4", 
                    "F8", "T3", "C3", "Cz", "C4", "T4", 
                    "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
        ch_types=["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
                  "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
                  "eeg", "eeg", "eeg"]
        #datdir = "aaffect_dat"
    elif dataset == _GONOGO:
        ch_w_CM  =   _N.arange(31)
        rm_chs   =  []
        ch_names = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "PZ", "T5'", "T6'", "O1'", "O2'", "P3\"", "P4\"", "PZ\"", "OZ", "I", "CB1\"", "CB2\"", "CB1", "CB2"]
        ch_types=["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
                  "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", 
                  "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg",
                  "eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg"]
        mntgfile = getDataFN(dataset, "channel_loc.loc")
        mntg = mne.channels.read_custom_montage(mntgfile)
    elif dataset == _SIM:
        ch_w_CM  =   _N.arange(sim_nchs)
        rm_chs   =  []
        ch_names = []
        for nc in range(sim_nchs):
            ch_names.append("ch%d" % (nc+1))
        ch_types=["eeg"] * sim_nchs

    return ch_w_CM, rm_chs, ch_names, ch_types

def set_montage(dataset, info):
    if dataset == _RPS or dataset == _STROOP:
        info.set_montage("standard_1020")
    elif dataset == _GONOGO:
        info.set_montage(mntg)
    elif dataset == _AAFFECT:
        info.set_montage("standard_1020")

def getResultFN(dataset, fn):
    global RPS_resultDir, SIM_resultDir
    if dataset==_RPS:
        return "%(bd)s/%(fn)s" % {"bd" : RPS_resultDir, "fn" : fn}
    elif dataset==_STROOP:
        return "%(bd)s/%(fn)s" % {"bd" : STROOP_resultDir, "fn" : fn}
    elif dataset==_SIM:
        return "%(bd)s/%(fn)s" % {"bd" : SIM_resultDir, "fn" : fn}

def getDataFN(dataset, fn):
    global dataDirBase
    if dataset==_RPS:
        return "%(bd)s/DSi_dat/%(fn)s" % {"bd" : dataDirBase, "fn" : fn}
    elif dataset==_STROOP:
        return "%(bd)s/DSi_dat/%(fn)s" % {"bd" : dataDirBase, "fn" : fn}
    elif dataset==_GONOGO:
        return "%(bd)s/gonogo_dat/%(fn)s" % {"bd" : dataDirBase, "fn" : fn}
    elif dataset==_SIM:
        return "%(bd)s/gcohsim_dat/%(fn)s" % {"bd" : dataDirBase, "fn" : fn}

