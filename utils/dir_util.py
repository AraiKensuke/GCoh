#import GCoh.datconfig as datconfig

resultDir     = "/Users/arai/nctc/Workspace/AIiRPS_Results"
dataDirBase   = "/Users/arai/nctc/Workspace/"

def getResultFN(fn):
    global resultDir
    return "%(rd)s/%(fn)s" % {"rd" : resultDir, "fn" : fn}

def getDataFN(dataset, fn):
    global dataDirBase
    if dataset==datconfig._RPS:
        return "%(bd)s/DSi_dat/%(fn)s" % {"bd" : dataDirBase, "fn" : fn}
    elif dataset==datconfig._STROOP:
        return "%(bd)s/DSi_dat/%(fn)s" % {"bd" : dataDirBase, "fn" : fn}
    elif dataset==datconfig._GONOGO:
        return "%(bd)s/gonogo_dat/%(fn)s" % {"bd" : dataDirBase, "fn" : fn}

