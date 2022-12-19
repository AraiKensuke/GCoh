import numpy as _N
import numpy.polynomial.polynomial as _Npp
import datetime
import os
import GCoh.datconfig as datconf

def AR2(f, amp, N, dt):
    """
    f in Hz
    """
    Nyqf = 0.5/dt

    zp = amp*(_N.cos(f*_N.pi/Nyqf) + 1j*_N.sin(f*_N.pi/Nyqf))
    zn = amp*(_N.cos(f*_N.pi/Nyqf) - 1j*_N.sin(f*_N.pi/Nyqf))

    F  = -1*_Npp.polyfromroots(_N.array([zp, zn]))[::-1].real

    xc = _N.zeros(N)

    e  = 0.1
    xc[0] = e*_N.random.randn()
    xc[1] = e*_N.random.randn()

    for t in range(2, N):
        xc[t] = F[1]*xc[t-1] + F[2]*xc[t-2] + e*_N.random.randn()
    xc /= _N.std(xc)
    return xc

def AR1(tau, N, dt):
    """
    tau in seconds
    """
    k = _N.exp(-dt/tau)    # tau = -1/_N.log(k)

    xc = _N.zeros(N)

    e  = 0.1
    for t in range(1, N):
        xc[t] = k*xc[t-1] + e*_N.random.randn()
    xc /= _N.std(xc)
    return xc

def mix(dt, EEG, rhy, grp, rel_phase, t0_sec, t1_sec):
    nInGrp = len(grp)

    for nIG in range(1, nInGrp):
        EEG[grp[nIG], int(t0_sec/dt)+rel_phase[nIG]:int(t1_sec/dt)+rel_phase[nIG]] += rhy[grp[0], int(t0_sec/dt):int(t1_sec/dt)] 

def generate_artfctrmvd_filename():
    month_str = ["Jan","Feb", "Mar", "Apr", "May", "Jun",
                 "Jul","Aug", "Sep", "Oct", "Nov", "Dec"]

    ######  name the 
    now     = datetime.datetime.now()
    day     = "%02d" % now.day
    mnthStr = month_str[now.month-1]
    year    = "%d" % now.year
    hour    = "%02d" % now.hour
    minute  = "%02d" % now.minute
    second  = "%02d" % now.second
    jh_fn_mod = "%(mth)s%(dy)s%(yr)s_%(hr)s_%(min)s_%(sec)s" % {"yr" : year, "mth" : mnthStr, "dy" : day, "hr" : hour, "min" : minute, "sec" : second}

    os.mkdir(datconf.getDataFN(datconf._SIM, "%s_artfctrmvd" % jh_fn_mod))
    os.mkdir(datconf.getDataFN(datconf._SIM, "%s_artfctrmvd/v1" % jh_fn_mod))
    outfn = datconf.getDataFN(datconf._SIM, "%(j)s_artfctrmvd/v1/%(j)s_artfctrmvd_v1.dat" % {"j" : jh_fn_mod})
    
    return outfn
