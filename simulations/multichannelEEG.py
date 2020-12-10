import numpy as _N
import numpy.polynomial.polynomial as _Npp

import utilities as _U
import numpy as _N
import scipy.stats as _ss


dt = 0.001
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

def mix(EEG, rhy, pkn, grp, t0, t1):
    EEG[grp[1:], t0:t1] = EEG[grp[0], t0:t1] 
    

#  grpA and grpB   can coexist
#  grpA and grpC   can coexist
#  grpB and grpD   can coexist

grpA = _N.array([1, 2, 5])
grpB = _N.array([3, 4, 8])
grpC = _N.array([7, 8, 9])
grpD = _N.array([1, 6, 7])

dt   = 0.005
nChs = 10
N    = 10000   #(50 s)
f    = 20
amp  = 0.995

#  up to 4 groups of
rhythms = _N.zeros((nChs, N))
pknzs   = _N.zeros((nChs, N))
tau     = 0.15

for nc in range(nChs):
    rhythms[nc] = AR2(f, 0.98, N, dt)
    pknzs[nc] = AR1(tau, N, dt)
EEG     = _N.array(rhythms)

mix(EEG, rhythms, pknzs, grpA, 10, 15)
mix(EEG, rhythms, pknzs, grpB, 18, 25)
mix(EEG, rhythms, pknzs, grpC, 28, 34)
mix(EEG, rhythms, pknzs, grpD, 40, 45)
