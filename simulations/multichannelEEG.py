import numpy as _N
import os

import GCoh.datconfig as datconf
import GCoh.simulations.sim_utils as _su

#  grpA and grpB   can coexist
#  grpA and grpC   can coexist
#  grpB and grpD   can coexist

grpA = _N.array([1, 2, 5])
grpB = _N.array([3, 4, 8])
grpC = _N.array([7, 8, 9])
grpD = _N.array([1, 6, 7])
grpA_phase_diff = _N.array([0, -2, 5])
grpB_phase_diff = _N.array([0, 13, 9])
grpC_phase_diff = _N.array([0, 1, 1])
grpD_phase_diff = _N.array([0, 0, 0])

dt   = 0.005
nChs = 10
N    = 10000   #(50 s)
f    = 20      #  oscillation frequency 
amp  = 0.995

#  up to 4 groups of
rhythms = _N.zeros((nChs, N))
pknzs   = _N.zeros((nChs, N))
tau     = 0.15    #  timescale of AR1

######  we generate simulated EEG by generating nChs of independent oscillatory
######  signals.  
for nc in range(nChs):
    rhythms[nc] = _su.AR2(f, 0.98, N, dt)
    pknzs[nc] = _su.AR1(tau, N, dt)
EEG     = _N.array(rhythms)

#mix(dt, EEG, rhythms, pknzs, grpA, 10, 15)
_su.mix(dt, EEG, rhythms, pknzs, grpB, grpB_phase_diff, 18, 45)
#mix(dt, EEG, rhythms, pknzs, grpD, 18, 45)
#mix(dt, EEG, rhythms, pknzs, grpC, 28, 34)
#mix(dt, EEG, rhythms, pknzs, grpD, 40, 45)

outfn = _su.generate_artfctrmvd_filename()
_N.savetxt(outfn, EEG.T, fmt=("%.4f " * nChs))
print("saved %s"  % outfn)
