import numpy as _N
import pandas as _pd

#(0) Time (s)       (1) F7(uV)  (2) Fp1(uV) (3) Fp2(uV) (4) F8(uV)  (5) F3(uV)  (6) Fz(uV)  (7) F4(uV)  (8) C3(uV)  (9) Cz(uV)  (10) P8(uV)  (11) P7(uV)  (12) Pz(uV)  (13) P4(uV)  (14) T3(uV)  (15) P3(uV)  (16) O1(uV)  (17) O2(uV)  (18) C4(uV)  (19) T4(uV)  (20) A2(uV)  (21) ExG 1 == A1(uV)       ACC21(mg)       ACC22(mg)       ACC23(mg)       Packet Counter(DIGITAL) TRIGGER(DIGITAL)        F7Impedance (kOhms)     Fp1Impedance (kOhms)    Fp2Impedance (kOhms)    F8Impedance (kOhms)     F3Impedance (kOhms)     FzImpedance (kOhms)     F4Impedance (kOhms)     C3Impedance (kOhms)     CzImpedance (kOhms)     P8Impedance (kOhms)     P7Impedance (kOhms)     PzImpedance (kOhms)     P4Impedance (kOhms)     T3Impedance (kOhms)     P3Impedance (kOhms)     O1Impedance (kOhms)     O2Impedance (kOhms)     C4Impedance (kOhms)     T4Impedance (kOhms)     A2Impedance (kOhms)     ExG 1Impedance (kOhms)

# Q20
#(1)  F7  (2) Fp1  (3) Fp2  (4) F8   (5) F3
#(6)  Fz  (7) F4   (8) C3   (9) Cz   (10) P8
#(11) P7  (12) Pz  (13) P4  (14) T3  (15) P3
#(16) O1  (17) O2  (18) C4  (19) T4  (20) A2
#(21) ExG 1 == A1

# DSi
#ch_names = ["P3", "C3", "F3", "Fz","F4",
#            "C4", "P4", "Cz", "Pz", "M1",
#            "Fp1", "Fp2", "T3", "T5", "O1",
#            "O2", "F7", "F8", "M2", "T6",
#            "T4"]
chs = _N.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 21, 22, 23]) + 1

# ch_names = ["xxP3", "xxC3", "xxF3", "xxFz","xxF4", "xxC4", "xxP4", "xxCz", "xxPz", "M1", "xxFp1", "xxFp2", "xxT3", "T5 (P7)", "xxO1", "xxO2", "xxF7", "xxF8", "M2", "T6 (P8)", "xxT4, "A2 = M2", A1 = M1"]
#  p8 p7

#  M1 is A1, M2 is A2, T5 is P7, T6 is P8

#q20_dat =  _pd.read_csv("Q20_dat/James2.txt", thousands=",", sep="\t", comment="#").to_numpy()
#q20_dat =  _pd.read_csv("Q20_dat/Sara_RPS_shrt.txt", thousands=",", sep="\t", comment="#").to_numpy()
#q20_dat =  _pd.read_csv("Q20_dat/Mavi_RPS_shrt.txt", thousands=",", sep="\t", comment="#").to_numpy()
q20_dat =  _pd.read_csv("../Q20_dat/Ali_RPS.txt", thousands=",", sep="\t", comment="#").to_numpy()


#for ch in range(1, 22):
#    q20_dat[:, ch] -= q20_dat[:, 21]
q20_dat[:, 20] = _N.random.randn(q20_dat.shape[0])
q20_dat[:, 21] = _N.random.randn(q20_dat.shape[0])
EEG_chans = _N.array([15,  8,  5,  6,  7,
                      18, 13,  9, 12, 21,
                      2,  3, 14, 11, 16,
                      17,  1,  4, 20, 10,
                      19])


dat500 = q20_dat[:, EEG_chans]

ts500 = _N.arange(q20_dat.shape[0])*0.002
N300 = int(ts500[-1] / (1/300))
ts300 = _N.arange(0, N300) * (1/300)

dat300 = _N.zeros((21, ts300.shape[0]))
dat_25_300    = _N.zeros((25, ts300.shape[0]))


for ch in range(21):
    dat300[ch] = _N.interp(ts300, ts500, dat500[:, ch])

dat_25_300[chs] = dat300
#"May142020_23_31_04"
#2021-05-26 13:18:41.125
#2021-05-26 14:37:19
#_N.savetxt("Q20_dat/May262021_13_18_41.csv", dat_25_300.T, fmt=("%.4f "*25))
#_N.savetxt("Q20_dat/May262021_14_37_19.csv", dat_25_300.T, fmt=("%.4f "*25))
_N.savetxt("../DSi_dat/May262021_15_05_54.csv", dat_25_300.T, fmt=("%.4f "*25))
#_N.savetxt("Q20_dat/May262021_14_06_30.csv", dat_25_300.T, fmt=("%.4f "*25))


