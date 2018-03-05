from scipy import signal
import numpy as np
import nolds

fs = 256
seglength = 512 ##2s
def psd(data, fs, seglength):
    f, pxx_den = signal.welch(data, fs, nperseg=seglength)
    return f, pxx_den

def alpha_power(f, pxx_den):
    p = 0
    for i in range(18, 28):
        p += pxx_den[i]
    p = p/10
    return p

def delta_power(f, pxx_den):
    p = 0
    for i in range(0, 8):
        p += pxx_den[i]
    p = p/8
    return p

def theta_power(f, pxx_den):
    p = 0
    for i in range(8, 18):
        p += pxx_den[i]
    p = p/10
    return p

def rela_alpha(f, pxx_den):
    """
    alpha relative power
    """
    p1 = alpha_power(f, pxx_den)
    p2 = delta_power(f, pxx_den)
    p3 = theta_power(f, pxx_den)
    p = p1+p2+p3
    rela = p1/p
    return rela

def rela_delta(f, pxx_den):
    """
    alpha relative power
    """
    p1 = alpha_power(f, pxx_den)
    p2 = delta_power(f, pxx_den)
    p3 = theta_power(f, pxx_den)
    p = p1+p2+p3
    rela = p2/p
    return rela

def rela_theta(f, pxx_den):
    """
    alpha relative power
    """
    p1 = alpha_power(f, pxx_den)
    p2 = delta_power(f, pxx_den)
    p3 = theta_power(f, pxx_den)
    p = p1+p2+p3
    rela = p3/p
    return rela

def hemi_ratio(pl, pr):
    ratio = np.abs(pr-pl)/(pr+pl)
    return ratio

def feature_extraction(data):
    """
    Input:
        - data: 4 * 15360
    Output:
        - feature array: 18*1
    """
    fq1, pxx1 = psd(data[0,:], fs, seglength)
    fq2, pxx2 = psd(data[1,:], fs, seglength)
    fq3, pxx3 = psd(data[2,:], fs, seglength)
    fq4, pxx4 = psd(data[3,:], fs, seglength)
    f1 = rela_alpha(fq2, pxx1) # feature 1: relative alpha power of channel 1
    f2 = rela_delta(fq1, pxx1) # feature 2: relative delta power of channel 1
    f3 = rela_theta(fq1, pxx1) # feature 3: relative theta power of channel 1
    f4 = rela_alpha(fq2, pxx2) # feature 4: relative alpha power of channel 2
    f5 = rela_delta(fq2, pxx2) # feature 5: relative delta power of channel 2
    f6 = rela_theta(fq2, pxx2) # feature 6: relative theta power of channel 2
    f7 = rela_alpha(fq3, pxx3) # feature 7: relative alpha power of channel 3
    f8 = rela_delta(fq3, pxx3) # feature 8: relative delta power of channel 3
    f9 = rela_theta(fq3, pxx3) # feature 9: relative theta power of channel 3
    f10 = rela_alpha(fq4, pxx4) # feature 10: relative alpha power of channel 4
    f11 = rela_delta(fq4, pxx4) # feature 11: relative delta power of channel 4
    f12 = rela_theta(fq4, pxx4) # feature 12: relative theta power of channel 4
    f13 = hemi_ratio(f1, f4) # feature 13: hemi-ratio of alpha band between front channels (C1, C2)
    f14 = hemi_ratio(f2, f5)
    f15 = hemi_ratio(f3, f6)
    f16 = hemi_ratio(f7, f10)
    f17 = hemi_ratio(f8, f11)
    f18 = hemi_ratio(f9, f12)
    """
    multi-domain
    """
    f19 = nolds.dfa(data[0,:])
    f20 = nolds.dfa(data[1,:])
    f21 = nolds.dfa(data[2,:])
    f22 = nolds.dfa(data[3,:])
    #f = np.array([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18])
    f = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22])
    return f