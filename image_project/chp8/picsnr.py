'''
    compress
'''

import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *

MS = 1
RMS = 2
DB = 3


def  RMS(src, new):
    assert (src.shape == new.shape)
    h,w = src.shape

    sum = 0.0
    for i in range(h):
        for j in range(w):
            sum += (src[i,j] - new[i,j])**2

    return np.sqrt(sum) / (h*w)


def SNR(src,new,type=DB):
    assert (src.shape == new.shape)
    h,w = src.shape

    sum1 = 0.0
    sum2 = 0.0
    sum3 = 0.0
    M = np.mean(src)
    for i in range(h):
        for j in range(w):
            sum1 += new[i,j]**2
            sum2 += (new[i,j]-src[i,j])**2
            sum3 += (src[i,j]-M)**2

    if  type == MS:
        return sum1 / sum2
    elif type == RMS:
        return np.sqrt(sum1/sum2)
    elif type == DB:
        return 10 * np.log10(sum3/sum2)
    else: #PSNR
        MAX = np.max(src)
        return 10*np.log10(MAX**2/sum2)