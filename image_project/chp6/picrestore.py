'''
    restore
'''
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *

def PSNR_test():
    data1 = get_image_data("../pic/lena.jpg")
    data2 = get_image_data("../pic/lena.BMP")

    assert(data1.shape == data2.shape)
    h,w = data1.shape
    sum = 0.0
    for i in range(h):
        for j in range(w):
            sum += (data2[i,j]-data1[i,j])**2
    MSE = sum / (h*w)

    MAX = np.max(data1)
    PSNR = 10*np.log10(MAX**2/MSE)

    print PSNR

def TV_model_test():
    pass

def mixture_model_test():
    pass


if __name__ == '__main__':
    PSNR_test()