'''
    DCT

'''

import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *

def ax(x,N):
    if x == 0:
        return np.sqrt(1.0/N)
    else:
        return np.sqrt(2.0/N)

def get_dctsum(image,u,v):
    h,w = image.shape
    assert (h == w)
    N = h
    sum = 0.0
    for i in range(h):
        for j in range(w):
            sum += image[i,j] * np.cos((2*i+1)*u*np.pi/(2*N)) * np.cos((2*j+1)*v*np.pi/(2*N))
    return sum

def get_idctsum(cdata,y,x):
    h,w = cdata.shape
    assert (h == w)
    N = h
    sum = 0.0
    for u in range(h):
        for v in range(w):
            sum += ax(u,N)*ax(v,N)*cdata[u,v] * np.cos((2*y+1)*u*np.pi/(2*N))* np.cos((2*x+1)*v*np.pi/(2*N))
    return sum


def IDCT(cdata):
    h,w = cdata.shape
    assert  (h == w)
    N = h
    data = np.zeros(cdata.shape)
    for i in range(h):
        for j in range(w):
            data[i,j] = ax(i,N)*ax(j,N)*get_idctsum(data,i,j)

    return data

def DCT_test():
    data = get_image_data("../pic/lena.jpg")
    h,w = data.shape

    plt.subplot(1, 3, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    assert (h == w)
    N = h
    cdata = np.zeros(data.shape)
    for i in range(h):
        for j in range(w):
            cdata[i,j] = ax(i,N)*ax(j,N)*get_dctsum(data,i,j)

    plt.subplot(1, 3, 2)
    plt.title('dct')
    plt.imshow(cdata, cmap=plt.get_cmap('gray'))

    idata = IDCT(cdata)
    plt.subplot(1, 3, 3)
    plt.title('idct')
    plt.imshow(idata, cmap=plt.get_cmap('gray'))

    plt.show()

def Wavelet_test():
    pass


if __name__ == '__main__':
    DCT_test()