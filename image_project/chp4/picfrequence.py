'''
    frequence domain
'''

import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *


def fft_test():
    data = get_image_data("../pic/lena.jpg")
    print data

    plt.subplot(2,2,1)
    plt.title('origin')
    plt.imshow(data,cmap = plt.get_cmap('gray'))

    plt.subplot(2,2,2)
    fft = np.fft.fft2(data)
    amp_spectrum = np.abs(fft)
    plt.title('spectrum')
    plt.imshow(np.log(amp_spectrum),cmap = plt.get_cmap('gray'))

    plt.subplot(2, 2, 3)
    fft_shift = np.fft.fftshift(fft)
    plt.title('fftshift')
    plt.imshow(np.log(np.abs(fft_shift)),cmap = plt.get_cmap('gray'))

    plt.subplot(2, 2, 4)
    ifft = np.fft.ifft2(fft)
    print ifft.real
    plt.title('ifft')
    plt.imshow(np.abs(ifft), cmap=plt.get_cmap('gray'))


    plt.savefig('fourier.jpg')


if __name__ == '__main__':
    fft_test()