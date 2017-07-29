'''
    frequence spacial
'''

import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *


def get_complex_distance(comp1,comp2):
    return np.sqrt((comp1.real-comp2.real)**2 + (comp1.imag-comp2.imag)**2)


def fft_test():
    data = get_image_data("../pic/lena.jpg")

    plt.subplot(4,2,1)
    plt.title('origin')
    plt.imshow(data,cmap = plt.get_cmap('gray'))

    fft = np.fft.fft2(data)
    # plt.subplot(4,2,2)
    # amp_spectrum = np.abs(fft)
    # plt.title('spectrum')
    # plt.imshow(np.log(amp_spectrum),cmap = plt.get_cmap('gray'))

    plt.subplot(4, 2, 2)
    fft_shift = np.fft.fftshift(fft)
    plt.title('fftshift')
    plt.imshow(np.log(np.abs(fft_shift)),cmap = plt.get_cmap('gray'))


    fft_shift_high = np.copy(fft_shift)
    h,w = fft_shift_high.shape
    for i in range(w):
        for j in range(h):
            d = np.sqrt((i-w/2)**2 + (j-h/2)**2)
            if  d <= 5:
                fft_shift_high[i,j] = 0j

    plt.subplot(4, 2, 3)
    plt.title('high pass')
    plt.imshow(np.abs(fft_shift_high), cmap=plt.get_cmap('gray'))

    plt.subplot(4, 2, 4)
    ifft_high = np.fft.ifft2(fft_shift_high)
    plt.title('ifft_high')
    plt.imshow(np.abs(ifft_high), cmap=plt.get_cmap('gray'))

    fft_shift_low = np.copy(fft_shift)
    h,w = fft_shift_low.shape
    for i in range(w):
        for j in range(h):
            d = np.sqrt((i-w/2)**2 + (j-h/2)**2)
            if  d >= 20:
                fft_shift_low[i,j] = 0j

    plt.subplot(4, 2, 5)
    plt.title('low pass')
    plt.imshow(np.abs(fft_shift_low), cmap=plt.get_cmap('gray'))


    plt.subplot(4, 2, 6)
    ifft_low = np.fft.ifft2(fft_shift_low)
    plt.title('ifft_low')
    plt.imshow(np.abs(ifft_low), cmap=plt.get_cmap('gray'))


    fft_shift_band = np.copy(fft_shift)
    h,w = fft_shift_band.shape
    for i in range(w):
        for j in range(h):
            d = np.sqrt((i-w/2)**2 + (j-h/2)**2)
            if  d >= 20  and d <=40:
                fft_shift_band[i,j] = 0j

    plt.subplot(4, 2, 7)
    plt.title('band pass')
    plt.imshow(np.abs(fft_shift_band), cmap=plt.get_cmap('gray'))


    plt.subplot(4, 2, 8)
    ifft_band = np.fft.ifft2(fft_shift_band)
    plt.title('ifft_band')
    plt.imshow(np.abs(ifft_band), cmap=plt.get_cmap('gray'))


    plt.savefig('fourier.jpg')
    plt.show()


def batworth_low_pass_filter():
    data = get_image_data("../pic/lena.jpg")
    fft = np.fft.fft2(data)
    fft_shift = np.fft.fftshift(fft)

    d0 = 5
    h,w = fft_shift.shape
    for i in range(w):
        for j in range(h):
            d = np.sqrt((i-w/2)**2 + (j-h/2)**2)
            if  d <= d0:
                fft_shift[i,j] = 1/(1+(d/d0)**2)


    print fft_shift

    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 2, 2)
    ifft = np.fft.ifft2(fft_shift)
    plt.title('ifft_bw')
    plt.imshow(np.abs(ifft), cmap=plt.get_cmap('gray'))

    plt.show()

if __name__ == '__main__':
    #fft_test()
    batworth_low_pass_filter()