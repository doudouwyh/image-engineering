'''
    fft test
'''

import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *

def fft_test():
    data = get_image_data("../pic/mosquitto-16x16.png")

    plt.subplot(2,4,1)
    plt.title('origin')
    plt.imshow(data,cmap = plt.get_cmap('gray'))

    fft = np.fft.fft2(data)

    plt.subplot(2, 4, 5)
    fft_shift = np.fft.fftshift(fft)
    plt.title('fftshift')
    plt.imshow(np.log(np.abs(fft_shift)),cmap = plt.get_cmap('gray'))

    fft_shift_high = np.copy(fft_shift)
    h,w = fft_shift_high.shape
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i-w/2)**2 + (j-h/2)**2)
            if  d <= 5:
                fft_shift_high[i,j] = 1

    plt.subplot(2, 4, 2)
    plt.title('high pass')
    plt.imshow(np.log(np.abs(fft_shift_high)), cmap=plt.get_cmap('gray'))

    plt.subplot(2, 4, 6)
    ifft_high = np.fft.ifft2(fft_shift_high)
    plt.title('ifft_high')
    plt.imshow(np.abs(ifft_high), cmap=plt.get_cmap('gray'))

    fft_shift_low = np.copy(fft_shift)
    h,w = fft_shift_low.shape
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i-w/2)**2 + (j-h/2)**2)
            if  d >= 20:
                fft_shift_low[i,j] = 1

    plt.subplot(2, 4, 3)
    plt.title('low pass')
    plt.imshow(np.log(np.abs(fft_shift_low)), cmap=plt.get_cmap('gray'))


    plt.subplot(2, 4, 7)
    ifft_low = np.fft.ifft2(fft_shift_low)
    plt.title('ifft_low')
    plt.imshow(np.abs(ifft_low), cmap=plt.get_cmap('gray'))


    fft_shift_band = np.copy(fft_shift)
    h,w = fft_shift_band.shape
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i-w/2)**2 + (j-h/2)**2)
            if 20 <= d <= 30:
                fft_shift_band[i,j] = 1

    plt.subplot(2, 4, 4)
    plt.title('band pass')
    plt.imshow(np.log(np.abs(fft_shift_band)), cmap=plt.get_cmap('gray'))


    plt.subplot(2, 4, 8)
    ifft_band = np.fft.ifft2(fft_shift_band)
    plt.title('ifft_band')
    plt.imshow(np.abs(ifft_band), cmap=plt.get_cmap('gray'))

    plt.savefig('fourier.jpg')
    plt.show()


def get_cumsum(image,y,x):
    h,w = image.shape
    sum = 0.0
    for i in range(h):
        for j in range(w):
            sum += image[y,x] * np.exp(-2*np.pi*(y*i+x*j)/h *1j)
    return sum/h

def dft_test():
    data = get_image_data("../pic/mosquitto-16x16.png")

    ft = np.zeros(data.shape)
    h,w = data.shape
    for i in range(h):
        for j in range(w):
            ft[i,j] = get_cumsum(data,i,j)
            if np.abs(ft[i,j]) == 0:
                ft[i,j] = 1

    plt.subplot(1,4,1)
    plt.title('origin')
    plt.imshow(data,cmap = plt.get_cmap('gray'))

    plt.subplot(1, 4, 2)
    plt.title('ft')
    plt.imshow(np.log(np.abs(ft)),cmap = plt.get_cmap('gray'))

    plt.subplot(1, 4, 3)
    fft_shift = np.fft.fftshift(ft)
    plt.title('ftshift')
    plt.imshow(np.log(np.abs(fft_shift)),cmap = plt.get_cmap('gray'))

    plt.subplot(1, 4, 4)
    ifft = np.fft.ifft2(fft_shift)
    plt.title('ifft')
    plt.imshow(np.log(np.abs(ifft)),cmap = plt.get_cmap('gray'))

    plt.show()

if __name__ == '__main__':
    # fft_test()
    dft_test()
