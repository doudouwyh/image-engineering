'''
    low frequence pass filter
'''

import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *

def Butterworth_low_pass_filter_test():

    data = get_image_data("../pic/lena.jpg")
    fft = np.fft.fft2(data)
    fft_shift = np.fft.fftshift(fft)

    n = 1
    d0 = 10
    fft_shift_low = np.copy(fft_shift)
    h,w = fft_shift_low.shape
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i-w/2)**2 + (j-h/2)**2)
            hf = 1/(1+(d/d0)**(2*n))
            fft_shift_low[i,j] = hf*fft_shift_low[i,j]

    plt.subplot(1, 4, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 2)
    plt.title('fftshift')
    plt.imshow(np.log(np.abs(fft_shift)), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 3)
    plt.title('fftshiftlow')
    plt.imshow(np.log(np.abs(fft_shift_low)), cmap=plt.get_cmap('gray'))


    plt.subplot(1, 4, 4)
    ifft = np.fft.ifft2(fft_shift_low)
    plt.title('ifft_bw_low_pass')
    plt.imshow(np.abs(ifft), cmap=plt.get_cmap('gray'))

    plt.show()

def   ladder_shaped_low_pass_filter_test():
    data = get_image_data("../pic/lena.jpg")
    fft = np.fft.fft2(data)
    fft_shift = np.fft.fftshift(fft)

    d0 = 20
    df = 10
    fft_shift_low = np.copy(fft_shift)
    h,w = fft_shift_low.shape
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i-w/2)**2 + (j-h/2)**2)
            if d >= d0:
                hf = 0
            elif df < d < d0:
                hf = (d-d0)/(df-d0)
            else:
                hf = 1
            fft_shift_low[i, j] = hf * fft_shift_low[i, j]

    plt.subplot(1, 4, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 2)
    plt.title('fftshift')
    plt.imshow(np.log(np.abs(fft_shift)), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 3)
    plt.title('fftshiftlow')
    plt.imshow(np.log(np.abs(fft_shift_low)), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 4)
    ifft = np.fft.ifft2(fft_shift_low)
    plt.title('ifft_ls_low_pass')
    plt.imshow(np.abs(ifft), cmap=plt.get_cmap('gray'))

    plt.show()

def index_low_pass_filter_test():
    data = get_image_data("../pic/lena.jpg")
    fft = np.fft.fft2(data)
    fft_shift = np.fft.fftshift(fft)

    n = 1
    d0 = 30
    fft_shift_low = np.copy(fft_shift)
    h, w = fft_shift_low.shape
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i - w / 2) ** 2 + (j - h / 2) ** 2)
            hf = np.exp(- (d / d0) ** n)
            fft_shift_low[i, j] = hf * fft_shift_low[i, j]

    plt.subplot(1, 4, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 2)
    plt.title('fftshift')
    plt.imshow(np.log(np.abs(fft_shift)), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 3)
    plt.title('fftshiftlow')
    plt.imshow(np.log(np.abs(fft_shift_low)), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 4)
    ifft = np.fft.ifft2(fft_shift_low)
    plt.title('ifft_index_low_pass')
    plt.imshow(np.abs(ifft), cmap=plt.get_cmap('gray'))

    plt.show()

if __name__ == '__main__':
    Butterworth_low_pass_filter_test()
    ladder_shaped_low_pass_filter_test()
    index_low_pass_filter_test()
