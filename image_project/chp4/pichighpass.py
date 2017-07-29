'''
    high frequence pass filter
'''

import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *

def Butterworth_high_pass_filter_test():

    data = get_image_data("../pic/lena.jpg")
    fft = np.fft.fft2(data)
    fft_shift = np.fft.fftshift(fft)

    n = 1
    d0 = 30
    fft_shift_high = np.copy(fft_shift)
    h,w = fft_shift_high.shape
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i-h/2)**2 + (j-w/2)**2)
            if d == 0:
                continue
            hf = 1/(1+(d0/d)**(2*n))
            fft_shift_high[i,j] = hf*fft_shift_high[i,j]

    plt.subplot(1, 4, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 2)
    plt.title('fftshift')
    plt.imshow(np.log(np.abs(fft_shift)), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 3)
    plt.title('fftshifthigh')
    plt.imshow(np.log(np.abs(fft_shift_high)), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 4)
    ifft = np.fft.ifft2(fft_shift_high)
    plt.title('ifft_bw_low_pass')
    plt.imshow(np.abs(ifft), cmap=plt.get_cmap('gray'))

    plt.show()


def high_enhance_filter_test():

    data = get_image_data("../pic/lena.jpg")
    fft = np.fft.fft2(data)
    fft_shift = np.fft.fftshift(fft)

    c = 0.6
    k = 2
    n = 1
    d0 = 30
    fft_shift_high = np.copy(fft_shift)
    h,w = fft_shift_high.shape
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i-h/2)**2 + (j-w/2)**2)
            if d == 0:
                continue
            hf = 1/(1+(d0/d)**(2*n))           #Batterworth
            fft_shift_high[i,j] = (k*hf+c)*fft_shift_high[i,j]

    plt.subplot(1, 4, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 2)
    plt.title('fftshift')
    plt.imshow(np.log(np.abs(fft_shift)), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 3)
    plt.title('fftshifthigh')
    plt.imshow(np.log(np.abs(fft_shift_high)), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 4)
    ifft = np.fft.ifft2(fft_shift_high)
    plt.title('ifft_bw_high_enhance')
    plt.imshow(np.abs(ifft), cmap=plt.get_cmap('gray'))

    plt.show()

def high_lifting_filter_test():

    data = get_image_data("../pic/lena.jpg")
    fft = np.fft.fft2(data)
    fft_shift = np.fft.fftshift(fft)

    A = 2
    n = 1
    d0 = 30
    fft_shift_high = np.copy(fft_shift)
    h,w = fft_shift_high.shape
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i-h/2)**2 + (j-w/2)**2)
            if d == 0:
                continue
            hf = 1/(1+(d0/d)**(2*n))           # Batterworth
            fft_shift_high[i,j] = hf*fft_shift_high[i,j]

    # lifting = (A-1)*origin + high
    fft_shift_lifting = (A-1)*fft_shift + fft_shift_high

    plt.subplot(1, 4, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 2)
    plt.title('fftshift')
    plt.imshow(np.log(np.abs(fft_shift)), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 3)
    plt.title('fftshifthigh')
    plt.imshow(np.log(np.abs(fft_shift_lifting)), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 4)
    ifft = np.fft.ifft2(fft_shift_lifting)
    plt.title('ifft_bw_high_lifting')
    plt.imshow(np.abs(ifft), cmap=plt.get_cmap('gray'))

    plt.show()

def  ladder_shaped_high_pass_filter_test():
    data = get_image_data("../pic/lena.jpg")
    fft = np.fft.fft2(data)
    fft_shift = np.fft.fftshift(fft)

    d0 = 10
    df = 20
    fft_shift_high = np.copy(fft_shift)
    h,w = fft_shift_high.shape
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i-h/2)**2 + (j-w/2)**2)
            if d <= d0:
                hf = 0
            elif df > d > d0:
                hf = (d-d0)/(df-d0)
            else:
                hf = 1
            fft_shift_high[i, j] = hf * fft_shift_high[i, j]

    plt.subplot(1, 4, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 2)
    plt.title('fftshift')
    plt.imshow(np.log(np.abs(fft_shift)), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 3)
    plt.title('fftshifthigh')
    plt.imshow(np.log(np.abs(fft_shift_high)), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 4)
    ifft = np.fft.ifft2(fft_shift_high)
    plt.title('ifft_ls_high_pass')
    plt.imshow(np.abs(ifft), cmap=plt.get_cmap('gray'))

    plt.show()


def index_high_pass_filter_test():
    data = get_image_data("../pic/lena.jpg")
    fft = np.fft.fft2(data)
    fft_shift = np.fft.fftshift(fft)

    n = 1
    d0 = 30
    fft_shift_high = np.copy(fft_shift)
    h, w = fft_shift_high.shape
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i - h / 2) ** 2 + (j - w / 2) ** 2)
            hf = 1 - np.exp(- (d / d0) ** n)
            fft_shift_high[i, j] = hf * fft_shift_high[i, j]

    plt.subplot(1, 4, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 2)
    plt.title('fftshift')
    plt.imshow(np.log(np.abs(fft_shift)), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 3)
    plt.title('fftshifthigh')
    plt.imshow(np.log(np.abs(fft_shift_high)), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 4)
    ifft = np.fft.ifft2(fft_shift_high)
    plt.title('ifft_index_high_pass')
    plt.imshow(np.abs(ifft), cmap=plt.get_cmap('gray'))

    plt.show()

if __name__ == '__main__':
    Butterworth_high_pass_filter_test()
    high_enhance_filter_test()
    high_lifting_filter_test()
    ladder_shaped_high_pass_filter_test()
    index_high_pass_filter_test()