'''
    band  pass filter  and band stop filter
'''

import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *

def Butterworth_band_stop_pass_filter_test():

    data = get_image_data("../pic/lena.jpg")
    fft = np.fft.fft2(data)
    fft_shift = np.fft.fftshift(fft)

    n = 1
    W = 10
    Dd = 40
    fft_shift_stop = np.copy(fft_shift)
    fft_shift_pass = np.copy(fft_shift)
    h,w = fft_shift_stop.shape
    for i in range(h):
        for j in range(w):
            d2 = (i-w/2)**2 + (j-h/2)**2
            d   = np.sqrt(d2)
            if d2 ==  Dd**2:
                continue
            hf = 1/(1+((d*W)/(d2-Dd**2))**(2*n))
            temp = fft_shift_stop[i,j]
            fft_shift_stop[i,j] = hf*temp
            fft_shift_pass[i,j] = (1-hf)*temp

    plt.subplot(3, 2, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(3, 2, 2)
    plt.title('fftshift')
    plt.imshow(np.log(np.abs(fft_shift)), cmap=plt.get_cmap('gray'))

    plt.subplot(3, 2, 3)
    plt.title('fftshiftstop')
    plt.imshow(np.log(np.abs(fft_shift_stop)), cmap=plt.get_cmap('gray'))

    plt.subplot(3, 2, 4)
    ifftstop = np.fft.ifft2(fft_shift_stop)
    plt.title('ifft_bw_band_stop')
    plt.imshow(np.abs(ifftstop), cmap=plt.get_cmap('gray'))

    plt.subplot(3, 2, 5)
    plt.title('fftshiftpass')
    plt.imshow(np.log(np.abs(fft_shift_pass)), cmap=plt.get_cmap('gray'))

    plt.subplot(3, 2, 6)
    ifftpass = np.fft.ifft2(fft_shift_pass)
    plt.title('ifft_bw_band_pass')
    plt.imshow(np.abs(ifftpass), cmap=plt.get_cmap('gray'))

    plt.show()

def Butterworth_notch_stop_filter_test():

    data = get_image_data("../pic/lena.jpg")
    fft = np.fft.fft2(data)
    fft_shift = np.fft.fftshift(fft)

    U = [77,159,84,167]
    V = [59,59,-54,-54]
    Dck = [30,50,30,50]
    n = 1

    fft_shift_stop = np.copy(fft_shift)
    h,w = fft_shift_stop.shape
    for i in range(h):
        for j in range(w):
            hf = 1
            for k in range(len(U)):
                dp = np.sqrt((i - w/2 - V[k])**2 + (j - h/2 -U[k])**2)
                dm = np.sqrt((i - w/2 + V[k])**2 + (j - h/2 +U[k])**2)
                if dp == 0 or dm == 0:
                    continue
                hf *= ((1/(1+(Dck[k]/dp)**(2*n))) * (1/(1+(Dck[k]/dm)**(2*n))))
            fft_shift_stop[i,j] = hf* fft_shift_stop[i,j]

    plt.subplot(1,4, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 2)
    plt.title('fftshift')
    plt.imshow(np.log(np.abs(fft_shift)), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 3)
    plt.title('fftshiftstop')
    plt.imshow(np.log(np.abs(fft_shift_stop)), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 4)
    ifftstop = np.fft.ifft2(fft_shift_stop)
    plt.title('ifft_bw_notch_stop')
    plt.imshow(np.abs(ifftstop), cmap=plt.get_cmap('gray'))

    plt.show()

def Butterworth_homomorphic_filter_test():
    data = get_image_data("../pic/lena.jpg")
    fft = np.fft.fft2(data)
    fft_shift = np.fft.fftshift(fft)

    HL = 0.5
    HH = 2.0
    n = 1
    d0 = 10
    fft_shift_hm = np.copy(fft_shift)
    h,w = fft_shift_hm.shape
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i-w/2)**2 + (j-h/2)**2)
            if d == 0:
                continue
            hf = 1/(1+(d0/d)**(2*n))
            hm = (HH-HL)*hf + HL
            fft_shift_hm[i,j] = hm*fft_shift_hm[i,j]

    plt.subplot(1, 4, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 2)
    plt.title('fftshift')
    plt.imshow(np.log(np.abs(fft_shift)), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 3)
    plt.title('fftshifthomo')
    plt.imshow(np.log(np.abs(fft_shift_hm)), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 4)
    ifft = np.fft.ifft2(fft_shift_hm)
    plt.title('ifft_bw_homo_filter')
    plt.imshow(np.abs(ifft), cmap=plt.get_cmap('gray'))

    plt.show()


def Gauss_homomorphic_filter_test():
    data = get_image_data("../pic/lena.jpg")
    fft = np.fft.fft2(data)
    fft_shift = np.fft.fftshift(fft)

    HL = 0.5
    HH = 2.0
    c = 2
    d0 = 10
    fft_shift_hm = np.copy(fft_shift)
    h,w = fft_shift_hm.shape
    for i in range(h):
        for j in range(w):
            d2 = (i-w/2)**2 + (j-h/2)**2
            if d2 == 0:
                continue
            hf = 1 - np.exp(-c*d2/(d0**2))
            hm = (HH-HL)*hf + HL
            fft_shift_hm[i,j] = hm*fft_shift_hm[i,j]

    plt.subplot(1, 4, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 2)
    plt.title('fftshift')
    plt.imshow(np.log(np.abs(fft_shift)), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 3)
    plt.title('fftshifthomo')
    plt.imshow(np.log(np.abs(fft_shift_hm)), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 4, 4)
    ifft = np.fft.ifft2(fft_shift_hm)
    plt.title('ifft_gauss_homo_filter')
    plt.imshow(np.abs(ifft), cmap=plt.get_cmap('gray'))

    plt.show()

if __name__ == '__main__':
    #Butterworth_band_stop_pass_filter_test()
    #Butterworth_notch_stop_filter_test()
    # Butterworth_homomorphic_filter_test()
     Gauss_homomorphic_filter_test()
