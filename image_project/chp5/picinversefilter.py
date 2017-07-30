'''
    inverse filter
'''

import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *

def inverse_filter_test():
    data = get_image_data("../pic/lena.jpg")
    fft = np.fft.fft2(data)
    fft_shift = np.fft.fftshift(fft)

    k = 0.0006
    atmblurdata = np.copy(fft_shift)
    h,w = atmblurdata.shape
    for i in range(h):
        for j in range(w):
            hf = np.exp(-k*((i**2 + j**2)**(5.0/6)))  #atmospheric turbulence blur model
            atmblurdata[i,j] = hf*atmblurdata[i,j]

    T = 1
    a = 0.02
    b = 0.02
    mblurdata = np.copy(fft_shift)
    h,w = atmblurdata.shape
    for i in range(h):
        for j in range(w):
            R = np.pi*(a*i + b*j)
            if R == 0:
                continue
            hf = (T/R * np.sin(R)) * np.exp(-1j*R)     #motion blur model
            mblurdata[i,j] = hf*mblurdata[i,j]

    w0 = 20
    #atm inverse filter
    atm_inverse_data = np.copy(fft_shift)
    h,w = atm_inverse_data.shape
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i-h/2)**2 + (j-w/2)**2)
            if d <= w0:
                atm_inverse_data[i,j] = fft_shift[i,j] / atmblurdata[i,j]


    #m inverse filter
    m_inverse_data = np.copy(fft_shift)
    h,w = m_inverse_data.shape
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i-h/2)**2 + (j-w/2)**2)
            if d <= w0:
                m_inverse_data[i,j] = fft_shift[i,j] / mblurdata[i,j]

    plt.subplot(1, 5, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 5, 2)
    ifft_atmblur = np.fft.ifft2(atmblurdata)
    plt.title('atmblur')
    plt.imshow(np.abs(ifft_atmblur), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 5, 3)
    ifft_mblur = np.fft.ifft2(mblurdata)
    plt.title('mblur')
    plt.imshow(np.abs(ifft_mblur), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 5, 4)
    ifft_atminverseblur = np.fft.ifft2(atm_inverse_data)
    plt.title('atmbluerinverse')
    plt.imshow(np.abs(ifft_atminverseblur), cmap=plt.get_cmap('gray'))

    plt.subplot(1, 5, 5)
    ifft_minverseblur = np.fft.ifft2(m_inverse_data)
    plt.title('mblurinverse')
    plt.imshow(np.abs(ifft_minverseblur), cmap=plt.get_cmap('gray'))

    plt.show()

if __name__ == '__main__':
    inverse_filter_test()
