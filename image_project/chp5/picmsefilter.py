'''
    mse filter
    wiener filter
'''

import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *

def motion_process(len, size):
    sx, sy = size
    PSF = np.zeros((sx, sy))
    PSF[int(sy / 2):int(sy /2 + 1), int(sx / 2 - len / 2):int(sx / 2 + len / 2)] = 1
    return PSF / PSF.sum()

def make_blurred(input, PSF, eps):
    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF) + eps
    blurred = np.fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(np.fft.fftshift(blurred))
    return blurred

def wiener(input, PSF, eps):
    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF) + eps
    result = np.fft.ifft2(input_fft / PSF_fft)
    result = np.abs(np.fft.fftshift(result))

    return result

def wiener_filter_test():
    image = get_image_data("../pic/lena.jpg")

    plt.figure(1)
    plt.xlabel("Origin")
    plt.gray()
    plt.imshow(image)

    plt.figure(2)
    plt.gray()
    data = image
    PSF = motion_process(30, data.shape)
    blurred = np.abs(make_blurred(data, PSF, 1e-3))

    plt.subplot(221)
    plt.xlabel("Motion blurred")
    plt.imshow(blurred)

    result = wiener(blurred, PSF, 1e-3)
    plt.subplot(222)
    plt.xlabel("wiener deblurred")
    plt.imshow(result)

    blurred += 0.1 * blurred.std() * np.random.standard_normal(blurred.shape)

    plt.subplot(223)
    plt.xlabel("motion & noisy blurred")
    plt.imshow(blurred)

    result = wiener(blurred, PSF, 0.1 + 1e-3)
    plt.subplot(224)
    plt.xlabel("wiener deblurred")
    plt.imshow(result)

    plt.show()

def mse_filter_test():
    pass

if __name__ == '__main__':
    wiener_filter_test()
