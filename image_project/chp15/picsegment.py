"""
    sobel
"""

import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *


def sobel_test():
    data = get_image_data("../pic/lena.jpg")
    h,w = data.shape

    xdata = np.zeros(data.shape)
    ydata = np.zeros(data.shape)
    N = 3
    sobel_x = np.matrix('-1 0 1; -2 0 2; -1 0 1')
    sobel_y = np.matrix('1 2 1; 0 0 0; -1 -2 -1')
    for i in range(h):
        for j in range(w):
            pass



    plt.subplot(1, 3, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 3, 2)
    plt.title('dct')
    plt.imshow(cdata, cmap=plt.get_cmap('gray'))

    plt.show()



if __name__ == '__main__':
    sobel_test()