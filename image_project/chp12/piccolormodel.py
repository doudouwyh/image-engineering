"""
    color
"""

import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *

def color_test():
    img = Image.open("../pic/lena_color.jpg").convert('RGBA')
    data = img.load()

    R = []
    G = []
    B = []
    A = []
    for w in range(img.size[0]):
        for h in range(img.size[1]):
            r, g, b, a = data[h,w]
            R.append(r)
            G.append(g)
            B.append(b)
            A.append(a)

    data_R = np.array(R).reshape(img.size)
    data_G = np.array(G).reshape(img.size)
    data_B = np.array(B).reshape(img.size)

    data_Gray = 0.2125*data_R + 0.7154*data_G +0.0721*data_B

    plt.subplot(1,4,1)
    plt.imshow(data_R,cmap=plt.get_cmap('gray'))
    plt.subplot(1, 4, 2)
    plt.imshow(data_G,cmap=plt.get_cmap('gray'))
    plt.subplot(1, 4, 3)
    plt.imshow(data_B,cmap=plt.get_cmap('gray'))
    plt.subplot(1, 4, 4)
    plt.imshow(data_Gray,cmap=plt.get_cmap('gray'))


    plt.show()

if __name__ == '__main__':
    color_test()