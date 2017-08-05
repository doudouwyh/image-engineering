'''
    affine tranfer
'''
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *
from chp2.picchange import coordinate_transfer

def similarty_transfer_test():

    data = get_image_data("../pic/lena.jpg")
    h,w = data.shape

    s = 1.5
    theta = 0
    tx = 4
    ty = 8

    t = np.array([[s*np.cos(theta), s*np.sin(theta),tx],[-s*np.sin(theta),s*np.cos(theta),ty],[0,0,1]])
    am = np.matrix(t)

    newdata = np.zeros(data.shape)
    for i in range(h):
        for j in range(w):
            y,x = coordinate_transfer(i,j,am)
            if 0 < y < h and 0 < x < w:
                newdata[y,x] = data[i,j]

    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 2, 2)
    plt.title('sim_transfer')
    plt.imshow(newdata, cmap=plt.get_cmap('gray'))

    plt.show()

def  isometric_transfer_test():

    data = get_image_data("../pic/lena.jpg")
    h,w = data.shape

    e = 1
    theta = np.pi/2
    tx = 2
    ty = 0

    t = np.array([[e*np.cos(theta), np.sin(theta),tx],[-e*np.sin(theta),np.cos(theta),ty],[0,0,1]])
    am = np.matrix(t)

    newdata = np.zeros(data.shape)
    for i in range(h):
        for j in range(w):
            y,x = coordinate_transfer(i,j,am)
            if 0 < y < h and 0 < x < w:
                newdata[y,x] = data[i,j]

    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 2, 2)
    plt.title('isometric_transfer')
    plt.imshow(newdata, cmap=plt.get_cmap('gray'))

    plt.show()


def Euclidean_transfer():

    data = get_image_data("../pic/lena.jpg")
    h,w = data.shape

    theta = np.pi/2
    tx = 2
    ty = 0

    t = np.array([[np.cos(theta), np.sin(theta),tx],[-np.sin(theta),np.cos(theta),ty],[0,0,1]])
    am = np.matrix(t)

    newdata = np.zeros(data.shape)
    for i in range(h):
        for j in range(w):
            y,x = coordinate_transfer(i,j,am)
            if 0 < y < h and 0 < x < w:
                newdata[y,x] = data[i,j]

    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 2, 2)
    plt.title('euclidean_transfer')
    plt.imshow(newdata, cmap=plt.get_cmap('gray'))

    plt.show()

if __name__ == '__main__':
    # similarty_transfer_test()
    # isometric_transfer_test()
    Euclidean_transfer()
