'''
    interpolation
'''
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *


def nn_interpolation_test():
    data = get_image_data("../pic/lena.jpg")
    h,w = data.shape

    dstsize = [512,512]
    dst = np.zeros(dstsize)
    factory = float(h) / dstsize[0]
    factorx = float(w) / dstsize[1]

    for i in range(dstsize[0]):
        for j in range(dstsize[1]):
            y = float(i) * factory
            x = float(j) * factorx
            if y + 1 > h:
                y -= 1
            if x + 1 > w:
                x -= 1
            cy = np.ceil(y)
            fy = cy - 1
            cx = np.ceil(x)
            fx = cx - 1

            sx = 0
            sy = 0
            if x - np.floor(x) > 1e-6 or y - np.floor(y) > 1e-6:
                if x - fx < cx -x:
                    sx = fx
                else:
                    sx = cx

                if y - fy < cy - y:
                    sy = fy
                else:
                    sy = cy
                dst[i,j] = data[int(sy),int(sx)]
            else:
                dst[i,j] = data[int(y),int(x)]

    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 2, 2)
    plt.title('nn_interpolation')
    plt.imshow(dst, cmap=plt.get_cmap('gray'))

    plt.show()


def bilinear_interpolation_test():
    data = get_image_data("../pic/lena.jpg")
    h,w = data.shape

    dstsize = [512,512]
    dst = np.zeros(dstsize)
    factory = float(h) / dstsize[0]
    factorx = float(w) / dstsize[1]

    for i in range(dstsize[0]):
        for j in range(dstsize[1]):
            y = float(i) * factory
            x = float(j) * factorx
            if y + 1 > h:
                y -= 1
            if x + 1 > w:
                x -= 1
            cy = np.ceil(y)
            fy = cy - 1
            cx = np.ceil(x)
            fx = cx - 1
            w1 = (cx - x) * (cy - y)
            w2 = (x - fx) * (cy - y)
            w3 = (cx - x) * (y - fy)
            w4 = (x - fx) * (y - fy)
            if x - np.floor(x) > 1e-6 or y - np.floor(y) > 1e-6:
                t = data[int(fy), int(fx)] * w1 + data[int(fy), int(cx)] * w2 + data[int(cy), int(fx)] * w3 + data[int(cy), int(cx)] * w4
                dst[i, j] = t
            else:
                dst[i, j] = data[int(y),int(x)]

    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 2, 2)
    plt.title('bilinear_interpolation')
    plt.imshow(dst, cmap=plt.get_cmap('gray'))

    plt.show()


def getdata(cy,cx,y,x,h,w,src):
    wy = [0,0,0,0]
    wx = [0,0,0,0]

    for i in range(4):
        if 0 <= cy[i] < h:
            if i == 0 or i == 3:
                wy[i] = 4 - 8*(np.abs(y-cy[i])) + 5*((np.abs(y-cy[i]))**2) - (np.abs(y-cy[i]))**3
            else:
                wy[i] = 1 - 2 * ((np.abs(y - cy[i]))**2) + (np.abs(y - cy[i])) ** 3

        if 0 <= cx[i] < w:
            if i == 0 or i == 3:
                wx[i] = 4 - 8*(np.abs(x-cx[i])) + 5*((np.abs(x-cx[i]))**2)  - (np.abs(x-cx[i]))**3
            else:
                wx[i] = 1 - 2 * ((np.abs(x - cx[i]))**2) + (np.abs(x - cx[i])) ** 3

    sum = 0.0
    for i in range(4):
        for j in range(4):
            if 0 <= cy[i] < h and 0 <= cx[j] < w:
                sum += wy[i]*wx[j]*src[cy[i],cx[j]]

    return sum


def trilinear_interpolation():
    data = get_image_data("../pic/lena.jpg")
    h,w = data.shape

    dstsize = [400,600]
    dst = np.zeros(dstsize)

    factory = float(h) / dstsize[0]
    factorx = float(w) / dstsize[1]

    for i in range(dstsize[0]):
        for j in range(dstsize[1]):
            y = float(i) * factory
            x = float(j) * factorx
            if y + 1 > h:
                y -= 1
            if x + 1 > w:
                x -= 1

            if x - np.floor(x) > 1e-6 or y - np.floor(y) > 1e-6:
                    if x - np.floor(x) <= 1e-6:
                        w1 = y - np.floor(y)
                        w2 = np.ceil(y) - y
                        dst[i,j] = data[int(np.floor(y)),int(x)]*w1 + data[int(np.ceil(y)),int(x)]*w2
                        continue
                    if y - np.floor(y) <= 1e-6:
                        w1 = x - np.floor(x)
                        w2 = np.ceil(x) - x
                        dst[i,j] = data[int(y),int(np.floor(x))]*w1 + data[int(y),int(np.ceil(x))]*w2
                        continue

                    cy = [int(np.floor(y)-1),int(np.floor(y)),int(np.ceil(y)),int(np.ceil(y)+1)]
                    cx = [int(np.floor(x)-1),int(np.floor(x)),int(np.ceil(x)),int(np.ceil(x)+1)]
                    dst[i,j] = getdata(cy, cx, y, x, h, w, data)
            else:
                dst[i,j] = data[int(y),int(x)]

    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 2, 2)
    plt.title('trilinear_interpolation')
    plt.imshow(dst, cmap=plt.get_cmap('gray'))

    plt.show()

if __name__ == '__main__':
    # nn_interpolation_test()
    # bilinear_interpolation_test()
    trilinear_interpolation()