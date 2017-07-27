'''
    operation between pictures: add,sub,div,mul
'''

import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *

def add_noise(data,noise):
    assert(data.shape == noise.shape)
    newdata = np.zeros(data.shape)
    width,height = data.shape
    for i in range(width):
        for j in range(height):
            newdata[i,j] = data[i,j]+noise[i,j]
            if newdata[i,j] > 255:
                newdata[i,j] = 255
    return newdata

def image_sub(imag1,imag2):
    assert imag1.shape == imag2.shape
    newdata = np.zeros(imag1.shape)
    width,height = imag1.shape
    for i in range(width):
        for j in range(height):
            newdata[i,j] = imag2[i,j] - imag1[i,j]
            if newdata[i,j] <0:
                newdata[i,j] = 0
    return newdata

def image_div(imag1,imag2):
    assert imag1.shape == imag2.shape
    newdata = np.zeros(imag1.shape)
    width,height = imag1.shape
    for i in range(width):
        for j in range(height):
            if imag1[i,j] == 0:
                newdata[i,j] = imag2[i,j]
            else:
                newdata[i, j] = imag2[i, j] / imag1[i,j]
    return newdata

def image_div2(imag1,c):
    assert (c>0)
    return imag1/c


def image_mul(imag1,imag2):
    assert imag1.shape == imag2.shape
    newdata = np.zeros(imag1.shape)
    width,height = imag1.shape
    for i in range(width):
        for j in range(height):
            newdata[i,j] = imag1[i,j]*imag2[i,j]
            if newdata[i,j] > 255:
                newdata[i,j] = 255
    return newdata

def image_mul2(imag1,c):
    assert (c>0)
    newdata = np.zeors(imag1.shape)
    width,height = imag1.shape
    for i in range(width):
        for j in range(height):
            newdata[i,j] = imag1[i,j]*c
            if newdata[i,j] > 255:
                newdata[i,j] = 255
    return newdata


def pic_add_test():
    data = get_image_data("../pic/lena.jpg")

    width,height = data.shape
    pics = []
    for i in range(10):
        noise = np.random.normal(loc=0.0, scale=16.0, size=width * height).reshape(data.shape)
        pic = add_noise(data,noise)
        pics.append(pic)

    last = sum(pics)/10
    plt.subplot(1,2,1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1,2,2)
    plt.title('last')
    plt.imshow(last, cmap=plt.get_cmap('gray'))

    plt.show()


def pic_sub_test():
    sub1 = get_image_data("../pic/img1.jpg")
    sub2 = get_image_data("../pic/img2.jpg")
    last = image_sub(sub1,sub2)

    plt.subplot(1,3,1)
    plt.title('origin1')
    plt.imshow(sub1, cmap=plt.get_cmap('gray'))

    plt.subplot(1,3,2)
    plt.title('origin2')
    plt.imshow(sub2, cmap=plt.get_cmap('gray'))

    plt.subplot(1,3,2)
    plt.title('origin2')
    plt.imshow(last, cmap=plt.get_cmap('gray'))

    plt.show()


def pic_div_test():
    div1 = get_image_data("../pic/img1.jpg")
    div2 = get_image_data("../pic/img2.jpg")
    last = image_div(div1,div2)

    plt.subplot(1,3,1)
    plt.title('origin1')
    plt.imshow(div1, cmap=plt.get_cmap('gray'))

    plt.subplot(1,3,2)
    plt.title('origin2')
    plt.imshow(div2, cmap=plt.get_cmap('gray'))

    plt.subplot(1,3,2)
    plt.title('origin1/origin2')
    plt.imshow(last, cmap=plt.get_cmap('gray'))

    plt.show()


def pic_div_test2():
    div1 = get_image_data("../pic/img1.jpg")

    # c= 10
    last = image_div2(div1,10)

    plt.subplot(1,2,1)
    plt.title('origin1')
    plt.imshow(div1, cmap=plt.get_cmap('gray'))

    plt.subplot(1,2,2)
    plt.title('origin1/10')
    plt.imshow(last, cmap=plt.get_cmap('gray'))

    plt.show()


def pic_mul_test():
    mul1 = get_image_data("../pic/img1.jpg")
    mul2 = get_image_data("../pic/img2.jpg")
    last = image_mul(mul1,mul2)

    plt.subplot(1,3,1)
    plt.title('origin1')
    plt.imshow(mul1, cmap=plt.get_cmap('gray'))

    plt.subplot(1,3,2)
    plt.title('origin2')
    plt.imshow(mul2, cmap=plt.get_cmap('gray'))

    plt.subplot(1,3,3)
    plt.title('origin1*origin2')
    plt.imshow(last, cmap=plt.get_cmap('gray'))

    plt.show()

def pic_mul_test2():
    mul1 = get_image_data("../pic/img1.jpg")

    #c=3
    last = image_mul2(mul1,3)

    plt.subplot(1,2,1)
    plt.title('origin1')
    plt.imshow(mul1, cmap=plt.get_cmap('gray'))

    plt.subplot(1,2,2)
    plt.title('origin1*3')
    plt.imshow(last, cmap=plt.get_cmap('gray'))

    plt.show()



if __name__ == '__main__':
    pic_add_test()
    # pic_sub_test()
    # pic_div_test()
    # pic_div_test2()
    # pic_mul_test()
    # pic_mul_test2()

