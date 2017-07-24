'''
    operation between pictures: add,sub,div,mul
'''

from PIL import Image
import numpy as np


def get_image_data(filename):
    im = Image.open(filename)
    data = im.getdata()
    data = np.array(data,'float').reshape(im.size)
    return data


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
    Image.fromarray(data.astype(np.uint8)).show("origin")

    width,height = data.shape
    pics = []
    for i in range(10):
        noise = np.random.normal(loc=0.0, scale=16.0, size=width * height).reshape(data.shape)
        pic = add_noise(data,noise)
        pics.append(pic)

    last = sum(pics)/10
    Image.fromarray(last.astype(np.uint8)).show("last")

def pic_sub_test():
    sub1 = get_image_data("../pic/img1.jpg")
    Image.fromarray(sub1.astype(np.uint8)).show("sub1")

    sub2 = get_image_data("../pic/img2.jpg")
    Image.fromarray(sub2.astype(np.uint8)).show("sub2")

    last = image_sub(sub1,sub2)
    Image.fromarray(last.astype(np.uint8)).show("last")


def pic_div_test():
    div1 = get_image_data("../pic/img1.jpg")
    Image.fromarray(div1.astype(np.uint8)).show("div1")

    div2 = get_image_data("../pic/img2.jpg")
    Image.fromarray(div2.astype(np.uint8)).show("div2")

    last = image_div(div1,div2)
    Image.fromarray(last.astype(np.uint8)).show("last")

def pic_div_test2():
    div1 = get_image_data("../pic/img1.jpg")
    Image.fromarray(div1.astype(np.uint8)).show("div1")

    div2 = get_image_data("../pic/img2.jpg")
    Image.fromarray(div2.astype(np.uint8)).show("div2")

    # c= 10
    last = image_div2(div1,10)
    Image.fromarray(last.astype(np.uint8)).show("last")

def pic_mul_test():
    mul1 = get_image_data("../pic/img1.jpg")
    Image.fromarray(mul1.astype(np.uint8)).show("mul1")

    mul2 = get_image_data("../pic/img2.jpg")
    Image.fromarray(mul2.astype(np.uint8)).show("mul2")

    last = image_mul(mul1,mul2)
    Image.fromarray(last.astype(np.uint8)).show("last")

def pic_mul_test2():
    mul1 = get_image_data("../pic/img1.jpg")
    Image.fromarray(mul1.astype(np.uint8)).show("mul1")

    #c=3
    last = image_mul2(mul1,3)
    Image.fromarray(last.astype(np.uint8)).show("last")



if __name__ == '__main__':
    #pic_add_test()
    #pic_sub_test()
    pic_div_test()
    pic_div_test2()
    pic_mul_test()
    pic_mul_test2()

