'''
    operation between pictures
'''

import Image
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
    pass

def pic_div_test():
    pass

def pic_mul_test():
    pass


if __name__ == '__main__':
    pic_add_test()

