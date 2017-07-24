'''
    gray mapping: reverse,enhance,DRC,GAMA_CORRECT
'''

from PIL import Image
import numpy as np


def get_image_data(filename):
    im = Image.open(filename)
    data = im.getdata()
    data = np.array(data,'float').reshape(im.size)
    return data


def  reverse(x):
    return 255-x


def contrast_enhance(x):
    assert (x >= 0)
    if x <=120:
        return x/2
    elif x>120 and x<136:
        return x*4-150
    else:
        return x/2

# Dynamic range compression
# t =Clog(1+|x|)
def DRC(x,c):
    t = c*np.log(1+x)
    if t < 0:
        return 0
    else:
        return t

# Gamma Correct
def gama_correct(x,c,r):
    t = c * (x**r)
    if t > 255:
        return 255
    else:
        return t

def img_contrast(img,fun):
    newdata = np.zeros(img.shape)
    width,height = img.shape
    for i in range(width):
        for j in range(height):
            newdata[i,j] = fun(img[i,j])
    return newdata


def img_contrast2(img,fun,c):
    newdata = np.zeros(img.shape)
    width,height = img.shape
    for i in range(width):
        for j in range(height):
            newdata[i,j] = fun(img[i,j],c)
    return newdata

def img_contrast3(img,fun,c,r):
    newdata = np.zeros(img.shape)
    width,height = img.shape
    for i in range(width):
        for j in range(height):
            newdata[i,j] = fun(img[i,j],c,r)
    return newdata

def pic_contrast_test():
    data = get_image_data("../pic/lena.jpg")
    Image.fromarray(data.astype(np.uint8)).show("origin")

    last = img_contrast(data,reverse)
    Image.fromarray(last.astype(np.uint8)).show("last")

def pic_contrast_test2():
    data = get_image_data("../pic/lena.jpg")
    Image.fromarray(data.astype(np.uint8)).show("origin")

    last = img_contrast(data,contrast_enhance)
    Image.fromarray(last.astype(np.uint8)).show("last")


def DRC_test():
    data = get_image_data("../pic/lena.jpg")
    Image.fromarray(data.astype(np.uint8)).show("origin")

    last = img_contrast2(data,DRC,c=3)
    Image.fromarray(last.astype(np.uint8)).show("last")

def gama_correct_test():
    data = get_image_data("../pic/lena.jpg")
    Image.fromarray(data.astype(np.uint8)).show("origin")

    last = img_contrast3(data,gama_correct,c=3,r=2.5)
    Image.fromarray(last.astype(np.uint8)).show("last")

if __name__ == '__main__':
    pic_contrast_test()
    pic_contrast_test2()
    DRC_test()
    gama_correct_test()

