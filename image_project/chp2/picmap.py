'''
    gray mapping: reverse,enhance,DRC,GAMA_CORRECT
'''
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *


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
    for i in range(height):
        for j in range(width):
            newdata[i,j] = fun(img[i,j])
    return newdata


def img_contrast2(img,fun,c):
    newdata = np.zeros(img.shape)
    width,height = img.shape
    for i in range(height):
        for j in range(width):
            newdata[i,j] = fun(img[i,j],c)
    return newdata

def img_contrast3(img,fun,c,r):
    newdata = np.zeros(img.shape)
    height,width = img.shape
    for i in range(height):
        for j in range(width):
            newdata[i,j] = fun(img[i,j],c,r)
    return newdata

def pic_contrast_test():
    data = get_image_data("../pic/lena.jpg")

    newdata = img_contrast(data,reverse)

    plt.subplot(1,2,1)
    plt.title("origin")
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1,2,2)
    plt.title("reverse")
    plt.imshow(newdata, cmap=plt.get_cmap('gray'))
    plt.show()

def pic_contrast_test2():
    data = get_image_data("../pic/lena.jpg")
    newdata = img_contrast(data,contrast_enhance)

    plt.subplot(1,2,1)
    plt.title("origin")
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1,2,2)
    plt.title("locale_nhance")
    plt.imshow(newdata, cmap=plt.get_cmap('gray'))
    plt.show()


def DRC_test():
    data = get_image_data("../pic/lena.jpg")
    newdata = img_contrast2(data,DRC,c=3)

    plt.subplot(1,2,1)
    plt.title("origin")
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1,2,2)
    plt.title("DRC")
    plt.imshow(newdata, cmap=plt.get_cmap('gray'))
    plt.show()

def gama_correct_test():
    data = get_image_data("../pic/lena.jpg")
    newdata = img_contrast3(data,gama_correct,c=0.5,r=1.5)
    print newdata

    plt.subplot(1,2,1)
    plt.title("origin")
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1,2,2)
    plt.title("gama_correct")
    plt.imshow(newdata, cmap=plt.get_cmap('gray'))
    plt.show()

if __name__ == '__main__':
    # pic_contrast_test()
    # pic_contrast_test2()
    # DRC_test()
      gama_correct_test()

