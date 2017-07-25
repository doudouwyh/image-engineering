'''
    template operation
'''
from PIL import Image
import numpy as np


def get_image_data(filename):
    im = Image.open(filename)
    data = im.getdata()
    data = np.array(data,'float').reshape(im.size)
    return data

def conv(image,x,y,template):
    w,h = template.shape
    c = []
    count = 0
    for i in range(x-w/2,x+w/2+1):
        for j in range(y-h/2,y+h/2+1):
            if i<0 or j<0:
                c.append(0)
            else:
                c.append(image[i,j])
                count += template[i+w/2,j+h/2]
    template.reshape(1,w*h)
    return np.sum(np.array(c) * template),count

def template_conv(image,template):
    width,height = image.shape
    newdata = np.zeros(image.shape)
    for i in range(width):
        for j in range(height):
            newdata[i,j],count = conv(image,i,j,template)
            if newdata[i,j] < 0:
                newdata[i,j] = 0
            if newdata[i,j] > 255:
                newdata[i,j] = 255

    return newdata

#flag:  1-smooth, 2-sharpen
def linear_filter(image,template,flag):
    width,height = image.shape
    newdata = np.zeros(image.shape)
    for i in range(width):
        for j in range(height):
            newdata[i,j],count = conv(image,i,j,template)
            if newdata[i,j] < 0:
                newdata[i,j] = 0
            if newdata[i,j] > 255:
                newdata[i,j] = 255
            if flag == 1:
                newdata[i,j] = newdata/(count*1.0)


    return newdata

#high-frequency emphasis filtering
# A : coefficient
def hfe_filter(image,A):
    #gauss smooth
    gauss = np.array([1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1]).reshape(5,5)
    smoothimg = linear_filter(image,gauss,1)

    #unsharp mask
    um = image - smoothimg

    #high-frequency emphasis
    newdata = [x*A for x in um] + image
    return newdata

def template_conv_test():
    data = get_image_data("../pic/lena.jpg")
    Image.fromarray(data.astype(np.uint8)).show("origin")

    sobel = np.array([-2,-1,0,-1,0,1,0,1,2]).reshape(3,3)
    last = template_conv(data,sobel)
    Image.fromarray(last.astype(np.uint8)).show("last")

def linear_filter_test():
    data = get_image_data("../pic/lena.jpg")
    Image.fromarray(data.astype(np.uint8)).show("origin")

    #weighted mean
    sobel = np.array([1,2,1,2,4,2,1,2,1]).reshape(3,3)
    lastmean = linear_filter(data,sobel,1)
    Image.fromarray(lastmean.astype(np.uint8)).show("mean")

    #gauss
    gauss = np.array([1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1]).reshape(5,5)
    lastgauss = linear_filter(data,gauss,1)
    Image.fromarray(lastgauss.astype(np.uint8)).show("gauss")

    #laplace
    laplace = np.array([0,-1,0,-1,4,-1,0,-1,0]).reshape(3,3)
    lastlap = linear_filter(data,laplace,2)
    Image.fromarray(lastlap.astype(np.uint8)).show("laplace")



if __name__ == '__main__':
    template_conv_test()
    linear_filter_test()