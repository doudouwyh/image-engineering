'''
    template operation
    linear filter
    nonlinear filter
'''
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from chp2.pichist import hist_enhance
from common.common import *

SMOOTH = 1
SHARPEN = 2

def conv(image,x,y,template):
    h,w = template.shape
    c = []
    count = 0
    for i in range(x-w/2,x+w/2+1):
        for j in range(y-h/2,y+h/2+1):
            if i<0 or i > 255 or j<0 or j>255:
                c.append(0)
            else:
                c.append(image[i,j])
                count += template[i-x+1,j-y+1]
    t = template.reshape(1,w*h)
    return np.sum(np.array(c) * t),count

def get_median(image,x,y,N):
    assert (N > 0)
    w = h = N
    c = []
    count = 0
    for i in range(x-w/2,x+w/2+1):
        for j in range(y-h/2,y+h/2+1):
            if i<0 or j<0 or i > 255 or j > 255:
                continue
            else:
                c.append(image[i,j])
                count += 1
    c.sort()
    if count % 2 ==0:
        return (c[(count-1)/2] + c[count/2])/2
    else:
        return c[count/2]

def get_percent(image,x,y,N,percent):
    w = h = N
    c = []
    count = 0
    for i in range(x-w/2,x+w/2+1):
        for j in range(y-h/2,y+h/2+1):
            if i < 0 or j < 0 or i > 255 or j > 255:
                continue
            else:
                c.append(image[i,j])
                count += 1
    c.sort()
    p = int(count * percent)
    if p == count:
        p = p -1
    return c[p]

def get_min_max_nearest(image,x,y,N):
    assert (N > 0)
    origin = image[x,y]
    w = h = N
    c = []
    count = 0
    for i in range(x-w/2,x+w/2+1):
        for j in range(y-h/2,y+h/2+1):
            if i<0 or j<0 or i > 255 or j > 255:
                continue
            else:
                c.append(image[i,j])
                count += 1
    c.sort()
    if np.abs(origin - c[0]) >= np.abs(origin -c[count-1]):
        return c[0]
    else:
        return c[count-1]

def template_conv(image,template):
    height,width = image.shape
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
    height,width = image.shape
    newdata = np.zeros(image.shape)
    for i in range(width):
        for j in range(height):
            newdata[i,j],count = conv(image,i,j,template)
            if flag == 1:
                newdata[i,j] /= (count*1.0)

            if newdata[i,j] < 0:
                newdata[i,j] = 0
            if newdata[i,j] > 255:
                newdata[i,j] = 255

    return newdata

#high-frequency emphasis filtering
# A : coefficient
def hfe_filter(image,A):
    #gauss smooth
    gauss = np.array([1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1]).reshape(5,5)
    smoothimg = linear_filter(image,gauss,SMOOTH)

    #unsharp mask
    um = image - smoothimg

    #high-frequency emphasis
    newdata = [x*A for x in um] + image
    return newdata

def template_conv_test():
    data = get_image_data("../pic/lena.jpg")

    sobel = np.array([-2,-1,0,-1,0,1,0,1,2]).reshape(3,3)
    newdata = template_conv(data,sobel)

    plt.subplot(1,3,1)
    plt.title("origin")
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1,3,2)
    plt.title("sobel")
    plt.imshow(sobel, cmap=plt.get_cmap('gray'))

    plt.subplot(1,3,3)
    plt.title("last")
    plt.imshow(newdata, cmap=plt.get_cmap('gray'))
    plt.show()


#template size: N*N
def median_filter(image, N):
    height,width = image.shape
    newdata = np.zeros(image.shape)

    for i in range(width):
        for j in range(height):
            newdata[i,j] = get_median(image,i,j,N)

    return newdata

def percent_filter(image,N, percent):
    width,height = image.shape
    newdata = np.zeros(image.shape)

    for i in range(width):
        for j in range(height):
            newdata[i,j] = get_percent(image,i,j,N,percent)

    return newdata


def min_max_sharpen_filter(image,N):
    height,width = image.shape
    newdata = np.zeros(image.shape)

    for i in range(width):
        for j in range(height):
            newdata[i,j] = get_min_max_nearest(image,i,j,N)

    return newdata


#hybrid filter:
#   step 1,linear filter: gauss(or others)
#   step 2,median
def hybrid_filter(image):
    #1,linear filter: gauss
    gauss = np.array([1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1]).reshape(5,5)
    gaussimg = linear_filter(image,gauss,1)

    #2,median
    lastimg = median_filter(gaussimg,5)

    return lastimg

def hist_local_enhance(image,N):
    height,width = image.shape
    newdata = np.zeros(image.shape)
    nw = width / N
    nh = height / N
    for i in range(nw):
        for j in range(nh):
            sub = hist_enhance(image[i*N:i*N+(N-1), j*N:j*N+(N-1)])
            newdata[i*N:i*N+(N-1), j*N:j*N+(N-1)] = sub

    return newdata

#k: mean coefficient
#l: variance coefficient
def mean_variance_local_enhance(image,k,l,E):
    height,width = image.shape
    newdata = np.zeros(image.shape)

    M,S = get_mean_variance(image.reshape(1,width*height).tolist()[0])
    print "M,S:",M,S

    for i in range(width):
        for j in range(height):
            nb = get_neighbor_data(image,i,j)
            mean,var = get_mean_variance(nb)
            if k*mean >= M and var <= l*S:
                newdata[i,j] = E*image[i,j]
                if newdata[i,j] > 255:
                    newdata[i,j] = 255
            else:
                newdata[i,j] = image[i,j]

    return newdata



def linear_filter_test():
    data = get_image_data("../pic/lena.jpg")

    #weighted mean
    sobel = np.array([1,2,1,2,4,2,1,2,1]).reshape(3,3)
    lastmean = linear_filter(data,sobel,SMOOTH)

    #gauss
    gauss = np.array([1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1]).reshape(5,5)
    lastgauss = linear_filter(data,gauss,SMOOTH)

    #laplace
    laplace = np.array([0,-1,0,-1,4,-1,0,-1,0]).reshape(3,3)
    lastlap = linear_filter(data,laplace,SHARPEN)

    #hfe high-frequency emphasis filtering
    hfe = hfe_filter(data,3)

    plt.subplot(2,3,1)
    plt.title("origin")
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(2,3,2)
    plt.title("mean_filter")
    plt.imshow(lastmean, cmap=plt.get_cmap('gray'))

    plt.subplot(2,3,3)
    plt.title("gauss-filter")
    plt.imshow(lastgauss, cmap=plt.get_cmap('gray'))

    plt.subplot(2,3,4)
    plt.title("laplace-filter")
    plt.imshow(lastlap, cmap=plt.get_cmap('gray'))

    plt.subplot(2,3,5)
    plt.title("hfe")
    plt.imshow(hfe, cmap=plt.get_cmap('gray'))

    plt.show()


def nonlinear_filter_test():
    data = get_image_data("../pic/lena.jpg")

    #median
    median = median_filter(data, 3)

    #percent
    percent = percent_filter(data, 3,0.75)

    plt.subplot(1,3,1)
    plt.title("origin")
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1,3,2)
    plt.title("median-filter")
    plt.imshow(median, cmap=plt.get_cmap('gray'))

    plt.subplot(1,3,3)
    plt.title("percent-filter")
    plt.imshow(percent, cmap=plt.get_cmap('gray'))

    plt.show()


def nonlinear_sharpen_test():
    data = get_image_data("../pic/lena.jpg")

    #min-max
    mm = min_max_sharpen_filter(data,3)

    plt.subplot(1,2,1)
    plt.title("origin")
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1,2,2)
    plt.title("min-mix-sharpen")
    plt.imshow(mm, cmap=plt.get_cmap('gray'))

    plt.show()


def hybrid_filter_test():
    data = get_image_data("../pic/lena.jpg")

    #hybrid
    hybrid = hybrid_filter(data)

    plt.subplot(1,2,1)
    plt.title("origin")
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1,2,2)
    plt.title("hybrid-filter")
    plt.imshow(hybrid, cmap=plt.get_cmap('gray'))

    plt.show()


def local_enhance_test():
    data = get_image_data("../pic/lena.jpg")
    print data

    #hist local enhance
    hle = hist_local_enhance(data,64)

    #mean variance local enhance
    mvle = mean_variance_local_enhance(data,1.5,1.5,2)
    print mvle

    plt.subplot(1,3,1)
    plt.title("origin")
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1,3,2)
    plt.title("hist-local-enhance")
    plt.imshow(hle, cmap=plt.get_cmap('gray'))

    plt.subplot(1,3,3)
    plt.title("mean-variance-local-enhance")
    plt.imshow(mvle, cmap=plt.get_cmap('gray'))

    plt.show()


if __name__ == '__main__':
    # template_conv_test()
    # linear_filter_test()
    # nonlinear_filter_test()
    # nonlinear_sharpen_test()
    # hybrid_filter_test()
     local_enhance_test()
