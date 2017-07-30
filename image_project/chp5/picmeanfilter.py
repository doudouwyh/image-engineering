'''
    mean filter
'''

import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *

def arithmetic_mean_filter_test():

    data = get_image_data("../pic/lena.jpg")
    h,w = data.shape

    N = 3
    meandata = np.zeros(data.shape)
    for i in range(h):
        for j in range(w):
            nb,n = get_template_cover_data(data,i,j,N)
            meandata[i,j] = np.sum(nb)/n

    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 2, 2)
    plt.title('arithmeticmeanfilter')
    plt.imshow(meandata, cmap=plt.get_cmap('gray'))

    plt.show()

def geometric_mean_filter_test():

    data = get_image_data("../pic/lena.jpg")
    h,w = data.shape

    N = 3
    meandata = np.zeros(data.shape)
    for i in range(h):
        for j in range(w):
            nb,n = get_template_cover_data2(data,i,j,N)
            meandata[i,j] = (np.cumprod(nb)[-1])**(1.0/n)

    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 2, 2)
    plt.title('geometricmeanfilter')
    plt.imshow(meandata, cmap=plt.get_cmap('gray'))

    plt.show()

def harmonic_mean_filter_test():

    data = get_image_data("../pic/lena.jpg")
    h,w = data.shape

    N = 3
    meandata = np.zeros(data.shape)
    for i in range(h):
        for j in range(w):
            nb,n = get_template_cover_data(data,i,j,N)
            sum = np.sum([1.0/x for x in nb if x != 0])
            meandata[i,j] = (n*1.0)/sum

    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 2, 2)
    plt.title('harmonicmeanfilter')
    plt.imshow(meandata, cmap=plt.get_cmap('gray'))

    plt.show()

def inverse_harmonic_mean_filter_test():

    data = get_image_data("../pic/lena.jpg")
    h,w = data.shape

    N = 3
    k  = -1
    meandata = np.zeros(data.shape)
    for i in range(h):
        for j in range(w):
            nb,n = get_template_cover_data(data,i,j,N)
            sum1 = np.sum([x**(k+1) for x in nb if x != 0])
            sum2 = np.sum([x**(k) for x in nb if x != 0])
            meandata[i,j] = sum1*1.0/sum2

    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 2, 2)
    plt.title('inverseharmonicmeanfilter')
    plt.imshow(meandata, cmap=plt.get_cmap('gray'))

    plt.show()

#h(x) = ln(x)
def nonlinear_mean_filter_test():

    data = get_image_data("../pic/lena.jpg")
    h,w = data.shape

    Wi = 1
    N  = 3
    meandata = np.zeros(data.shape)
    for i in range(h):
        for j in range(w):
            nb, n = get_template_cover_data2(data, i, j, N)
            sum = np.sum([Wi*np.log(x) for x in nb]) * 1.0
            meandata[i,j] = np.exp(sum/(Wi*n))

    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 2, 2)
    plt.title('nonlinearmeanfilter')
    plt.imshow(meandata, cmap=plt.get_cmap('gray'))

    plt.show()

def cut_mean_filter_test():

    data = get_image_data("../pic/lena.jpg")
    h,w = data.shape

    d = 4
    N  = 3
    meandata = np.zeros(data.shape)
    for i in range(h):
        for j in range(w):
            nb, n = get_template_cover_data(data, i, j, N)
            if n <= d:
                meandata[i,j] = np.sum(nb)*1.0/n
            else:
                nb.sort()
                meandata[i,j] = np.sum(nb[d/2:-d/2])*1.0/(n-d)

    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 2, 2)
    plt.title('cutmeanfilter')
    plt.imshow(meandata, cmap=plt.get_cmap('gray'))

    plt.show()

def adaptive_median_filter_test():

    data = get_image_data("../pic/lena.jpg")
    h,w = data.shape

    Smax = 7
    S = np.arange(3,9,2)
    meandata = np.zeros(data.shape)
    for i in range(h):
        for j in range(w):
            for N in S:
                if N > Smax:
                    break
                nb, n = get_template_cover_data(data, i, j, N)
                nb.sort()
                gmin = nb[0]
                gmax = nb[n-1]
                gmed = nb[n/2]
                A1 = gmed - gmin
                A2 = gmed - gmax
                if A1 >0 and A2 < 0:
                    B1 = data[i,j] - gmin
                    B2 = data[i,j] - gmax
                    if B1 > 0  and B2 < 0:
                        meandata[i,j] =  data[i,j]
                    else:
                        meandata[i,j] =  gmed
                    break
                else:
                        continue

    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 2, 2)
    plt.title('cadaptivemedianfilter')
    plt.imshow(meandata, cmap=plt.get_cmap('gray'))

    plt.show()

if __name__ == '__main__':
    arithmetic_mean_filter_test()
    geometric_mean_filter_test()
    harmonic_mean_filter_test()
    inverse_harmonic_mean_filter_test()
    nonlinear_mean_filter_test()
    cut_mean_filter_test()
    adaptive_median_filter_test();

