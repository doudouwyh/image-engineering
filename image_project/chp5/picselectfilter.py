'''
    select filter
'''

import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *

def select_filter_test():

    data = get_image_data("../pic/lena.jpg")
    h,w = data.shape

    Tv = 20
    Tn = 0.2
    meandata = np.zeros(data.shape)
    for i in range(h):
        for j in range(w):
            nb = get_neighbor_data(data,i,j)
            assert(len(nb) == 8)
            diff = [x for x in nb if np.abs(x-data[i,j]) > Tv]
            if  len(diff)*1.0/len(nb) > Tn:
                meandata[i,j] = np.sort(nb)[len(nb)/2 - 1] #median
            else:
                meandata[i,j] = np.sum(nb)*1.0 /len(nb)   #mean

    plt.subplot(1, 2, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1, 2, 2)
    plt.title('selectfilter')
    plt.imshow(meandata, cmap=plt.get_cmap('gray'))

    plt.show()

if __name__ == '__main__':
    select_filter_test()

