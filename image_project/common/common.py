'''
    common
'''

from PIL import Image
import numpy as np

def get_image_data(filename):
    im = Image.open(filename)
    data = im.getdata()
    data = np.array(data,'float').reshape(im.size)
    return data

#8-neighbor
def get_neighbor_data(image,x,y):
    nb = []
    for i in range(x-1,x+2):
        for j in range(y-1,y+2):
            if i < 0 or j < 0 or i == x or j == y:
                continue
            else:
                nb.append()
    return nb

def get_mean_variance(a):
    mean = np.sum(a)/len(a)
    var = 0
    for i in range(len(a)):
        var += (a[i] - mean) ** 2

    return mean,var