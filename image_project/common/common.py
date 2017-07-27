'''
    common
'''

from PIL import Image
import numpy as np

#get gray image
def get_image_data(filename):
    im = Image.open(filename).convert('L')
    data = im.getdata()
    data = np.array(data,'float').reshape(im.size)
    return data

#8-neighbor
def get_neighbor_data(image,x,y):
    nb = []
    for i in range(x-1,x+2):
        for j in range(y-1,y+2):
            if i == x and j == y:
                continue
            if i < 0 or j < 0 or i > 255 or j > 255:
                nb.append(image[x,y])   #append itsself
            else:
                nb.append(image[i,j])
    return nb

def get_mean_variance(a):
    mean = np.sum(a)/len(a)
    var = 0
    for i in range(len(a)):
        var += (a[i] - mean) ** 2

    return mean,var