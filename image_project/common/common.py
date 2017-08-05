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

#templatesize: N*N
def get_template_cover_data(image,x,y,N):
    c = []
    count = 0
    for i in range(x - N / 2, x + N / 2 + 1):
        for j in range(y - N / 2, y + N / 2 + 1):
            if i < 0 or i > 255 or j < 0 or j > 255:
                c.append(0)
            else:
                c.append(image[i,j])
                count += 1
    return c, count

# for multiply
#templatesize: N*N
def get_template_cover_data2(image,x,y,N):
    c = []
    count = 0
    for i in range(x - N / 2, x + N / 2 + 1):
        for j in range(y - N / 2, y + N / 2 + 1):
            if i < 0 or i > 255 or j < 0 or j > 255:
                c.append(1)
            else:
                c.append(image[i,j])
                count += 1
    return c, count

def get_mean_variance(a):
    mean = np.sum(a)/len(a)
    var = 0
    for i in range(len(a)):
        var += (a[i] - mean) ** 2

    return mean,var
