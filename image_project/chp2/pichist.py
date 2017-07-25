'''
    hist enhance
'''
from PIL import Image
import numpy as np


def get_image_data(filename):
    im = Image.open(filename)
    data = im.getdata()
    data = np.array(data,'float').reshape(im.size)
    return data

def get_counts(img):
    counts = {}
    width,height = img.shape
    for i in range(width):
        for j in range(height):
            counts.setdefault(img[i,j],0)
            counts[img[i,j]] += 1
    return counts

def getcdf(a):
    return np.cumsum(a)/(np.sum(a)*1.0)

def hist_enhance(image):
    width,height = image.shape
    counts = get_counts(image)
    keys = counts.keys()
    cdf = getcdf(counts.values())
    size =  len(keys)-1
    dict = {}
    for i in keys:
        dict[i] = int(size * cdf[i] + 0.5)
    newdata = np.zeros(image.shape)
    for i in range(width):
        for j in range(height):
            newdata[i,j] = dict[image[i,j]] * keys[-1]

    print newdata

#dest: spec dict
def hist_spec_SML(image,dest):
    counts = get_counts(image)
    keys = counts.keys()
    size = len(keys)-1

    destkeys = np.copy(keys)
    destvalues = []
    for k in destkeys:
        if  k in dest:
            destvalues[k] = dest[k]
        else:
            destvalues[k] = 0

    srccdf = getcdf(counts.values())
    destcdf = getcdf(destvalues)

    srcmin = np.array([0.0]*size**2).reshape(size,size)
    for x in range(size):
        for y in range(size):
            srcmin[y,x] = srccdf[x] - destcdf[y]

    specmap = {}
    for x in range(size):
        miny = 0
        minv = srcmin[0,x]
        for y in range(1,size):
            if minv > srcmin[y,x]:
                minv = srcmin[y,x]
                miny = y
        specmap[y] = miny

    newdata = np.zeros(image.shape)
    width,height = image.shape
    for i in range(width):
        for j in range(height):
            newdata[i,j] = specmap[image[i,j]]

    return newdata

def hist_spec_GML(image,dest):
    counts = get_counts(image)
    keys = counts.keys()
    size = len(keys)-1

    destkeys = np.copy(keys)
    destvalues = []
    for k in destkeys:
        if  k in dest:
            destvalues[k] = dest[k]
        else:
            destvalues[k] = 0

    srccdf = getcdf(counts.values())
    destcdf = getcdf(destvalues)

    srcmin = np.array([0.0]*size**2).reshape(size,size)
    for x in range(size):
        for y in range(size):
            srcmin[y,x] = srccdf[x] - destcdf[y]

    specmap = {}
    laststarty = 0
    lastendy = 0
    starty = 0
    endy = 0
    for x in range(size):
        minv = srcmin[x,0]
        for y in range(1,size):
            if minv > srcmin[x,y]:
                minv = srcmin[x,y]
                endy = y

        if starty != laststarty or endy != lastendy:
            for i in range(starty,endy):
                specmap[i] = x
            laststarty = starty
            lastendy = endy
            starty = lastendy + 1

    newdata = np.zeros(image.shape)
    width,height = image.shape
    for i in range(width):
        for j in range(height):
            newdata[i,j] = specmap[image[i,j]]

    return newdata


def hist_spec_SML_test():
    data = get_image_data("../pic/lena.jpg")
    Image.fromarray(data.astype(np.uint8)).show("origin")

    spec = {80:0.2,160:0.6,240:0.2}
    newdata = hist_spec_SML(data,spec)
    Image.fromarray(newdata.astype(np.uint8)).show("last")

def hist_spec_GML_test():
    data = get_image_data("../pic/lena.jpg")
    Image.fromarray(data.astype(np.uint8)).show("origin")

    spec = {80:0.2,160:0.6,240:0.2}
    newdata = hist_spec_GML(data,spec)
    Image.fromarray(newdata.astype(np.uint8)).show("last")


def hist_enhance_test():
    data = get_image_data("../pic/lena.jpg")
    Image.fromarray(data.astype(np.uint8)).show("origin")


    newdata = hist_enhance(data)
    Image.fromarray(newdata.astype(np.uint8)).show("last")



if __name__ == '__main__':
    hist_enhance_test()
    hist_spec_SML_test()
    hist_spec_GML_test()

