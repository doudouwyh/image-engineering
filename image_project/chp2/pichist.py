'''
    hist enhance
'''
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *

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
    height,width = image.shape
    counts = get_counts(image)
    keys = counts.keys()
    cdf = getcdf(counts.values())
    size =  len(keys)-1
    dict = {}
    for i in range(len(keys)):
        dict[keys[i]] = int(size * cdf[i] + 0.5)
    newdata = np.zeros(image.shape)
    for i in range(width):
        for j in range(height):
            newdata[i,j] = dict[image[i,j]] * keys[-1]

    return newdata

#dest: spec dict
def hist_spec_SML(image,dest):
    counts = get_counts(image)
    keys = counts.keys()
    size = len(keys)

    #pdf
    destkeys = np.copy(keys)
    destvalues = np.zeros(len(keys))
    for k in range(len(destkeys)):
        if  destkeys[k] in dest:
            destvalues[k] = dest[destkeys[k]]

    #cdf
    srccdf = getcdf(counts.values())
    destcdf = getcdf(destvalues)

    srcmin = np.array([0.0]*size**2).reshape(size,size)
    for x in range(size):
        for y in range(size):
            srcmin[y,x] = srccdf[x] - destcdf[y]

    specmap = {}
    for x in range(size):
        miny = 0
        minv = abs(srcmin[0,x])
        for y in range(1,size):
            if minv > abs(srcmin[y,x]):
                minv = abs(srcmin[y,x])
                miny = y

        specmap[keys[x]] = keys[miny]

    newdata = np.zeros(image.shape)
    height,width = image.shape
    for i in range(width):
        for j in range(height):
            newdata[i,j] = specmap[int(image[i,j])]

    return newdata

def hist_spec_GML(image,dest):
    counts = get_counts(image)
    keys = counts.keys()
    sizesrc = len(keys)

    destkeys = dest.keys()
    sizedest   =  len(destkeys)

    #cdf
    srccdf = getcdf(counts.values())
    destcdf = getcdf(dest.values())

    srcmin = np.array([0.0]*sizesrc*sizedest).reshape(sizesrc,sizedest)
    for x in range(sizedest):
        for y in range(sizesrc):
            srcmin[y,x] = destcdf[x] - srccdf[y]

    specmap = {}
    starty = 0
    endy  = 0
    for x in range(sizedest):
        minv = abs(srcmin[0,x])
        for y in range(1,sizesrc):
            if  minv > abs(srcmin[y,x]):
                minv = abs(srcmin[y,x])
                endy = y

        for i in range(starty,endy+1):
            specmap[keys[i]] = destkeys[x]
        starty = endy+1

    newdata = np.zeros(image.shape)
    height,width = image.shape
    for i in range(width):
        for j in range(height):
            newdata[i,j] = specmap[int(image[i,j])]

    return newdata

def hist_spec_test():
    data = get_image_data("../pic/lena.jpg")

    spec = {80:0.2,160:0.6,251:0.2}
    smldata = hist_spec_SML(data,spec)
    gmldata = hist_spec_GML(data,spec)

    plt.subplot(1,3,1)
    plt.title("origin")
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1,3,2)
    plt.title("SML")
    plt.imshow(smldata, cmap=plt.get_cmap('gray'))

    plt.subplot(1,3,3)
    plt.title("GML")
    plt.imshow(gmldata, cmap=plt.get_cmap('gray'))

    plt.show()

def hist_enhance_test():
    data = get_image_data("../pic/lena.jpg")
    newdata = hist_enhance(data)

    plt.subplot(1,2,1)
    plt.title("origin")
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1,2,2)
    plt.title("histenhance")
    plt.imshow(newdata, cmap=plt.get_cmap('gray'))
    plt.show()


if __name__ == '__main__':
    # hist_enhance_test()
     hist_spec_test()


