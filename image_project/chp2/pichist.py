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

def hist_enhance(image):
    pass


def hist_enhance_test():
    data = get_image_data("../pic/lena.jpg")
    # Image.fromarray(data.astype(np.uint8)).show("origin")
    # print data

    counts = get_counts(data)
    keys = counts.keys()
    cdf = counts.values()/(np.sum(counts.values())*1.0)
    size =  len(keys)-1
    ndatas = []
    for i in range(size):
        ndatas.append(int(size * cdf[i] + 0.5))

    newdata = np.array(ndatas).reshape(data.shape)

    print newdata




if __name__ == '__main__':
    hist_enhance_test()

