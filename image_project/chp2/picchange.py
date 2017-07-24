'''
    picture change: trnasfer ,scaling ,rotate,cut
'''

from PIL import Image
import numpy as np


def get_image_data(filename):
    im = Image.open(filename)
    print im.size
    data = im.getdata()
    data = np.array(data,'float').reshape(im.size)
    return data

def coordinate_transfer(xorigin,yorigin,transmatrix):
    newcord = np.asarray(np.dot(transmatrix, (np.matrix([xorigin, yorigin, 1]).T)))
    return int(newcord[0]),int(newcord[1])

def coordinate_scaling(xorigin,yorigin,scalmatrix):
    newcord = np.asarray(np.dot(scalmatrix, (np.matrix([xorigin, yorigin, 1]).T)))
    return int(newcord[0]),int(newcord[1])

def coordinate_roate(xorigin,yorigin,roatematrix):
    newcord = np.asarray(np.dot(roatematrix, (np.matrix([xorigin, yorigin, 1]).T)))
    return int(newcord[0]),int(newcord[1])

def coordinate_cut(xorigin,yorigin,cutmatrix):
    newcord = np.asarray(np.dot(cutmatrix, (np.matrix([xorigin, yorigin, 1]).T)))
    return int(newcord[0]),int(newcord[1])



def tranfser_test():
    data = get_image_data("../pic/lena.jpg")
    Image.fromarray(data.astype(np.uint8)).show("origin")

    new_data = np.copy(data)

    #transfer 5: x->x+5,y->y+5
    tm = np.matrix('1 0 5; 0 1 5; 0 0 1')
    print tm

    #x: 50-80,y:50-80
    for i in range(50,81):
        for j in range(50,81):
            x, y = coordinate_transfer(i, j, tm)
            new_data[x,y] = data[i,j]

    Image.fromarray(new_data.astype(np.uint8)).show("after")

def scaling_test():
    data = get_image_data("../pic/lena.jpg")
    Image.fromarray(data.astype(np.uint8)).show("origin")

    width,height = data.shape


    #scaling 2: x->2*x,y->y*2
    new_data = np.zeros((2*width,2*height))
    print new_data.shape
    sm = np.matrix('2 0 1; 0 2 1; 0 0 1')
    print sm

    for i in range(width):
        for j in range(height):
            x, y = coordinate_scaling(i, j, sm)
            print x,y
            new_data[x,y] = data[i,j]

    Image.fromarray(new_data.astype(np.uint8)).show("after")

def rotate_test():
    data = get_image_data("../pic/lena.jpg")
    Image.fromarray(data.astype(np.uint8)).show("origin")

    width,height = data.shape

    new_data = np.copy(data)


    #rotate: pi/6, range: 101-156,101-156
    rm = np.matrix('0.86 0.5 0; -0.5 0.86 0; 0 0 1')
    print rm

    for i in range(101,157):
        for j in range(101,157):
            x, y = coordinate_roate(i, j, rm)
            print x,y
            new_data[x,y] = data[i,j]

    Image.fromarray(new_data.astype(np.uint8)).show("after")

def cut_test():
    data = get_image_data("../pic/lena.jpg")
    Image.fromarray(data.astype(np.uint8)).show("origin")

    width,height = data.shape

    #cut: x
    new_data = np.zeros(data.shape)
    #cm = np.matrix('1 0.5 0; 0 1 0; 0 0 1')
    cm = np.matrix('1 0 0; 0.5 1 0; 0 0 1')
    print cm

    for i in range(width):
        for j in range(height):
            x, y = coordinate_cut(i, j, cm)
            if x>=width or y>=height:
                continue
            new_data[x,y] = data[i,j]

    Image.fromarray(new_data.astype(np.uint8)).show("after")


if __name__ == '__main__':
    #tranfser_test()
    #scaling_test()
    #rotate_test()
    cut_test()

