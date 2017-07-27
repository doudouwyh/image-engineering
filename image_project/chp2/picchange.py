'''
    picture change: trnasfer ,scaling ,rotate,cut
'''

import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *

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

    new_data = np.copy(data)

    #transfer 50: x->x+50,y->y+50
    tm = np.matrix('1 0 50; 0 1 50; 0 0 1')

    #origin area:
    #x: 50-80,y:50-80
    for i in range(50,81):
        for j in range(50,81):
            x, y = coordinate_transfer(i, j, tm)
            new_data[x,y] = data[i,j]

    plt.subplot(1,2,1)
    plt.title("origin")
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1,2,2)
    plt.title("transfer")
    plt.imshow(new_data, cmap=plt.get_cmap('gray'))
    plt.show()


def scaling_test():
    data = get_image_data("../pic/lena.jpg")

    width,height = data.shape

    #scaling 2: x->2*x,y->y*2
    new_data = np.zeros((2*width,2*height))
    sm = np.matrix('2 0 1; 0 2 1; 0 0 1')

    for i in range(width):
        for j in range(height):
            x, y = coordinate_scaling(i, j, sm)
            new_data[x,y] = data[i,j]

    plt.subplot(1,2,1)
    plt.title("origin")
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1,2,2)
    plt.title("scaling")
    plt.imshow(new_data, cmap=plt.get_cmap('gray'))
    plt.show()

def rotate_test():
    data = get_image_data("../pic/lena.jpg")

    new_data = np.copy(data)

    #rotate: pi/6, range: 101-156,101-156
    rm = np.matrix('0.86 0.5 0; -0.5 0.86 0; 0 0 1')

    for i in range(101,157):
        for j in range(101,157):
            x, y = coordinate_roate(i, j, rm)
            new_data[x,y] = data[i,j]

    plt.subplot(1,2,1)
    plt.title("origin")
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1,2,2)
    plt.title("rotate")
    plt.imshow(new_data, cmap=plt.get_cmap('gray'))
    plt.show()

def cut_test():
    data = get_image_data("../pic/lena.jpg")

    width,height = data.shape

    new_data_x = np.zeros(data.shape)
    new_data_y = np.zeros(data.shape)
    cmx = np.matrix('1 0 0; 0.5 1 0; 0 0 1')
    cmy = np.matrix('1 0.5 0; 0 1 0; 0 0 1')

    for i in range(width):
        for j in range(height):
            x, y = coordinate_cut(i, j, cmx)
            if x < width and y < height:
                new_data_x[x, y] = data[i, j]

            x, y = coordinate_cut(i, j, cmy)
            if x < width and y < height:
                new_data_y[x, y] = data[i, j]


    plt.subplot(1,3,1)
    plt.title("origin")
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    plt.subplot(1,3,2)
    plt.title("cut_x")
    plt.imshow(new_data_x, cmap=plt.get_cmap('gray'))

    plt.subplot(1,3,3)
    plt.title("cut_y")
    plt.imshow(new_data_y, cmap=plt.get_cmap('gray'))
    plt.show()


if __name__ == '__main__':
    #tranfser_test()
    #scaling_test()
    #rotate_test()
     cut_test()

