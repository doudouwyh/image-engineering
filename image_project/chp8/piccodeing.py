'''
    Golomb
    Huffman
    Shannon-Fano
'''
import operator
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *

class Node:
    def __init__(self,freq):
        self.left = None
        self.right = None
        self.father = None
        self.freq = freq

    def isleft(self):
        return self.father.left == self


def createnodes(freqs):
    return [Node(freq) for freq in freqs]

def createHuffmanTree(nodes):
    queue = nodes[:]
    while len(queue) > 1:
        queue.sort(key=lambda item : item.freq)
        node_left = queue.pop(0)
        node_right = queue.pop(0)
        node_father = Node(node_left.freq + node_right.freq)
        node_father.left = node_left
        node_father.right = node_right
        node_left.father = node_father
        node_right.father = node_father
        queue.append(node_father)

    queue[0].father = None
    return queue[0]


def huffman_encode(nodes,root):
    codes = [''] * len(nodes)
    for i in range(len(nodes)):
        node_tmp = nodes[i]
        while node_tmp != root:
            if node_tmp.isleft():
                codes[i] = '0' + codes[i]
            else:
                codes[i] = '1' + codes[i]
            node_tmp = node_tmp.father
    return codes


def golomb_encode(n,m):
    assert (isinstance(n,int))
    assert (isinstance(m,int))

    I = int(np.floor(n/m))
    onecode = '1'*I+'0'
    k = int(np.ceil(np.log2(m)))
    c = 2**k - m
    r  = n % m
    l = len(bin(n)) -2
    second = ''
    if   0 <= r < c:
        s = '0'*l
        br = bin(int(r))[2:]
        tmp = s[:-len(br)] + br
        print tmp
        if k-1 > 0:
            if k - 1 > l:
                s = '0' * (k - 1)
                tmp = s[-(k-1):-l] + br
            second = tmp[-(k-1):]
    else:
        s = '0' * l
        brc = bin(int(r+c))[2:]
        tmp = s[:-len(brc)] + brc
        if k > 0:
            if k > l:
                s = '0' * k
                tmp = s[-(k):-l] + brc
            second = tmp[-k:]

    return onecode+second


def  golomb_encode_test():
    for m in [1,2,4]:
        print "m=",m
        for n in range(10):
            print "n=",n,golomb_encode(n,m)
        print ""


def huffman_encode_test():
    data = get_image_data("../pic/lena.jpg")
    h,w = data.shape
    dt = {}
    for i in range(h):
        for j in range(w):
            dt.setdefault(data[i,j],0)
            dt[data[i,j]] += 1

    data_freqs = dt.items()

    nodes = createnodes([item[1] for item in data_freqs])
    root = createHuffmanTree(nodes)
    codes = huffman_encode(nodes,root)
    dtcodes = {}
    for item in zip(data_freqs,codes):
        dtcodes[item[0][0]] = item[1]
    print dtcodes


class SFNode:
    def __init__(self,mf):
        self.left = None
        self.right = None
        self.father = None
        self.meanfreq = mf


def create_sftree(father,data,direct):
    if len(data) == 0:
        return

    if len(data) == 1:
        node = SFNode(data[0][1])
        node.father = father
        if direct == 'L':
            father.left = node
        else:
            father.right = node
        return

    sumfreq = sum([x[1] for x in data])

    node = SFNode(0.5 * sumfreq)
    node.father = father

    if direct == 'L':
        father.left = node
    else:
        father.right = node

    partsum = 0
    s = 0
    while partsum < 0.5 * sumfreq:
        partsum += data[s][1]
        s += 1

    left = data[:s]
    right = data[s:]

    if len(left) > 0:
        if len(left) == len(data):
            node = SFNode(0.5 * sumfreq)
            father.left = node
        else:
            create_sftree(node,left,'L')

    if len(right) > 0:
        create_sftree(node,right,'R')


def search_sftree(root, value):
        t = root
        out = ''
        while (t.left is not None) or (t.right is not None):
                if value >= t.meanfreq:
                    out += '1'
                    t = t.left
                else:
                    out += '0'
                    t = t.right
        return out


def Shannon_Fano_encode_test():
    data = get_image_data("../pic/lena.jpg")
    h,w = data.shape
    dt = {}
    for i in range(h):
        for j in range(w):
            dt.setdefault(data[i,j],0)
            dt[data[i,j]] += 1

    sorteddt = sorted(dt.items(),key=operator.itemgetter(1),reverse=True)

    sumfreq = sum([x[1] for x in sorteddt])

    root = SFNode(0.5 * sumfreq)

    partsum = 0
    s = 0
    while partsum <= 0.5 * sumfreq:
        partsum += sorteddt[s][1]
        s += 1

    left = sorteddt[:s]
    right = sorteddt[s:]

    if len(left) > 0:
        if len(left) != len(data):
            create_sftree(root,left,'L')
    if len(right) > 0:
        create_sftree(root,right,'R')

    for item in sorteddt:
        code = search_sftree(root,item[1])
        print "data:", item[0], "freq:", item[1], "code:", code


###########################


def arithmatic_encode():
    pass

def arithmatic_encode_test():
    pass

#bit-plane encoding
def bit_plane_encode():
    pass

def bit_plane_encode_test():
    data = get_image_data("../pic/lena.jpg")
    h,w = data.shape
    bpes = []
    for k in range(8):
        bpes.append(np.zeros(data.shape))

    for i in range(h):
        for j in range(w):
            for k in range(8):
                bpes[k][i,j] = (int(data[i,j]) & (1 << k)) >> k

    print bpes[7]

    plt.subplot(1, 9, 1)
    plt.title('origin')
    plt.imshow(data, cmap=plt.get_cmap('gray'))

    for k in range(8):
        plt.subplot(1, 9, k+2)
        title = 'bep'+str(k)
        plt.title(title)
        plt.imshow(bpes[k], cmap=plt.get_cmap('gray'))

    plt.show()

if __name__ == '__main__':
    # golomb_encode_test()
    # huffman_encode_test()
    # bit_plane_encode_test()
    Shannon_Fano_encode_test()