'''
    Golomb
    Huffman
'''

import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from common.common import *

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

def  huffman_encode():
    pass

def huffman_decode():
    pass

def Shannon_Fano_encode():
    pass

def Shannon_Fano_decode():
    pass

def arithmatic_encode():
    pass

def arithmatic_decode():
    pass

#bit-plane encoding
def bpe():
    pass

if __name__ == '__main__':
    golomb_encode_test()