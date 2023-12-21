import numpy as np
from os import urandom
import copy

plain_bits = 128
key_bits = 256
word_size = 32



rotations = [16,12,8,7]

def lrot(a, r, word_size = 32):
    return (a<<r) | (a>>(word_size-r))


def quarter_round(a1, b1, c1, d1, word_size = 32):
    a = a1 + b1
    d = lrot(d1 ^ a, rotations[0])
    c = c1 + d
    b = lrot(b1 ^ c, rotations[1])

    a += b
    d = lrot(d^a, rotations[2])
    c += d
    b = lrot(b ^ c, rotations[3])
    return a, b, c, d

def encrypt(p, k, r, add_with_X0 = False):
    k1, k2, k3, k4, k5, k6, k7, k8 = convert_from_binary(k).transpose()
    p1, p2, n1, n2 = convert_from_binary(p).transpose()

    # Constants
    c1 = np.repeat(np.uint32([1634760805]), len(k1))
    c2 = np.repeat(np.uint32([857760878]), len(k1))
    c3 = np.repeat(np.uint32([2036477234]), len(k1))
    c4 = np.repeat(np.uint32([1797285236]), len(k1))

    X = np.array([c1, c2,c3,c4, k1,k2,k3,k4,k5,k6,k7,k8,p1,p2,n1,n2], dtype = np.uint32)

    for rr in range(r):
        if rr%2 == 0:
            X[0], X[4], X[8], X[12] = quarter_round(X[0], X[4], X[8], X[12])
            X[1], X[5], X[9], X[13] = quarter_round(X[1], X[5], X[9], X[13])
            X[2], X[6], X[10], X[14] = quarter_round(X[2], X[6], X[10], X[14])
            X[3], X[7], X[11], X[15] = quarter_round(X[3], X[7], X[11], X[15])
        else:
            X[0], X[5], X[10], X[15] = quarter_round(X[0], X[5], X[10], X[15])
            X[1], X[6], X[11], X[12] = quarter_round(X[1], X[6], X[11], X[12])
            X[2], X[7], X[8], X[13] = quarter_round(X[2], X[7], X[8], X[13])
            X[3], X[4], X[9], X[14] = quarter_round(X[3], X[4], X[9], X[14])

    if add_with_X0:
        return convert_to_binary(X + np.array([c1,c2,c3,c4,k1,k2,k3,k4,k5,k6,k7,k8,p1,p2,n1,n2], dtype = np.uint32))
    return convert_to_binary(X)






def convert_from_binary(arr, _dtype=np.uint32):
    return np.packbits(arr, axis = 1).view(dtype = ">i4").astype(np.uint32)

def convert_to_binary(arr, word_size = 32):
    X = np.unpackbits(arr.transpose().copy().astype(dtype=">i4").view(np.uint8), axis = 1)
    return X




def check_testvector():

    k = np.array([[0x03020100], [0x07060504], [0x0b0a0908], [0x0f0e0d0c], [0x13121110], [0x17161514], [0x1b1a1918],                  [0x1f1e1d1c]], dtype=np.uint32).reshape(8,1).repeat(10, axis = 1)
    p = np.array([[0x00000001], [0x09000000], [0x4a000000], [0x00000000]], dtype=np.uint32).reshape(4,1).repeat(10, axis = 1)
    stream = np.array([
           [0xe4e7f110], [0x15593bd1], [0x1fdd0f50], [0xc47120a3],
           [0xc7f4d1c7], [0x0368c033], [0x9aaa2204], [0x4e6cd4c3],
           [0x466482d2], [0x09aa9f07], [0x05d7c214], [0xa2028bd9],
           [0xd19c12b5], [0xb94e16de], [0xe883d0cb], [0x4e3c50a2]
       ], dtype=np.uint32).flatten()
    key = convert_to_binary(k)
    p = convert_to_binary(p)
    expected = [hex(x) for x in stream]
    enc = encrypt(p, key, 20, add_with_X0 = True)
    for x in enc:
        produced = [hex(a) for a in convert_from_binary(x.reshape(1, 512))[0]]
        assert expected == produced


check_testvector()

