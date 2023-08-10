from os import urandom

import numpy as np
from typing import List

plain_bits = 384
key_bits = 8 # Temporary placeholder value until key size of 0 is handled in the optimizer. The key is never used.
word_size = 32

def rol(x,k):
    assert np.all(x.shape[1:] == (4, 32))
    return np.roll(x, -k, axis = 2)

def shift(x,k):
    assert np.all(x.shape[1:] == (4, 32))
    x = rol(x, k)
    x[:, :, -k:] = 0
    return x

def convert_to_bin(x, bits):
    return np.array([int(i) for i in bin(x)[2:].zfill(bits)], dtype=np.uint8).reshape(1, bits)

def test_vector():
    x = 0x000000009e3779ba3c6ef37adaa66d4678dde7241715611ab54cdb2e53845566f1bbcfc88ff34a5a2e2ac522cc624026
    X = convert_to_bin(x, 384)
    #print(X)
    expected = 0xba11c85a91bad119380ce880d24c2c683eceffea277a921c4f73a0bdda5a9cd884b673f034e52ff79e2bef49f41bb8d6
    C = encrypt(X, 0, 24)
    assert(np.all(encrypt(X, 0, 24) == convert_to_bin(expected, 384)))


def perm(state) :
    x = rol(state[:, 0, :], 24)
    y = rol(state[:, 1, :], 9)
    z= state[:,2,:]

    xy = shift(x&y, 3)
    xz = shift(x|z, 1)
    yz = shift(y&z, 2)
    zs = shift(z, 1)

    ox= z ^ y ^ xy
    oy= y ^ x ^ xz
    oz= x ^ zs ^ yz

    state[:,0,:]=ox
    state[:,1,:]=oy
    state[:,2,:]=oz
    return state


def small_swap(state):
    state[:, 0, [0, 1, 2, 3]] = state[:, 0, [1, 0, 3, 2]]
    return state


def big_swap(state):
    state[:, 0, [0, 1, 2, 3]] = state[:, 0, [2, 3, 0, 1]]
    return state




def encrypt(p, k, r) :
    state = p.copy().reshape(-1, 3, 4, 32)
    for _r in range(r,0,-1):
       perm(state)
       if _r%4==0:
         state=small_swap(state)
         state[:,0,0]=state[:,0,0]^convert_to_bin(0x9e377900^_r, word_size)
       if _r%4==2:
         state=big_swap(state)
    return state.reshape(-1, 384)







test_vector()


