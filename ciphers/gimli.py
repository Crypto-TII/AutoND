from os import urandom

import numpy as np
from typing import List

plain_bits = 384
key_bits = 8 # Temporary placeholder value until key size of 0 is handled in the optimizer. The key is never used.
word_size = 32

R = 24
S = 9

Sheet = List[int]
State = List[Sheet]

"""State are seen as  a 3 x 4 x 32 array """

MASK_VAL = 2 ** word_size - 1;



def rol(x,k):
    return(((x << k) & MASK_VAL) | (x >> (word_size - k)));

def ror(x,k):
    return((x >> k) | ((x << (word_size - k)) & MASK_VAL));

def test_vector():
    x = [(i * i * i + i * 0x9e3779b9) % (2 ** 32) for i in range(12)]
    x = [[x[4 * i + j] for i in range(3)] for j in range(4)]
    X=np.transpose(np.array(x))
    X=np.expand_dims(X, axis=0)

    Cb = encrypt(convert_to_binary(X), 0, 24)
    H = convert_from_binary(Cb)[0].flatten().tolist()


    assert (H[:4]==[0xba11c85a, 0x91bad119, 0x380ce880, 0xd24c2c68])


def perm(state) :
    """Apply the non linear permutation on the 96 sheet of the state"""
    x= state[:,0,:]
    y= state[:,1,:]
    z= state[:,2,:]
    x = rol(x,24)
    y=rol(y,9)

    ox=(z ^ y ^ ((x & y) << 3))&MASK_VAL
    oy= (y ^ x ^ ((x | z) << 1))&MASK_VAL
    oz= (x ^ (z << 1) ^ ((y & z) << 2))&MASK_VAL

    state[:,0,:]=ox
    state[:,1,:]=oy
    state[:,2,:]=oz
    return state


def small_swap(state):
    stmp=np.copy(state)
    stmp[:,0,0]=state[:,0,1]
    stmp[:,0,1]=state[:,0,0]
    stmp[:,0,2]=state[:,0,3]
    stmp[:,0,3]=state[:,0,2]
    return stmp


def big_swap(state):
    stmp=np.copy(state)
    stmp[:,0,0]=state[:,0,2]
    stmp[:,0,1]=state[:,0,3]
    stmp[:,0,2]=state[:,0,0]
    stmp[:,0,3]=state[:,0,1]
    return stmp


def print_to_hex(x):
    for i in range(3):
        print(hex(x[:,0,:].zfill(8)))
    print("----------------------")


def encrypt(p, k, r) :
    """ Number of rounds has still to be determined """

    state = convert_from_binary(p)
    for _r in range(r,0,-1):
       perm(state)
       if _r%4==0:
         state=small_swap(state)
         state[:,0,0]=state[:,0,0]^0x9e377900^_r
       if _r%4==2:
         state=big_swap(state)
    return convert_to_binary(state)

def convert_to_binary(arr):
  arr = arr.transpose().reshape((12, -1))
  X = np.zeros((len(arr) * word_size,len(arr[0])),dtype=np.uint8);
  for i in range(len(arr) * word_size):
    index = i // word_size;
    offset = word_size - (i % word_size) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);



# n x 384 -> n x 4 x 3 -> transpose
def convert_from_binary(arr, _dtype=np.uint32):
  num_words = arr.shape[1]//word_size
  X = np.zeros((len(arr), num_words),dtype=_dtype);
  for i in range(num_words):
    for j in range(word_size):
        pos = word_size*i+j
        X[:, i] += 2**(word_size-1-j)*arr[:, pos]
  X = X.reshape((len(arr), 4, 3))
  X = X.swapaxes(2,1)
  return(X);



test = np.random.randint(2**32-1, size = (10, 3, 4), dtype = np.uint32)
a = convert_to_binary(test)
b = convert_from_binary(a)
assert(np.all(b==test))

test_vector()
