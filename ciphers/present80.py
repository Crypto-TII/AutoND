# This is a vectorized implementation of the SPECK32 cipher, compatible with our optimizer and neural distinguisher.

#### ADDING A CIPHER
# In order to be compatible with this repo, a cipher implementation must:
# - Provide plain_bits and key_bits variables, giving respectively the number of bits in the plaintext and in the key.
# - Provide a vectorized “encrypt(p, k, r)” function, that takes as input, for n samples:
#       - An n by plain_bits binary matrix p of numpy.uint8 for the plaintexts;
#       - An n by key_bits binary matrix of numpy.uint8 for the keys;
#       - A number of rounds r.
# The encrypt function must return an n by plain_bits matrix of numpy.uint8 containing the ciphertexts. The encrypt function in the provided in this file exemplifies the use of the functions “convert_to_binary” and “convert_from_binary” to translate between binary matrices and the native format of the cipher implementation.

import numpy as np

plain_bits = 64
key_bits = 80

def WORD_SIZE():
    return(64);

Sbox = np.uint8([0xc, 0x5, 0x6, 0xb, 0x9, 0x0, 0xa, 0xd, 0x3, 0xe, 0xf, 0x8, 0x4, 0x7, 0x1, 0x2])
PBox = np.uint8([0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51,
        4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55,
        8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59,
        12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63])


def SB(arr):
    num_words = arr.shape[1]//4
    S = arr.copy()
    for i in range(num_words):
        to_sub = 0
        for j in range(4):
            pos = 4*i+j
            to_sub += 2**(3-j)*arr[:, pos]
        S[:, 4*i:4*(i+1)] = np.unpackbits(Sbox[to_sub[:, None]], axis = 1)[:, -4:]
    return S

def P(arr):
    arr[:, PBox] = arr[:, np.arange(64)]
    return arr



def expand_key(k, t):
    ks = [0 for i in range(t)];
    key = k.copy()
    for r in range(t):
        ks[r] = key[:, :64]
        key = np.roll(key, 19, axis = 1)
        key[:, :4] = SB(key[:, :4])
        key[:, -23:-15] ^= np.unpackbits(np.uint8(r+1))
    return ks



# The encrypt function must adhere to this format, with p and k being binary matrices representing the plaintexts and the key, and the return value being a binary matrix as well.
def encrypt(p, k, r):
    ks = expand_key(k, r)
    c = p.copy()
    for i in range(r-1):
        c ^= ks[i]
        c = SB(c)
        c = P(c)
    return c^ks[-1]



#convert_to_binary takes as input an array of ciphertext pairs
#where the first row of the array contains the lefthand side of the ciphertexts,
#the second row contains the righthand side of the ciphertexts,
#the third row contains the lefthand side of the second ciphertexts,
#and so on
#it returns an array of bit vectors containing the same data
def convert_to_binary(arr):
  X = np.zeros((len(arr) * WORD_SIZE(),len(arr[0])),dtype=np.uint8);
  for i in range(len(arr) * WORD_SIZE()):
    index = i // WORD_SIZE();
    offset = WORD_SIZE() - (i % WORD_SIZE()) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);

# Convert_from_binary takes as input an n by num_bits binary matrix of type np.uint8, for n samples,
# and converts it to an n by num_words array of type dtype.
def convert_from_binary(arr, _dtype=np.uint64):
  num_words = arr.shape[1]//WORD_SIZE()
  X = np.zeros((len(arr), num_words),dtype=_dtype);
  for i in range(num_words):
    for j in range(WORD_SIZE()):
        pos = WORD_SIZE()*i+j
        X[:, i] += 2**(WORD_SIZE()-1-j)*arr[:, pos]
  return(X);

def check_testvector():
    p = np.zeros((1, 64), dtype = np.uint8)
    k = np.zeros((1, 80), dtype = np.uint8)
    C = convert_from_binary(encrypt(p,k,32))
    Chex = hex(C[0][0])
    assert Chex == "0x5579c1387b228445"

check_testvector()



