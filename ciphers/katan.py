import numpy as np
from os import urandom



plain_bits = 32
key_bits = 80
word_size = 16

if plain_bits == 32:
    LEN_L1 = 13
    LEN_L2 = 19
    X = (None, 12, 7, 8, 5, 3)
    Y = (None, 18, 7, 12, 10, 8, 3)
elif plain_bits == 48:
    LEN_L1 = 19
    LEN_L2 = 29
    X = (None, 18, 12, 15, 7, 6)
    Y = (None, 28, 19, 21, 13, 15, 6)
else:
    LEN_L1 = 25
    LEN_L2 = 39
    X = (None, 24, 15, 20, 11, 9)
    Y = (None, 38, 25, 33, 21, 14, 9)

IR = (
    1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0,
    1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0,
    0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
    1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1,
    0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
    1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1,
    0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1,
    1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1,
    1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1,
    0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1,
    1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0,
    0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1,
    0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
    1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0
    )

MASK_VAL = 2 ** plain_bits - 1;





def encrypt(plaintext = [], K = [], nr = 0):
  P = np.flip(plaintext, axis = 1)
  ks = np.zeros(shape = (len(P), np.max([2*nr, 80])), dtype = np.uint8)
  ks[:, :80]  = np.flip(K, axis=1)
  for i in range(80, nr*2):
      ks[:, i] = ks[:, i-80] ^ ks[:, i-61] ^ks[:, i-50] ^ks[:, i-13]  
  for i in range(nr):
      fa = P[:, LEN_L2 + X[1]]^P[:, LEN_L2+ X[2]] ^ ks[:, 2*i] ^ (P[:,LEN_L2+ X[3]] & P[:, LEN_L2+X[4]]) ^ (P[:,LEN_L2+ X[5]] & IR[i])
      fb = P[:, Y[1]]^P[:, Y[2]] ^ (P[:, Y[3]] & P[:, Y[4]]) ^ (P[:, Y[5]] & P[:, Y[6]]) ^ ks[:, 2*i+1]
      P = np.roll(P, 1, axis=1)
      P[:, 0] = fa
      P[:, LEN_L2] = fb

  return np.flip(P, axis=1)


#convert_to_binary takes as input an array of ciphertext pairs
#where the first row of the array contains the lefthand side of the ciphertexts,
#the second row contains the righthand side of the ciphertexts,
#the third row contains the lefthand side of the second ciphertexts,
#and so on
#it returns an array of bit vectors containing the same data
def convert_to_binary(arr):
  X = np.zeros((len(arr) * plain_bits,len(arr[0])),dtype=np.uint8);
  for i in range(len(arr) * plain_bits):
    index = i // plain_bits;
    offset = plain_bits - (i % plain_bits) - 1;
    X[i] = (arr[index] >> offset) & 1;
  X = X.transpose();
  return(X);

# Convert_from_binary takes as input an n by num_bits binary matrix of type np.uint8, for n samples,
# and converts it to an n by num_words array of type dtype.
def convert_from_binary(arr, _dtype=np.uint64):
  num_words = arr.shape[1]//plain_bits
  X = np.zeros((len(arr), num_words),dtype=_dtype);
  for i in range(num_words):
    for j in range(plain_bits):
        pos = plain_bits*i+j
        X[:, i] += 2**(plain_bits-1-j)*arr[:, pos]
  return(X);



def check_testvectors():
    p = np.zeros((2, plain_bits), dtype = np.uint8)
    p[1]^=1
    k = np.zeros((2, key_bits), dtype = np.uint8)
    k[0]^=1
    C = convert_from_binary(encrypt(p,k,254)).flatten()
    assert hex(C[0]) == '0x7e1ff945'  
    assert hex(C[1]) == '0x432e61da'  


check_testvectors()




