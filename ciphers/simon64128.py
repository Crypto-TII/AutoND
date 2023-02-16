import numpy as np
from os import urandom

plain_bits = 64
key_bits = 128
word_size = 32

def WORD_SIZE():
    return(32)
def ALPHA():
    return(1)
def BETA():
    return(8)
def GAMMA():
    return(2)

MASK_VAL = 2**WORD_SIZE() - 1

def rol(x, k):
    return(((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k)))
def ror(x, k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MASK_VAL))

def enc_one_round(p, k):
    tmp, c1 = p[0], p[1]
    tmp = rol(tmp, ALPHA()) & rol(tmp, BETA())
    tmp = tmp ^ rol(p[0], GAMMA())
    c1 = c1 ^ tmp
    c1 = c1 ^ k
    return(c1, p[0])

def dec_one_round(c, k):
    p0, p1 = c[0], c[1]
    tmp = tmp ^ rol(p1, GAMMA())
    p1 = tmp ^ c[0] ^ k
    p0 = c1
    return(p0, p1)



def encrypt(p, k, r):
    P = convert_from_binary(p)
    K = convert_from_binary(k).transpose()
    ks = expand_key(K, r)
    x, y = P[:, 0], P[:, 1];
    for i in range(r):
        rk = ks[i]
        x,y = enc_one_round((x,y), rk);
    return convert_to_binary([x, y]);


def expand_key(k, t):
    ks = [0 for i in range(t)]
    ks[0:4] = reversed(k[0:4])
    m = 4
    round_constant = MASK_VAL ^ 3
    z = 0b11110000101100111001010001001000000111101001100011010111011011
    for i in range(m, t):
        c_z = ((z >> ((i-m) % 62)) & 1) ^ round_constant
        tmp = ror(ks[i-1], 3)
        tmp = tmp ^ ks[i-3]
        tmp = tmp ^ ror(tmp, 1)
        ks[i] = ks[i-m] ^ tmp ^ c_z
    return(ks)



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
def convert_from_binary(arr, _dtype=np.uint32):
  num_words = arr.shape[1]//WORD_SIZE()
  X = np.zeros((len(arr), num_words),dtype=_dtype);
  for i in range(num_words):
    for j in range(WORD_SIZE()):
        pos = WORD_SIZE()*i+j
        X[:, i] += 2**(WORD_SIZE()-1-j)*arr[:, pos]
  return(X);


def check_testvectors():
  p = np.uint32([0x656b696c, 0x20646e75]).reshape(-1, 1)
  k = np.uint32([0x1b1a1918,0x13121110,0x0b0a0908,0x03020100]).reshape(-1, 1)
  pb = convert_to_binary(p)
  kb = convert_to_binary(k)
  c = convert_from_binary(encrypt(pb, kb, 44))
  assert np.all(c[0] == [0x44c8fc20, 0xb9dfa07a])

check_testvectors()


