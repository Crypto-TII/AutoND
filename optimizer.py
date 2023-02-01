import numpy as np
import os
from random import randint
import sys
from os import urandom
from copy import deepcopy
from types import FunctionType
from timeit import default_timer as timer
import numpy as np

NUM_GENERATIONS = 5      # 50 in the paper, set to 5 here for demonstration
NUM_SAMPLES = 10**3      # 10**3 in the paper. The number of samples used to compute the bias score

def bitArrayToIntegers(arr):
    packed = np.packbits(arr,  axis = 1)
    return [int.from_bytes(x.tobytes(), 'big') for x in packed]

# Computes the bias scores of several candidate_differences, based on the initial plaintexts and keys pt0, keys and the corresponding ciphertexts C0, for nr rounds of a cipher with plain_bits plaintext bits and key_bits key bits.
def evaluate_multiple_differences(candidate_differences, pt0, keys, C0, nr, plain_bits, key_bits, encrypt, scenario = "single-key"):
    dp = candidate_differences[:, :plain_bits]
    pt1   = (np.broadcast_to(dp[:, None, :], (len(candidate_differences), len(pt0), plain_bits))^pt0).reshape(-1, plain_bits)
    if scenario == "related-key":
        dk = candidate_differences[:, plain_bits: ]
    else:
        dk = np.zeros((len(candidate_differences), key_bits), dtype=np.uint8)
    keys1 = (np.broadcast_to(dk[:, None, :], (len(candidate_differences), len(pt0), key_bits))^keys).reshape(-1, key_bits)
    C1 = encrypt(pt1, keys1, nr)
    differences_in_output =  C1.reshape(len(candidate_differences), len(pt0),-1)^C0
    scores = np.average(np.abs(0.5-np.average(differences_in_output, axis = 1)), axis=1)
    zero_diffs = np.where(np.sum(candidate_differences, axis = 1)==0)
    scores[zero_diffs] = 0
    return scores

# Evolutionary algorithm based on the encryption function f, running for n generations, using differences of num_bits bits, a population size of L, an optional initial population gen, and verbosity set to 0 for silent or 1 for verbose.
def evo(f, n=NUM_GENERATIONS, num_bits=32, L = 32, gen=None, verbose = 1):
    mutProb = 100
    #gen = np.array([1<<i for i in range(num_bits)], dtype=object)
    #gen = np.array([randint(1,2**num_bits-1) for i in range(pop_size)], dtype=object)
    if gen is None:
        gen = np.random.randint(2, size = (L**2, num_bits), dtype=np.uint8)
    scores = f(gen)
    idx = np.arange(len(gen))
    explored = np.copy(gen)
    good = idx[np.argsort(scores)][-L:]
    gen = gen[good]
    scores = scores[good]
    cpt = len(gen)
    for generation in range(n):
        # New generation
        kids = np.array([gen[i] ^ gen[j] for i in range(len(gen)) for j in range(i+1, len(gen))], dtype = np.uint8);
        # Mutation: selecting mutating kids
        selected = np.where(np.random.randint(0,100, len(kids))>(100-mutProb))
        numMut = len(selected[0])
        # Selected kids are XORed with 1<<r (r random)
        tmp = kids[selected].copy()
        kids[selected[0].tolist(), np.random.randint(num_bits, size = numMut)] ^=1
        # Removing kids that have been explored before and duplicates
        kids = np.unique(kids[(kids[:, None] != explored).any(-1).all(-1)], axis=0)
        # Appending to explored
        explored = np.vstack([explored, kids])
        cpt+=len(kids)
        # Computing the scores
        if len(kids)>0:
            scores = np.append(scores, f(kids))
            gen = np.vstack([gen, kids])
         # Sorting, keeping only the L best ones
            idx = np.arange(len(gen))
            good = idx[np.argsort(scores)][-L:]
            gen = gen[good]
            scores = scores[good]
        if verbose:
            genInt = np.packbits(gen[-4:, :],  axis = 1)
            hexGen = [hex(int.from_bytes(x.tobytes(), 'big')) for x in genInt]
            print(f'Generation {generation}/{n}, {cpt} nodes explored, {len(gen)} current, best is {[x for x in hexGen]} with {scores[-4:]}', flush=True)
        if np.all(scores == 0.5):
            break
    return gen, scores

def optimize(plain_bits, key_bits, encryption_function, nb_samples=NUM_SAMPLES, scenario = "single-key"):
    allDiffs = None
    totalScores = {}
    diffs = None
    T = 0.05
    current_round = 1
    if scenario == "single-key":
        bits_to_search = plain_bits
    else:
        bits_to_search = plain_bits+key_bits
    while True:
        keys0 = (np.frombuffer(urandom(nb_samples*key_bits),dtype=np.uint8)&1).reshape(nb_samples, key_bits);
        pt0 = (np.frombuffer(urandom(nb_samples*plain_bits),dtype=np.uint8)&1).reshape(nb_samples, plain_bits);
        C0 = encryption_function(pt0, keys0, current_round)
        #diffs, scores = evo(f=lambda x: evaluate_multiple_differences(x, pt0, keys0, C0, current_round, plain_bits, key_bits, encryption_function), num_bits = bits_to_search, L=32, gen=None)
        # The initial set of differences can be set to None, or to the differences returned for the previous round. We use the second option here, as opposed to the first (above) in the paper.
        diffs, scores = evo(f=lambda x: evaluate_multiple_differences(x, pt0, keys0, C0, current_round, plain_bits, key_bits, encryption_function, scenario=scenario), num_bits = bits_to_search, L=32, gen=diffs)
        if allDiffs is None:
            allDiffs = diffs
        else:
            allDiffs = np.concatenate([allDiffs, diffs])
        if scores[-1] < T:
            break
        print(current_round, scores[-1])
        current_round += 1

        # Reevaluate all differences for best round:
    print("Final : ", current_round)
    finalScores = [None for i in range(current_round)]
    allDiffs = np.unique(allDiffs, axis=0)
    cumulativeScores = np.zeros(len(allDiffs))
    weightedScores = np.zeros(len(allDiffs))
    for nr in range(1, current_round):
        keys0 = (np.frombuffer(urandom(nb_samples*key_bits),dtype=np.uint8)&1).reshape(nb_samples, key_bits);
        pt0 = (np.frombuffer(urandom(nb_samples*plain_bits),dtype=np.uint8)&1).reshape(nb_samples, plain_bits);
        C0 = encryption_function(pt0, keys0, nr)
        finalScores[nr] = evaluate_multiple_differences(allDiffs, pt0, keys0, C0, nr, plain_bits, key_bits, encryption_function, scenario = scenario)
        cumulativeScores += np.array(finalScores[nr])
        weightedScores += nr*np.array(finalScores[nr])

        idx = np.arange(len(allDiffs))
        good = idx[np.argsort(finalScores[nr])]
        diffs = allDiffs[good]
        scores = finalScores[nr][good]

        print(f'Best at {nr} : \n {[hex(x) for x in bitArrayToIntegers(diffs)]}\n {scores.tolist()}', flush=True)

    idx = np.arange(len(allDiffs))
    good = idx[np.argsort(weightedScores)]
    diffs = allDiffs[good]
    scores = weightedScores[good]
    print(f'Best Weighted : \n {[hex(x) for x in bitArrayToIntegers(diffs)]}\n {scores.tolist()}', flush=True)

    return(diffs, current_round)


def optimize_multi(plain_bits, key_bits, encryption_functionA, encryption_functionB, nb_samples=NUM_SAMPLES, scenario = "single-key"):
    allDiffs = None
    totalScores = {}
    diffs = None
    T = 0.05
    current_round = 1
    if scenario == "single-key":
        bits_to_search = plain_bits
    else:
        bits_to_search = plain_bits+key_bits
    while True:
        keys0 = (np.frombuffer(urandom(nb_samples*key_bits),dtype=np.uint8)&1).reshape(nb_samples, key_bits);
        pt0 = (np.frombuffer(urandom(nb_samples*plain_bits),dtype=np.uint8)&1).reshape(nb_samples, plain_bits);
        C0 = encryption_functionA(pt0, keys0, current_round)
        #diffs, scores = evo(f=lambda x: evaluate_multiple_differences(x, pt0, keys0, C0, current_round, plain_bits, key_bits, encryption_function), num_bits = bits_to_search, L=32, gen=None)
        # The initial set of differences can be set to None, or to the differences returned for the previous round. We use the second option here, as opposed to the first (above) in the paper.
        diffs, scores = evo(f=lambda x: evaluate_multiple_differences(x, pt0, keys0, C0, current_round, plain_bits, key_bits, encryption_functionB, scenario=scenario), num_bits = bits_to_search, L=32, gen=diffs)
        if allDiffs is None:
            allDiffs = diffs
        else:
            allDiffs = np.concatenate([allDiffs, diffs])
        if scores[-1] < T:
            break
        print(current_round, scores[-1])
        current_round += 1

        # Reevaluate all differences for best round:
    print("Final : ", current_round)
    finalScores = [None for i in range(current_round)]
    allDiffs = np.unique(allDiffs, axis=0)
    cumulativeScores = np.zeros(len(allDiffs))
    weightedScores = np.zeros(len(allDiffs))
    for nr in range(1, current_round):
        keys0 = (np.frombuffer(urandom(nb_samples*key_bits),dtype=np.uint8)&1).reshape(nb_samples, key_bits);
        pt0 = (np.frombuffer(urandom(nb_samples*plain_bits),dtype=np.uint8)&1).reshape(nb_samples, plain_bits);
        C0 = encryption_functionA(pt0, keys0, nr)
        finalScores[nr] = evaluate_multiple_differences(allDiffs, pt0, keys0, C0, nr, plain_bits, key_bits, encryption_functionB, scenario = scenario)
        cumulativeScores += np.array(finalScores[nr])
        weightedScores += nr*np.array(finalScores[nr])

        idx = np.arange(len(allDiffs))
        good = idx[np.argsort(finalScores[nr])]
        diffs = allDiffs[good]
        scores = finalScores[nr][good]

        print(f'Best at {nr} : \n {[hex(x) for x in bitArrayToIntegers(diffs)]}\n {scores.tolist()}', flush=True)

    idx = np.arange(len(allDiffs))
    good = idx[np.argsort(weightedScores)]
    diffs = allDiffs[good]
    scores = weightedScores[good]
    print(f'Best Weighted : \n {[hex(x) for x in bitArrayToIntegers(diffs)]}\n {scores.tolist()}', flush=True)

    return(diffs, current_round)

