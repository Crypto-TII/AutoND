# AutoND: A Cipher-Agnostic Neural Training Pipeline  with Automated Finding of Good Input Differences

_(This work has been submitted and is currently under review. Please keep this code confidential.)_

## Note 
In our manuscript we present results on SPECK64, SPECK128, SIMON64, SIMON128, the GIMLI-PERMUTATION, HIGHT, LEA, TEA, XTEA and PRESENT.
Our cipher implementations are automatically generated with a proprietary library, so we prefer to not publish them for now. 
We provide the SPECK implementation of Gohr [TODO] as an example.

## Demo for SPECK32
Please run the code by executing 
```bash
python main.py
```
The output should be similar to the following: 

```bash
======================================================================
PART 1: Find the `best input difference` and the `highest round` using the evolutionary optimizer...
Generation 0/5, 528 nodes explored, 32 current, best is ['0x222d802', ..., '0x46c9c00'] with [0.2751875 ... 0.325375 ]
Generation 1/5, 1024 nodes explored, 32 current, best is ['0xf02337e0', ..., '0x44e9c00'] with [0.3186875 ...  0.34425  ]
...
Final :  6
```
This output shows the search process, where the current best few differences and their scores are displayed at each generation, for each round. 
`Final 6` means that the search ended at round 6 (because more rounds don't appear to be biased).

```bash
Best at 1 : 
 ['0xd729a02d', '0x4671800d', ..., '0x400000']
 [0.14318750000000005, 0.20981249999999999, ..., , 0.5]
...
Best at 5 : 
 ['0x44e9c00', '0x8001010', ..., '0x400000']
[0.009593750000000009, 0.009812500000000005, ..., 0.06493750000000001]
```
This output shows the best differences for each round and their scores, ranked from bad to good. 

```bash
Best Weighted : 
 ['0xd729a02d', '0x4671800d', ..., , '0x400000']
[0.34765625000000017, 0.42078125000000016,... , 3.31446875]
```
The final output bit is the weighted scores (sum of the scores at each round times the round number). The best difference overall is the last one in the best weighted list.

The best difference and the highest round found by the evolutionary optimizer are passed to the neural network distinguisher. 
The training progress for each round of the staged training is shown.
```bash
======================================================================
PART 2: Train DBitNet using staged training  
INFO:root:CREATE NEURAL NETWORK MODEL.
INFO:root:determined cipher input size = 64
INFO:root:CREATE DATA for round 5...
INFO:root:TRAIN neural network for round 5...
Epoch 1/40
2000/2000 [==============================] - 39s 17ms/step - loss: 0.0999 - acc: 0.8725 - val_loss: 0.0894 - val_acc: 0.8859
...
Best validation accuracy:  0.9282900094985962
INFO:root:CREATE DATA for round 6...
INFO:root:TRAIN neural network for round 6...
Epoch 1/40
2000/2000 [==============================] - 34s 17ms/step - loss: 0.1524 - acc: 0.7807 - val_loss: 0.1505 - val_acc: 0.7831
...
Best validation accuracy:  0.7862939834594727
INFO:root:CREATE DATA for round 7...
INFO:root:TRAIN neural network for round 7...
Epoch 1/40
2000/2000 [==============================] - 34s 17ms/step - loss: 0.2350 - acc: 0.6054 - val_loss: 0.2334 - val_acc: 0.6092
...
Best validation accuracy:  0.6146360039710999
INFO:root:CREATE DATA for round 8...
INFO:root:TRAIN neural network for round 8...
Epoch 1/40
2000/2000 [==============================] - 34s 17ms/step - loss: 0.2510 - acc: 0.5082 - val_loss: 0.2506 - val_acc: 0.5115
Epoch 2/40
2000/2000 [==============================] - 33s 17ms/step - loss: 0.2504 - acc: 0.5128 - val_loss: 0.2503 - val_acc: 0.5113
Epoch 3/40
2000/2000 [==============================] - 33s 17ms/step - loss: 0.2501 - acc: 0.5142 - val_loss: 0.2503 - val_acc: 0.5102
...
Best validation accuracy:  0.5114840269088745
INFO:root:CREATE DATA for round 9...
INFO:root:TRAIN neural network for round 9...
...
INFO:root:ABORT TRAINING (best validation accuracy <= 50.5)
```
The training is aborted in the round where the best validation accuracy falls below 50.5%. 

### Test SPECK64 or SPECK128
Please change `main.py` accordingly, e.g. for SPECK64: 
```python
#import speck3264 as cipher
import speck64128 as cipher
#import speck128256 as cipher
```

## Reproduce results from table 5 and 6 in the manuscript
For demonstration purposes, the settings in the provided code are reduced to 
```python
NUM_GENERATIONS = 5 # 50 in the paper, set to 5 here for demonstration in optimizer.py
EPOCHS = 5          # 40 in the paper, set to 5 here for demonstration in train_nets.py
```
Please set them to the original values to reproduce the values obtained in the manuscript. For SPECK32 the demonstration settings should still reach round 8 with 50.9% validation accuracy.

## Adding a new cipher
Please create a `mycipher.py` Python file with the structure as shown in e.g. `speck3264.py`. Modify `main.py` accordingly:
```python
#import speck3264 as cipher
#import speck64128 as cipher
#import speck128256 as cipher
import mycipher as cipher
```

## Prerequisites
The code execution relies on standard Python modules, except for `tensorflow`.
If you start from an empty Python Anaconda environment, the following installation should be sufficient: 
```bash
conda create -n tf-gpu tensorflow-gpu
conda activate tf-gpu
conda install -c nvidia cuda-nvcc
conda install pandas
```
On [GoogleColaboratory](https://colab.research.google.com/) the code will run out-of-the-box without any additional installations.
