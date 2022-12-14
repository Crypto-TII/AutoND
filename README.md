# AutoND: A Cipher-Agnostic Neural Training Pipeline  with Automated Finding of Good Input Differences

_(This work has been submitted and is currently under review. Please keep this code confidential.)_

## Note 
In our manuscript we present results on SPECK64, SPECK128, SIMON64, SIMON128, the GIMLI-PERMUTATION, HIGHT, LEA, TEA, XTEA and PRESENT.
Our cipher implementations are automatically generated with a proprietary library, so we prefer to not publish them for now. 
We provide the SPECK implementation of Gohr [TODO] as an example.

## Demo 
Please run the code by executing 
```bash
python main.py
```
The output should be similar to the following (for SPECK32/64): 
```bash
INFO:root:CREATE NEURAL NETWORK MODEL.
INFO:root:determined cipher input size = 64
INFO:root:CREATE DATA for round 5...
INFO:root:TRAIN neural network for round 5...
Epoch 1/40
2000/2000 [==============================] - 39s 17ms/step - loss: 0.0999 - acc: 0.8725 - val_loss: 0.0894 - val_acc: 0.8859
...
Epoch 40/40
2000/2000 [==============================] - 33s 17ms/step - loss: 0.0556 - acc: 0.9291 - val_loss: 0.0561 - val_acc: 0.9283
Best validation accuracy:  0.9282900094985962
INFO:root:CREATE DATA for round 6...
INFO:root:TRAIN neural network for round 6...
Epoch 1/40
2000/2000 [==============================] - 34s 17ms/step - loss: 0.1524 - acc: 0.7807 - val_loss: 0.1505 - val_acc: 0.7831
...
Epoch 40/40
2000/2000 [==============================] - 33s 17ms/step - loss: 0.1473 - acc: 0.7891 - val_loss: 0.1488 - val_acc: 0.7861
Best validation accuracy:  0.7862939834594727
INFO:root:CREATE DATA for round 7...
INFO:root:TRAIN neural network for round 7...
Epoch 1/40
2000/2000 [==============================] - 34s 17ms/step - loss: 0.2350 - acc: 0.6054 - val_loss: 0.2334 - val_acc: 0.6092
...
Epoch 40/40
2000/2000 [==============================] - 33s 17ms/step - loss: 0.2301 - acc: 0.6203 - val_loss: 0.2320 - val_acc: 0.6144
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
Epoch 40/40
2000/2000 [==============================] - 33s 17ms/step - loss: 0.2498 - acc: 0.5305 - val_loss: 0.2521 - val_acc: 0.5048
Best validation accuracy:  0.5114840269088745
INFO:root:CREATE DATA for round 9...
INFO:root:TRAIN neural network for round 9...

INFO:root:ABORT TRAINING (best validation accuracy <= 50.5)
```