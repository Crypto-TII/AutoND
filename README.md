# AutoND: A Cipher-Agnostic Neural Training Pipeline  with Automated Finding of Good Input Differences

## Note 
In our manuscript we present results on SPECK32, SPECK64, SPECK128, SIMON32, SIMON64, SIMON128, GIMLI, HIGHT, LEA, TEA, XTEA, PRESENT and KATAN.
The SPECK implementation is adapted from that of Gohr (https://github.com/agohr/deep_speck), and the contents of the train_nets.py from the same repo
served as a base for gohrnet.py. 

## Demo for SPECK32
Please run the code by executing 
```bash
python main.py
```
The output should be similar to the following: 

```bash
======================================================================
PART 1: Finding the 0.1-close input differences and the `highest round` using the evolutionary optimizer for  speck3264_single-key ...
Found 1 0.1-close differences: ['0x400000'].
The highest round with a bias score above the threshold was 7.
The best differences and their scores for each round are stored under results/speck3264_single-key, and the full list of differences along with their weighted scores are stored under results/speck3264_single-key_best_weighted_differences.csv.

======================================================================
PART 2: Training DBitNet using the simple training pipeline...
Training dbitnet for input difference 0x400000, starting from round 5...
Training on 10 epochs ...
2000/2000 [==============================] - 113s 54ms/step - loss: 0.1203 - acc: 0.8486 - val_loss: 0.0889 - val_acc: 0.8867
Epoch 2/5
2000/2000 [==============================] - 110s 55ms/step - loss: 0.0828 - acc: 0.8957 - val_loss: 0.0730 - val_acc: 0.9095
Epoch 3/5
2000/2000 [==============================] - 110s 55ms/step - loss: 0.0717 - acc: 0.9108 - val_loss: 0.0661 - val_acc: 0.9167
Epoch 4/5
2000/2000 [==============================] - 111s 55ms/step - loss: 0.0657 - acc: 0.9171 - val_loss: 0.0650 - val_acc: 0.9174
Epoch 5/5
2000/2000 [==============================] - 110s 55ms/step - loss: 0.0648 - acc: 0.9176 - val_loss: 0.0646 - val_acc: 0.9177
dbitnet, round 5. Best validation accuracy: 0.9176750183105469
...
dbitnet, round 6. Best validation accuracy: 0.7640489935874939
...
dbitnet, round 7. Best validation accuracy: 0.6031960248947144
...
dbitnet, round 8. Best validation accuracy: 0.507686972618103
...
Epoch 1/5
2000/2000 [==============================] - 110s 55ms/step - loss: 0.2502 - acc: 0.5000 - val_loss: 0.2505 - val_acc: 0.5003
Epoch 2/5
2000/2000 [==============================] - 109s 55ms/step - loss: 0.2502 - acc: 0.5012 - val_loss: 0.2501 - val_acc: 0.5003
Epoch 3/5
2000/2000 [==============================] - 109s 55ms/step - loss: 0.2501 - acc: 0.5022 - val_loss: 0.2501 - val_acc: 0.5003
Epoch 4/5
2000/2000 [==============================] - 109s 55ms/step - loss: 0.2501 - acc: 0.5027 - val_loss: 0.2502 - val_acc: 0.4998
Epoch 5/5
2000/2000 [==============================] - 109s 55ms/step - loss: 0.2501 - acc: 0.5032 - val_loss: 0.2501 - val_acc: 0.4996
dbitnet, round 9. Best validation accuracy: 0.5003229975700378
{'Difference': '0x400000', 'dbitnet': {'Best round': 8, 'Validation accuracy': 0.507686972618103}}
```

In the first part, the evolutionary algorithm returns the input differences that scored within 10% of the optimal score (epsilon = 0.1). 
Here, only one was found: 0x400000. A bias was found up to round 7, and all the explored differences were stored in the indicated files.

In the second part, DBitNet is trained iteratively, according to our simple training pipeline, on all the input differences returned by part 1 (here, only 0x400000). After each round, the validation
accuracy is displayed. The training history, trained networks and final results are stored under results. The best significant distinguisher was trained for 8 rounds, with an accuracy of 0.51.

The results/speck3264_single-key file should look as follows:
```bash
New log start, reached round 6
Best at 1:
[0x4000, 0.4693]
[0x40c000, 0.4693]
[0x400000, 0.5]
[0x8000, 0.5]
[0x408000, 0.5]
...
Best at 6:
[0x302000, 0.0195]
[0x200000, 0.0196]
[0x702000, 0.0199]
[0x102000, 0.0211]
[0x400000, 0.0225]
Best Cumulative:
[0x102000, 1.2522]
[0x600000, 1.2539]
[0x200000, 1.2615]
[0x408000, 1.4173]
[0x400000, 1.5091]
Best Weighted:
[0x600000, 2.6753]
[0x200000, 2.7175]
[0x102000, 2.7579]
[0x408000, 3.0846]
[0x400000, 3.4593]
```
This output shows the best differences for each round and their scores, sorted. 

The results/speck3264_single-key_best_weighted_differences.csv file is a csv file contatining the weighted scores of all the differences explored during the search:
```bash
,Difference,Weighted score
0,{'0x9c1d528'},{0.4273}
1,{'0x882804c2'},{0.4289}
...
96,{'0x408000'},{3.0846}
97,{'0x400000'},{3.4593}
```

### Testing other ciphers
The ciphers folder contains all the supported primitives. The format is: 
```bash
python3 main.py [cipher] [model] 
```
cipher is a cipher name from the ciphers folder, and mode is 'single-key' or 'related-key'.
For instance, to run the tool on present in the related-key model:
```bash
python3 main.py present80 single-key
```
Please consider increasing the number of generations `NUM_GENERATIONS` and number of epochs `EPOCHS` parameters as discussed [below](#reproduce-results-from-table-5-and-6-in-the-manuscript).

## Reproducing the results from the manuscript
For demonstration purposes, the settings in the provided code are reduced to 
```python
NUM_GENERATIONS = 5 # 50 in the paper, set to 5 here for demonstration in optimizer.py
NUM_SAMPLES = 10**3 # 10**4 in the paper, set to 5 here for demonstration in optimizer.py
EPOCHS = 5          # 10 to 40 in the paper, set to 5 here for demonstration in train_nets.py
```
Please set them to the original values to reproduce the values obtained in the manuscript. 

In order to run Gohr's resnet, using our simple training pipeline, the trainNeuralDistinguishers function from main.py can be called with the corresponding argument:
```python
results = trainNeuralDistinguishers(cipher_name, scenario, output_dir, input_difference, max(1, highest_round-2), nets =['gohr'])
```

To run both models sequentially, one may call:
```python
results = trainNeuralDistinguishers(cipher_name, scenario, output_dir, input_difference, max(1, highest_round-2), nets =['gohr, dbitnet'])
```

## Adding a new cipher
Additional ciphers can be added following the template of e.g. `present.py`. The cipher file must include:
The parameter variables:
```python
plain_bits = 64
key_bits = 80
word_size = 4
```
An encryption function, which takes as input numpy binary matrices, with one row per sample, representing respectively the plaintext and the key, and the number of rounds:
```python
def encrypt(p, k, r):
```

## Evaluate an existing model on fresh test datasets
To evaluate an existing model with weights stored in `modelpath` on fresh test datasets, please run, for example
For example, running 

```bash
python AutoND/eval_nets.py --model_path 'AutoND/model_weights/speck3264_single-key_0x400000_round8_polished.h5' \
--model_type 'dbitnet' --cipher 'speck3264' --input_difference '0x400000' --round_number '8'
```
results in 
```bash
INFO:root:Creating model 'dbitnet' with weights from path 
	 'AutoND/model_weights/speck3264_single-key_0x400000_round8_polished.h5'...
INFO:root:Creating new validation dataset for 
                        cipher: speck3264, 
                        scenario: single-key, 
                        input difference: 0x400000, 
                        round number: 8, 
                        cipher.plain_bits: 32, 
                        cipher.key_bits: 64
                        ...
INFO:root:Running evaluations...
INFO:root:	 acc=0.5143 	 tpr=0.5119 	 tnr=0.5168
INFO:root:	 acc=0.5145 	 tpr=0.5138 	 tnr=0.5152
INFO:root:	 acc=0.5143 	 tpr=0.5125 	 tnr=0.5162
INFO:root:	 acc=0.5146 	 tpr=0.5120 	 tnr=0.5173
INFO:root:	 acc=0.5139 	 tpr=0.5119 	 tnr=0.5159
INFO:root:Saving results to AutoND/model_weights/speck3264_single-key_0x400000_round8_polished_eval.npz
```
and loading the results via numpy and computing the mean and standard deviation yields the accuracy from our manuscript: 
```
import numpy  as np
data = np.load('AutoND/model_weights/speck3264_single-key_0x400000_round8_polished_eval.npz')
print(f"{np.mean(data['accs']):.4f} +- {np.std(data['accs']):.4f}")
>>> 0.5144 +- 0.0003
```

Similarly, running
```bash
python AutoND/eval_nets.py --model_path "AutoND/model_weights/simon3264_single-key_0x400_round11_polished.h5" \
--model_type 'dbitnet' --cipher 'simon3264' --input_difference '0x400' --round_number '11'
```
yields 
```
INFO:root:Running evaluations...
INFO:root:	 acc=0.5185 	 tpr=0.4780 	 tnr=0.5591
INFO:root:	 acc=0.5183 	 tpr=0.4780 	 tnr=0.5586
INFO:root:	 acc=0.5183 	 tpr=0.4766 	 tnr=0.5598
INFO:root:	 acc=0.5185 	 tpr=0.4781 	 tnr=0.5590
INFO:root:	 acc=0.5176 	 tpr=0.4774 	 tnr=0.5579
...
>>> 0.5182 +- 0.0004
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
