### 13/10 added qSLP and Utils_general:
- *qSLP.ipynb*: is a step by step creation of a working qSLP.
- *Utils_general*: contains a function that accepts multiple parameters and allow the caller to change the state preparation of the circuit. It implements circuits from both the files _Generalised qSLP_ and _Experiment.ipynb_

From the repeated runs of the function we observe the following results for the MNIST dataset:
| Pad | Siqbit | d | AcTrain | AcTest | Cost  | Iteration  |
|-----|--------|---|---------|--------|-------|-------|
| T   | F      | 1 | 0.919   | 0.844  | 0.072 | 3     |
| T   | F      | 2 | 0.854   | 0.777  | 0.066 | 3     |
| T   | F      | 3 | 0.496   | 0.533  | 0.044 | 3     |
| T   | F      | 4 | 0.726   | 0.689  | 0.042 | 3     |
| T   | F      | 5 | 0.704   | 0.767  | 0.051 | 2     |
| F   | T      | 1 | 0.481   | 0.667  | 0.000 | 1     |
| F   | T      | 2 | 0.481   | 0.667  | 0.002 | 1     |
| F   | T      | 3 | 0.481   | 0.667  | 0.002 | 1     |
| F   | T      | 4 | 0.481   | 0.667  | 0.003 | 1     |
| F   | T      | 5 | 0.481   | 0.667  | 0.005 | 1     |

Here we can clearly see that some results are wrong, namely the one with Pad=False. Solving this error will be the main objective going forward.

A qbit count is not present but to obtain how many qbit are used by the circuit we just solve:
$$  d + 4^{p} $$ where  $$p = 0\: \it{if}\:\: \bold{Siqbit}== True,$$$$ p = 1  \:\:\it{otherwise}  $$

### 7/11 modified qSLP
Now the Pad=False outputs a different, apparently working, result.

### 9/11 added QVSM
- **QVSM**: step by step creation of a quantum SVM both with _qiskit_ QSVC and with _sikitlearn_ SVC

runned QVSC on the MINS dataset and obtained:
**kernel classification test score**: 0.9333333333333333
Note that this result don't change when changing the _feature_dim_ parameter, suggesting a possible problem in the initialization.

### 16/11 added qiskit QNN
Added:
- **qnnc.ipybn** a step by step construction and training of the two classification methods for NN made by qiskit.
- **IBMQ.csv** containing results from _qncc.ipynb_
- **qSLP.csv** containing results from _qSLP.ipynb_

Changed the overall structure of the repository for a better understanding of the content. 

From the experiments runned it  appears that both QVSM and QNNC utilize the same state preparation techniques. In the experiments this corresponds in the ZZFeatureMap but this can be changed in any feature map from the library.

### 26/11/2021
Slightly modified collect_results.ipynb to return the mean results in markdown 
|(dataset, padding, control Qbits) |   ac_train |   ac_test |   cost_test |
|:---------------------|-----------:|----------:|------------:|
| ('MNIST', False, 1)  |   0.66675  |  0.6003   |   0.053     |
| ('MNIST', False, 2)  |   0.664333 |  0.5553   |   0.0513    |
| ('MNIST', False, 3)  |   0.67     |  0.6      |   0.057     |
| ('MNIST', False, 4)  |   0.6595   |  0.533    |   0.057     |
| ('MNIST', True, 1)   |   0.686667 |  0.5557   |   0.0427    |
| ('MNIST', True, 2)   |   0.7315   |  0.5503   |   0.066     |
| ('MNIST', True, 3)   |   0.6266   |  0.48     |   0.0514    |
| ('MNIST', True, 4)   |   0.716333 |  0.6443   |   0.0423    |
| ('MNIST', True, 5)   |   0.6815   |  0.667    |   0.053     |
| ('breast', False, 1) |   0.526    |  0.4      |   0.017     |
| ('breast', False, 2) |   0.563    |  0.333    |   0.023     |
| ('breast', False, 3) |   0.57     |  0.4      |   0.02      |
| ('breast', False, 4) |   0.563    |  0.333    |   0.033     |
| ('breast', True, 1)  |   0.519    |  0.333    |   0.054     |
| ('breast', True, 2)  |   0.496    |  0.533    |   0.033     |
| ('breast', True, 3)  |   0.496    |  0.533    |   0.032     |
| ('breast', True, 4)  |   0.489    |  0.6      |   0.028     |
| ('moon', False, 1)   |   0.8      |  0.6      |   0.038     |
| ('moon', False, 2)   |   0.756    |  0.9      |   0.043     |
| ('moon', False, 3)   |   0.778    |  0.5      |   0.041     |
| ('moon', False, 4)   |   0.8      |  0.5      |   0.043     |
| ('moon', True, 1)    |   0.833    |  0.9      |   0.149     |
| ('moon', True, 2)    |   0.744    |  0.9      |   0.056     |
| ('moon', True, 3)    |   0.533    |  0.6      |   0.046     |
| ('moon', True, 4)    |   0.633    |  0.7      |   0.05      |

### 27/11/2021
Changed **sQLP**: the circuit involving padding now uses cU3 instead of temporary qubits with swaps.

### 28/11/2021
**Utils_general** cleaning.

### 29/11/2021
Started experimenting with different initial parameters for training. The chosen approach is to obtain a random sample from a normal distribution with mean 1 and std 1.
### 30/11/2021
new results available for the MNIST dataset comparing classes 0-9 and 3-8. The best results are:


|     | name    |   d | pad   |   acc_train |   acc_test |   cost_test |    training time (s) |
|----:|:--------|----:|:------|------------:|-----------:|------------:|---------:|
|  86 | MNIST09 |   1 | False |       0.767 |        0.7 |       0.027 |  203.914 |
|  42 | MNIST09 |   2 | False |       0.767 |        0.6 |       0.04  |  607.436 |
|  72 | MNIST09 |   1 | True  |       0.922 |        0.8 |       0.053 |  801.578 |
|  68 | MNIST09 |   2 | True  |       0.922 |        0.9 |       0.072 | 1163.92  |
|  20 | MNIST38 |   1 | False |       0.656 |        0.6 |       0.032 |  229.683 |
| 105 | MNIST38 |   2 | False |       0.578 |        0.6 |       0.013 |  585.79  |
|  87 | MNIST38 |   1 | True  |       0.8   |        0.9 |       0.212 |  836.254 |
|  59 | MNIST38 |   2 | True  |       0.8   |        0.9 |       0.208 | 1228.69  |
| 157 | iris01  |   1 | False |       0.833 |        1   |       0.002 |  259.099 |
| 151 | iris01  |   2 | False |       0.822 |        0.9 |       0.003 |  598.938 |
| 148 | iris01  |   1 | True  |       0.633 |        0.6 |       0.047 |  852.883 |
| 166 | iris01  |   2 | True  |       0.667 |        0.6 |       0.035 | 1215.74  |
| 182 | iris02  |   1 | False |       0.878 |        0.9 |       0.007 |  294.972 |
| 164 | iris02  |   2 | False |       0.9   |        0.9 |       0.007 |  599.695 |
| 149 | iris02  |   1 | True  |       0.989 |        1   |       0.143 |  783.584 |
| 155 | iris02  |   2 | True  |       1     |        1   |       0.012 | 1217.35  |
| 141 | iris12  |   1 | False |       0.544 |        0.5 |       0.016 |  267.435 |
| 150 | iris12  |   2 | False |       0.544 |        0.5 |       0.015 |  599.481 |
| 147 | iris12  |   1 | True  |       0.533 |        0.5 |       0.02  |  828.236 |
| 177 | iris12  |   2 | True  |       0.544 |        0.5 |       0.024 | 1217.25  |

Padding seems to work better but requires more than twice the training time.
Launched experiments with every Iris combination.

### 01/12/2021
Corrected table from 30/01.
Changed file **collect_results.ipybn** into a utility module **result_handler.py** that given the csv file with all the results can store the best in another file along with their parameters and (__store_best_param__) and can read and return the desidered parameters (__get_best_par__).
Changed the format of qSLP_result to include parameter after the optimization.
Cleaned **Utils_general.py**, **utils_pad.py**, and **Utils.py** from unused and duplicated methods.

### 06/12/2021
- **qSLP.py**: new file containing a new version of qSLP inclosed in a class. It contains the class **Pad_circ** that stores a parametrized vesion of qSLP with padding. This class can be used to insert qSLP into **CircuitQNN** and **NeuralNetworkClassifier** to train the model like the ones in the qiskit library.
The class qSLP is still missing a parametrized version of the non padded state preparation.

### 09/12/2021
- **temp.py** playground to test **qSLP.py**
Known problems: 
- It happears that some experiments have some difficulties in moving in the loss landscape and often get trapped for many iteration in some local minima. 
- Couldn instantiate a "paddless" parametrized state preparation  

### 13/12/2021
- Created a working class sQLP in **sQLP.py**, it takes as arguments pad and d and can create the two types of circuit. It only stores a parametried version of the state_preparation and the ansatz
- Renamed **qSLP.ipynb** into **old_qSLP.ipynb** since now it is possible to train the model with a qiskit built-in function
- Created **qSLP.ipynb** as a playgruond to test the new class. All tests were successfull and the class qSLP can be indeed used 

### 15/12/2021 
- Created a function that returns ansatz and state preparation depending on parameters. 
 It is now possible to merge qslp with padding with ZZFeatureMap and RealAmplitude
- tmp.ipynb contains the first implementation of a pipeline to get and train different architectures.

### 1/1/2022
Terminated test. Found error in qSLP.py. I will keep the trasining data of QSVM and qNNC and test again qSLP models.

### 2/1/2022
Error found and corrected started another batch of tests.
Partial results:

time represent the lenght of the training in seconds. Loss is the default for **NeuralNetworkClassifier** (_squared errors_)

**MNIST38**

| dataset   | model       |   d |   train_score |   test_score |     time |   final_loss |
|:----------|:------------|----:|--------------:|-------------:|---------:|-------------:|
| MNIST38   | QSVM        |   - |      0.821875 |       0.8    |  796     |       -      |
| MNIST38   | qNNC_v1     |   - |      0.546875 |       0.4625 |  195     |     140.8    | 
| MNIST38   | qNNC_v2     |   - |      0.775    |       0.75   |  243     |     122.319  |
| MNIST38   | qSLP_no_pad |   1 |      0.690625 |       0.6125 |  178     |      130.128 |
| MNIST38   | qSLP_no_pad |   2 |      0.684375 |       0.625  |  564     |      130.105 |
| MNIST38   | qSLP_no_pad |   3 |      0.68125  |       0.625  |  800     |      130.531 |
| MNIST38   | qSLP_pad    |   1 |      0.8625   |       0.825  |  685     |      101.39  |
| MNIST38   | qSLP_pad    |   2 |      0.840625 |       0.8125 |  960     |      104.059 |
| MNIST38   | qSLP_pad    |   3 |      0.8375   |       0.8125 | 1224     |      112.949 |
**MNIST09**

| dataset   | model       |   d |   train_score |   test_score |     time |   final_loss |
|:----------|:------------|----:|--------------:|-------------:|---------:|-------------:|
| MNIST09   | QSVM        |   - |      0.965625 |       0.9375 |  792     |       -      |
| MNIST09   | qNNC_v1     |   - |      0.859375 |       0.8625 |  182     |     109.237  |
| MNIST09   | qNNC_v2     |   - |      0.85     |       0.8375 |  254     |      95.6455 |
| MNIST09   | qSLP_no_pad |   1 |      0.859375 |       0.8125 |  239     |      80.7695 |
| MNIST09   | qSLP_no_pad |   2 |      0.865625 |       0.825  |  550     |      80.3418 |
| MNIST09   | qSLP_no_pad |   3 |      0.88125  |       0.825  |  797     |      81.417  |
| MNIST09   | qSLP_pad    |   1 |      0.89375  |       0.925  |  617     |      82.8154 |
| MNIST09   | qSLP_pad    |   2 |      0.9125   |       0.8875 |  973     |      83.3145 |
| MNIST09   | qSLP_pad    |   3 |      0.921875 |       0.9375 | 1226     |      84.375  |

**iris01**
| dataset   | model       |   d |   train_score |   test_score |     time |   final_loss |
|:----------|:------------|----:|--------------:|-------------:|---------:|-------------:|
| iris01    | QSVM        |   - |        1      |         1    |  47      |       -      |
| iris01    | qNNC_v1     |   - |        0.925  |         0.85 |  41      |      19.5918 |
| iris01    | qNNC_v2     |   - |        0.9625 |         0.95 |  55      |      18.501  |
| iris01    | qSLP_no_pad |   1 |        1      |         1    |  58      |      11.1934 |
| iris01    | qSLP_no_pad |   2 |        1      |         1    | 165      |      11.4238 |
| iris01    | qSLP_no_pad |   3 |        1      |         1    | 213      |      11.6934 |
| iris01    | qSLP_pad    |   1 |        1      |         1    | 183      |      12.7266 |
| iris01    | qSLP_pad    |   2 |        1      |         1    | 275      |      13.2285 |
| iris01    | qSLP_pad    |   3 |        1      |         1    | 338      |      14.3496 |

**iris12**

| dataset   | model       |   d |   train_score |   test_score |     time |   final_loss |
|:----------|:------------|----:|--------------:|-------------:|---------:|-------------:|
| iris12    | QSVM        |   - |        0.9625 |         1    |  48      |       -      |
| iris12    | qNNC_v1     |   - |        0.925  |         1    |  50      |      22.2041 |
| iris12    | qNNC_v2     |   - |        0.8875 |         0.8  |  53      |      26.9326 |
| iris12    | qSLP_no_pad |   1 |        0.8    |         0.65 |  68      |      28.6309 |
| iris12    | qSLP_no_pad |   2 |        0.8    |         0.65 | 159      |      29.1299 |
| iris12    | qSLP_no_pad |   3 |        0.7875 |         0.6  | 212      |      29.1455 |
| iris12    | qSLP_pad    |   1 |        0.825  |         0.6  | 168      |      28.6299 |
| iris12    | qSLP_pad    |   2 |        0.825  |         0.65 | 253      |      28.4258 |
| iris12    | qSLP_pad    |   3 |        0.825  |         0.6  | 317      |      29.5859 |

**iris02**

| dataset   | model       |   d |   train_score |   test_score |     time |   final_loss |
|:----------|:------------|----:|--------------:|-------------:|---------:|-------------:|
| iris02    | QSVM        |   - |        1      |         1    |  48      |       -      |
| iris02    | qNNC_v1     |   - |        1      |         1    |  45      |      17.3779 |
| iris02    | qNNC_v2     |   - |        1      |         0.95 |  51      |      17.9932 |
| iris02    | qSLP_no_pad |   1 |             1 |         1    |  59      |      10.6104 |
| iris02    | qSLP_no_pad |   2 |             1 |         1    | 164      |      10.5508 |
| iris02    | qSLP_no_pad |   3 |             1 |         1    | 211      |      10.7529 |
| iris02    | qSLP_pad    |   1 |             1 |         1    | 186      |      12.3643 |
| iris02    | qSLP_pad    |   2 |             1 |         1    | 251      |      13.834  |
| iris02    | qSLP_pad    |   3 |             1 |         1    | 317      |      17.1045 |


### 6/02/2022
A lot of minor changes to complete the thesis, leaving today the develop branch and starting the more complete research branch

# Creation Research branch. 
### 07/02/2022
Cleaning of some notebooks. Results are moved on the old results folder.

### 8/02/2022
Added two methods to the qSLP class: Train, predict to automatically train the model.
