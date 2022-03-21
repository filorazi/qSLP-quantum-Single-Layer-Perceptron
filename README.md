# Quantum single layer perceptron
This repository contains the first implementation of a quantum single layer perceptron done through the use of the [MAQA Framework]() and the experimental comparison between the proposed model and soma baseline models.

## Description
The core idea is to leverage the superposition of the data qubits with multiple control qubits to obtain a quantum version of an aggregation of weighted functions. We investigated the performance of various implementations of a quantum single layer perceptron and compared them with three baseline models proposed in the literature: two quantum neural network classifiers and one quantum support vector classifier.

## Repository structure:
- **experiments**: folder with the notebooks able to reproduce the experiments. It contains:
    - **experiments_qSLP.ipynb**: file to reproduce the experiments involving the proposed model. You only need to choose the number of control qubits and the state preparation strategy. If the seed and the starting points are set it produces the results seen in the research
    - **experiments_qNNC.ipynb**: file to reproduce the qnnc models' performance. If the seed and the starting point are set it reproduces the results obtained in the research
    - **experiments_QSVC.ipynb**: file to reproduce the QSVC model performance. If the seed and the starting point are set it reproduces the results obtained in the research
- **results**: a folder containing the results of the testing:
    - **test_real**: results obtained by the best models in a real quantum environment 
    - **test_simulation**: results obtained by the best models in a simulated environment
    - **training**: all the models trained (qSLP, qNNC, QSVC)
- **model_training.py**: script to perform the training and testing on a simulated environment
- **real_device_test.py**: script to perform the testing of the best models on a real environment
- **result_analysis.ipynb**: notebook for the visualization of the performance of the models.

## Usage
The code is organized in three main parts:
- Training and testing on the simulation are done through the *model_triaining.py* script. Run it and on completion, it writes the results on the *results/training* and *results/test_simulation* folders. To run the script simply use 
  `python Variational_Algorithm_qSLP/model_training.py`
- Testing on a real device is done through the *real_device_test.py* script. To run it there should be an [IBMQ token](https://quantum-computing.ibm.com/lab/docs/iql/manage/account/ibmq) in the system to have access to the IBM's quantum devices. Results are stored in the folder *results/test_real*. `python Variational_Algorithm_qSLP/real_device_test.py`
- To run and test the best model of each category refer to the notebooks in the *experimetns_qSLP* folder. There you can choose the specific model of each category to run, obtaining the results shown in the research.

## Installation
To run the code and reproduce the results of the paper, it is recommended to re-create the same testing environment following the procedure below.

*Note: it's assumed Anaconda is installed*
- First clone the repository: 
 `git clone https://github.com/amacaluso/MAQA.git`
- Second, create a conda environment from scratch and install the requirements specified in the requirements.txt file:  
    ```
    # enter the repository
    cd MAQA

    # create an environment with desired dependencies found in the requirements.txt file
    conda env create
    pip install -r requirements.txt
    ```


