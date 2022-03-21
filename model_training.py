import pandas as pd
import pickle
from time import time

from utils.Utils import *
from utils.Utils_pad import padding
from utils.data_visualization import *
from utils.import_data import get_dataset
from utils.qSLP import qSLP

from sklearn.metrics import accuracy_score

from qiskit.utils import QuantumInstance
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit.circuit.library import RealAmplitudes
from qiskit import Aer, QuantumCircuit

seed = 42


def run_qnnc1(x_train, y_train, x_test, y_test, qinstance, optimizer, dataset,k):
    # utils
    tot_qubit = 2
    output_shape = 2
    objective_func_vals = []
    def callback_values(weights, obj_func_eval):
        objective_func_vals.append(obj_func_eval)

    # model
    feature_map = ZZFeatureMap(feature_dimension=2,
                            reps=1, entanglement='linear')
    ansatz = RealAmplitudes(2, reps=1)
    interpreter = parity 
    qc = QuantumCircuit(tot_qubit)
    qc.append(feature_map, range(tot_qubit))
    qc.append(ansatz, range(tot_qubit))
    np.random.seed()

    starting_points = np.random.rand(ansatz.num_parameters)
    circuit_qnn = CircuitQNN(circuit=qc,
                            input_params=feature_map.parameters,
                            weight_params=ansatz.parameters,
                            interpret=interpreter,
                            output_shape=output_shape,
                            quantum_instance=qinstance)

    circuit_classifier = NeuralNetworkClassifier(neural_network=circuit_qnn,
                                            optimizer=optimizer,
                                            callback=callback_values, 
                                            warm_start=True,
                                            initial_point = starting_points )


    # train
    start_time = time()
    circuit_classifier.fit(x_train, y_train)
    end_time = time()

    # save
    ending_points = circuit_classifier._fit_result[0]
    train_score = circuit_classifier.score(x_train, y_train)
    test_score = circuit_classifier.score(x_test, y_test)

    result = {
        "dataset": [dataset],
        "model": ["qNNC_v1"],
        "train_score": [train_score],
        "test_score": [test_score],
        "time": [end_time-start_time],
        "final_train_loss": [objective_func_vals[-1]],
        "starting_points": [starting_points],
        "ending_points":[ending_points.tolist()],
        "k": [k]
    }
    la = {
        "dataset": [dataset],
        "model": [f"qNNC_v1"],
        "k": [k],
        "loss": [objective_func_vals],
        "accuracy": [0]
        }

    return result, la



def run_qnnc2(x_train, y_train, x_test, y_test, qinstance, optimizer, dataset,k):

    # utils
    objective_func_vals = []
    def callback_values(weights, obj_func_eval):
        objective_func_vals.append(obj_func_eval)

    tot_qubit = 2
    output_shape = 2

    #model
    feature_map = ZFeatureMap(feature_dimension=2,
                            reps=1)

    ansatz = RealAmplitudes(2, reps=2)
    interpreter = parity 
    qc = QuantumCircuit(tot_qubit)
    qc.append(feature_map, range(tot_qubit))
    qc.append(ansatz, range(tot_qubit))
    np.random.seed()

    starting_points = np.random.rand(ansatz.num_parameters)

    circuit_qnn = CircuitQNN(circuit=qc,
                            input_params=feature_map.parameters,
                            weight_params=ansatz.parameters,
                            interpret=interpreter,
                            output_shape=output_shape,
                            quantum_instance=qinstance)

    circuit_classifier = NeuralNetworkClassifier(neural_network=circuit_qnn,
                                            optimizer=optimizer,
                                            callback=callback_values,
                                            warm_start=True,
                                            initial_point = starting_points )

    #train
    start_time = time()
    circuit_classifier.fit(x_train, y_train)
    end_time = time()

    #save
    ending_points = circuit_classifier._fit_result[0]
    train_score = circuit_classifier.score(x_train, y_train)
    test_score = circuit_classifier.score(x_test, y_test)

    result = {
        "dataset": [dataset],
        "model": ["qNNC_v2"],
        "train_score": [train_score],
        "test_score": [test_score],
        "time": [end_time-start_time],
        "final_train_loss": [objective_func_vals[-1]],
        "starting_points": [starting_points],
        "ending_points":[ending_points.tolist()],
        "k": [k]
    }
    la = {
        "dataset": [dataset],
        "model": [f"qNNC_v2"],
        "k": [k],
        "loss": [objective_func_vals],
        "accuracy": [0]
        }

    return result, la


def run_qsvc(x_train, y_train, x_test, y_test, qinstance, optimizer,dataset,k, path):
    # model
    feature_map = ZZFeatureMap(feature_dimension=2, entanglement='linear')
    kernel = QuantumKernel(feature_map=feature_map, quantum_instance=qinstance)
    qsvc = QSVC(quantum_kernel=kernel)
    
    #training
    start_time = time()
    qsvc = qsvc.fit(x_train, y_train)
    end_time = time()

    #save
    train_score = qsvc.score(x_train, y_train)
    test_score = qsvc.score(x_test, y_test)

    result = {
        "dataset": [dataset],
        "model": [f"QSVC"],
        "train_score": [train_score],
        "test_score": [test_score],
        "time": [end_time-start_time],
        "final_train_loss": [0],
        "starting_points": [["0"]],
        "ending_points":[["0"]],
        "k": [k]
    }
    filename = path + f"QSVC_{dataset}_{k}.sav"
    pickle.dump(qsvc, open(filename, 'wb'))
    return result


def main(path_res, path_models, path_la):
    datasets = ["iris01","MNIST09", "MNIST38",  "iris12", "iris02"]
    debug = True
    max_iter = 150
    optimizer = COBYLA(maxiter=max_iter, tol=0.01, disp=False)
    qinstance = QuantumInstance(Aer.get_backend('aer_simulator'),seed_simulator=seed,seed_transpiler=seed, shots=1024)
    qinstance.backend.set_option("seed_simulator", seed)

    for k in range(10): 
        print("="*50 +"\n" +"="*50 + "\n" + f"iteration = {k}")
        for dataset in datasets:
            X_train, X_test, Y_train, Y_test = get_dataset(dataset)
            X_train_pad = padding(X_train)
            X_test_pad = padding(X_test)
            
            for d in range(1,4):
                ## sdq model

                model = qSLP(d,False,seed=seed)
                starting_points = model.get_weights()

                ## train
                start_time = time()
                hist = model.train(X_train,Y_train, optimizer, qinstance)
                end_time = time()

                ## save
                ending_points = model.get_weights()
                train_acc = accuracy_score(Y_train,model.predict(X_train, qinstance))
                test_acc = accuracy_score(Y_test,model.predict(X_test, qinstance))
                timer = end_time - start_time
                if debug:
                    print(f"model sdq qSLP d = {d}:\n\t\
                            training time = {timer},\n\t\
                            train accuracy = {train_acc},\n\t\
                            test accuracy = {test_acc},\n")
                
                result = {
                    "dataset": [dataset],
                    "model": [f"sdq_qSLP_{d}"],
                    "train_score": [train_acc],
                    "test_score": [test_acc],
                    "time": [timer],
                    "final_train_loss": [hist["loss"][-1]],
                    "starting_points": [starting_points],
                    "ending_points":[ending_points],
                    "k": [k]
                }
                la = {
                    "dataset": [dataset],
                    "model": [f"sdq_qSLP_{d}"],
                    "k": [k],
                    "loss": [hist["loss"]],
                    "accuracy": [hist["accuracy"]]
                }

                #pad model
                model = qSLP(d,True,seed=seed)
                starting_points = model.get_weights()

                ##Train
                start_time = time()
                hist = model.train(X_train_pad,Y_train, optimizer, qinstance)
                end_time = time()

                ##Save
                ending_points = model.get_weights()
                train_acc = accuracy_score(Y_train,model.predict(X_train_pad, qinstance))
                test_acc = accuracy_score(Y_test,model.predict(X_test_pad, qinstance))
                timer = end_time - start_time
                if debug:
                    print(f"model pad qSLP d = {d}:\n\t\
                            training time = {timer},\n\t\
                            train accuracy = {train_acc},\n\t\
                            test accuracy = {test_acc},\n")
                
                result["dataset"].append(dataset)
                result["model"].append(f"pad_qSLP_{d}")
                result["train_score"].append(train_acc)
                result["test_score"].append(test_acc)
                result["time"].append(timer)
                result["final_train_loss"].append(hist["loss"][-1])
                result["starting_points"].append(starting_points)
                result["ending_points"].append(ending_points)
                result["k"].append(k)
                la["dataset"].append(dataset)
                la["model"].append(f"pad_qSLP_{d}")
                la["loss"].append(hist["loss"])
                la["k"].append(k)
                la["accuracy"].append(hist["accuracy"])
                LA = pd.DataFrame(la)
                res = pd.DataFrame(result)
                res.to_csv(path_res, mode = "a", header=False, index=False)
                LA.to_csv(path_la, mode = "a", header=False, index=False)


            #qnnc1
            result, la = run_qnnc1(X_train, Y_train, X_test, Y_test, qinstance, optimizer, dataset,k)
            print(f'model qnnc1:\n\t\
                    training time = {result["time"]},\n\t\
                    train accuracy = {result["train_score"]},\n\t\
                    test accuracy = {result["test_score"]},\n')
            LA = pd.DataFrame(la)
            res = pd.DataFrame(result)
            res.to_csv(path_res, mode = "a", header=False, index=False)
            LA.to_csv(path_la, mode = "a", header=False, index=False)


            result, la = run_qnnc2(X_train, Y_train, X_test, Y_test, qinstance, optimizer, dataset,k)
            print(f'model qnnc2:\n\t\
                    training time = {result["time"]},\n\t\
                    train accuracy = {result["train_score"]},\n\t\
                    test accuracy = {result["test_score"]},\n')
            LA = pd.DataFrame(la)
            res = pd.DataFrame(result)
            res.to_csv(path_res, mode = "a", header=False, index=False)
            LA.to_csv(path_la, mode = "a", header=False, index=False)

            result = run_qsvc(X_train, Y_train, X_test, Y_test, qinstance, optimizer,dataset,k,path_models)
            print(f'model qsvc:\n\t\
                    training time = {result["time"]},\n\t\
                    train accuracy = {result["train_score"]},\n\t\
                    test accuracy = {result["test_score"]},\n')

            #LA = pd.DataFrame(la)
            res = pd.DataFrame(result)
            res.to_csv(path_res, mode = "a", header=False, index=False)
            #LA.to_csv(path_la, mode = "a", header=False, index=False)

    # find best performing models
    f_train_res = "results/test_simulation/simulated_best.csv"
    columns = [ "dataset","model","train_score","test_score","time","final_train_loss","starting_points","ending_points","k"]
    df = pd.read_csv(path_res,names=columns).drop(["starting_points", "ending_points"], axis=1)
    idx = df.groupby(["dataset", "model"])['train_score'].idxmax()
    df = df.loc[idx]
    df.to_csv(f_train_res, index=False)
    # add columns to file results
    df = pd.read_csv(path_res,names=columns)
    df.to_csv(path_res, index=False)
    # add colums to file_la
    columns = ["dataset","model","k","loss","accuracy"]
    df = pd.read_csv(path_la,names=columns)
    df.to_csv(path_la, index=False)


if __name__ == "__main__":
    args = ['results/training/file_result.txt', 'results/training/qsvm/', 'results/training/file_loss.txt' ]
    if len(args) != 3:
        raise Exception("Wrong number of arguments, specify: csv file for results,  path to qsvc model save folder, csv file to save loss/accuracy ")
    path_res = args[0]
    path_model = args[1]
    path_la = args[2]
    main(path_res, path_model, path_la)
