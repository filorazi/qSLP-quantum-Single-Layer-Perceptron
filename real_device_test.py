from utils.qSLP import qSLP
from qiskit.utils import QuantumInstance
from qiskit import Aer, QuantumCircuit
from utils.data_visualization import *
from utils.Utils_pad import padding
from utils.import_data import get_dataset
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit.circuit.library import RealAmplitudes
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit import QuantumCircuit, execute, BasicAer
import pickle
from utils.Utils import get_params, parity
from sklearn.metrics import accuracy_score
import pandas as pd
import sys

def get_quantum_instance():
    IBMQ.load_account() # Load account from disk
    provider = IBMQ.get_provider(hub='ibm-q')
    small_devices = provider.backends(filters=lambda x: x.configuration().n_qubits == 5
                                    and not x.configuration().simulator
                                    and x.status().operational== True)
    least_busy(small_devices)
    backend = least_busy(small_devices)
    
    # Comment to run on real devices
    # backend = Aer.get_backend('aer_simulator')
    return QuantumInstance(backend, shots=1024)

def main(path_results, path_models, path_save):
    path_res = path_results
    datasets = ["iris01","MNIST09", "MNIST38",  "iris12", "iris02"]
    for dataset in datasets:
        qinstance =  get_quantum_instance()
        X_train, X_test, Y_train, Y_test = get_dataset(dataset)
        X_test_pad = padding(X_test)

        for d in range(1,4):
            # Create model
            model_name = f"pad_qSLP_{d}"
            print(model_name)
            params = get_params(model_name, dataset)
            model = qSLP(d, True)
            qc, sp_par, ansatz_par = model.get_full_circ()

            # Set params
            weights = dict(zip(ansatz_par, params))
            qc = qc.bind_parameters(weights)
            ris = []

            # Execute tests
            for i in range(X_test.shape[0]):
                inp = dict(zip(sp_par, X_test_pad[i]))
                q = qc.bind_parameters(inp)

                res = execute(q, qinstance.backend, shots=1024).result()
                ris.append(res.get_counts())

            # Process and save results
            ris = [int(max(el, key=el.get)) for el in ris]
            acc = accuracy_score(ris, Y_test)
            result = {
                    "model": [model_name],
                    "real_dev_score" : [acc]
            }       
            res = pd.DataFrame(result)
            res.to_csv(path_save, mode = "a", header=False, index = False)



            # Create model 
            model_name = f"sdq_qSLP_{d}"
            print(model_name)
            params = get_params(model_name, dataset)
            model = qSLP(d, False)
            qc, sp_par, ansatz_par = model.get_full_circ()

            # Set params
            weights = dict(zip(ansatz_par, params))
            qc = qc.bind_parameters(weights)
            ris = []

            # Execute circuit
            for i in range(X_test.shape[0]):
                inp = dict(zip(sp_par, X_test[i]))
                q = qc.bind_parameters(inp)
                res = execute(q, qinstance.backend, shots=1024).result()
                ris.append(res.get_counts())

            # Process and save results
            ris = [int(max(el, key=el.get)) for el in ris]
            acc = accuracy_score(ris, Y_test)
            result = {
                    "model": [model_name],
                    "real_dev_score" : [acc]
            }       
            res = pd.DataFrame(result)
            res.to_csv(path_save, mode = "a", header=False, index = False)
   
   
    # Create model qnnC_v1

        model_name = "qNNC_v1"
        print(model_name)
        tot_qubit = 2
        feature_map = ZZFeatureMap(feature_dimension=2,
                                reps=1, entanglement='linear')
        ansatz = RealAmplitudes(2, reps=1)
        interpreter = parity 
        qc = QuantumCircuit(tot_qubit)
        qc.append(feature_map, range(tot_qubit))
        qc.append(ansatz, range(tot_qubit))
        qc.measure_all()

        params = get_params(model_name, dataset)
        weights = dict(zip(ansatz.parameters, params))
        qc = qc.bind_parameters(weights)
        ris = []
        for i in range(X_test.shape[0]):
            weigths = dict(zip(feature_map.parameters, X_test[i]))
            q = qc.bind_parameters(weigths)
            res = execute(q, qinstance.backend, shots=1024).result()
            ris.append(max(res.get_counts(), key=res.get_counts().get).count('1') % 2)
        acc = accuracy_score(ris, Y_test)
        #acc = accuracy_score([max(el, key=el.get).count('1') % 2 for el in ris], Y_test)

        result = {
            "model": [model_name],
            "real_dev_score" : [acc]
        }       
        res = pd.DataFrame(result)
        res.to_csv(path_save, mode = "a", header=False, index = False)


        # Create model qnnC_v2

        model_name = "qNNC_v2"
        print(model_name)
        tot_qubit = 2
        feature_map = ZFeatureMap(feature_dimension=2,
                                reps=1)
        ansatz = RealAmplitudes(2, reps=2)
        interpreter = parity 
        qc = QuantumCircuit(tot_qubit)
        qc.append(feature_map, range(tot_qubit))
        qc.append(ansatz, range(tot_qubit))
        qc.measure_all()

        params = get_params(model_name, dataset)
        weights = dict(zip(ansatz.parameters, params))
        qc = qc.bind_parameters(weights)
        ris = []
        for i in range(X_test.shape[0]):
            weigths = dict(zip(feature_map.parameters, X_test[i]))
            q = qc.bind_parameters(weigths)
            res = execute(q, qinstance.backend, shots=1024).result()
            ris.append(max(res.get_counts(), key=res.get_counts().get).count('1') % 2)
        acc = accuracy_score(ris, Y_test)
        result = {
            "model": [model_name],
            "real_dev_score" : [acc]
        }       
        res = pd.DataFrame(result)
        res.to_csv(path_save, mode = "a", header=False, index = False)



        # Create model QSVC
        model_name = "QSVC"
        print(model_name)
        best_df = pd.read_csv("results/test_simulation/simulated_best.csv")
        best_qsvc = best_df[best_df["model"] == model_name]
        k = best_qsvc[best_qsvc["dataset"] == dataset]["k"].item()
        loaded_model = pickle.load(open(f"results/training/qsvm/{model_name}_{dataset}_{k}.sav", 'rb'))
        rus= loaded_model.predict(X_test)
        acc = accuracy_score(rus, Y_test)
        result = {
            "model": [model_name],
            "real_dev_score" : [acc]
        }       
        res = pd.DataFrame(result)
        res.to_csv(path_save, mode = "a", header=False, index = False)

    columns = [ "model","real_dev_score" ]
    df = pd.read_csv(path_save,names=columns)
    df.to_csv(path_save, index=False)


if __name__ == "__main__":
    #args = sys.argv[1:]
    args = ['results/training/file_result.txt', 'results/training/qsvm/', 'results/test_real/acc_real.txt' ]

    if len(args) != 3:
        raise Exception("Wrong number of arguments, specify: csv file for results,  path to qsvc model save folder, csv file to save loss/accuracy ")
    path_results = args[0]
    path_models = args[1]
    path_save = args[2]
    main(path_results, path_models, path_save)
