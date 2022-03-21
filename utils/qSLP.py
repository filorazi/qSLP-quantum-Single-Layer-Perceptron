import sys
from numpy.random import rand
from sklearn.metrics import log_loss, accuracy_score
sys.path.append('\.')
from utils.Utils_pad import *
from utils.import_data import *
from time import time
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.circuit.library import RawFeatureVector


class qSLP():
    def __init__(self, d, pad, cu = "cu3", sigma = None, seed = None) -> None:
        self.__d = d
        self.__pad = pad
        self.__sigma = sigma
        if cu.lower() == "cu3":
            self.__par_per_gate = 3
        else:
            raise Exception("Feature not yet implemented")
            self.__par_per_gate = 2

        if self.__pad:
            self.__num_ansatz_par = d * self.__par_per_gate*4 + d
            self.__n_sp_par = 5
            self.__full_par =d * 16 + d

            self.__x = ParameterVector("x", self.__n_sp_par)
            self.n_qubit = d + 2
            self.__n_data_qbit = 2
        else:
            self.__num_ansatz_par = d * self.__par_per_gate*2 + d
            self.__full_par =d * 8 + d
            self.n_qubit = d + 1
            self.__n_data_qbit = 1
            self.__n_sp_par = 2

        self.__seed = seed
        self.__par_values = None        
        self.__num_theta = self.__num_ansatz_par - self.__d
        self.__par_ansatz_beta = ParameterVector("beta", self.__d)
        #self.__par_ansatz_theta = ParameterVector("theta", self.__num_ansatz_par - self.__d)
        self.__par_ansatz_theta = ParameterVector("theta", self.__full_par - self.__d)
        self.__circuit = self.get_full_circ()[0]
        np.random.seed(None)
        self.__par_values = rand(self.__num_ansatz_par)


        
    def get_ansatz(self):
        if self.__pad:
            return self.pad_ansatz()
        else:
            return self.sqb_ansatz()

    def pad_ansatz(self):
        data = QuantumRegister(2, "data")
        control = QuantumRegister(self.__d, 'control')
        #c = ClassicalRegister(1)
        qc = QuantumCircuit(control, data )
        
        beta = self.__par_ansatz_beta
        theta = self.__par_ansatz_theta

        for j, b in zip(range(self.__d), beta):
            qc.ry(b, control[j])

        qc.barrier()

        for i in range(0, self.__d):
            qc.cu3(theta[i*12 + 0], theta[i*12 + 1], theta[i*12 + 2], control[i], data[0])
            qc.cu3(theta[i*12 + 3], theta[i*12 + 4], theta[i*12 + 5], control[i], data[1])
            qc.ccx(control[i],data[0], data[1])

            qc.x(control[i])

            qc.cu3(theta[i*12 + 6], theta[i*12 + 7], theta[i*12 + 8], control[i], data[0])
            qc.cu3(theta[i*12 + 9], theta[i*12 + 10], theta[i*12 + 11], control[i], data[1])
            qc.ccx(control[i],data[0], data[1])
            qc.barrier()

        
        qc.compose(self.sigma(), qubits=data, inplace=True)
        qc.barrier()

        return qc

    def sqb_ansatz(self):
        data = QuantumRegister(1, "data")
        control = QuantumRegister(self.__d, 'control')
        beta = self.__par_ansatz_beta
        theta = self.__par_ansatz_theta
        qc = QuantumCircuit(control, data )

        for j, b in zip(range(self.__d), beta):
            qc.ry(b, control[j])

        qc.barrier()

        for i in range(0, self.__d):
            qc.cu3(theta[i*6 +0], theta[i*6 +1], theta[i*6 +2], control[i], data)
            qc.x(control[i])
            qc.cu3(theta[i*6 +3], theta[i*6 +4], theta[i*6 +5], control[i], data)
            qc.barrier()

        qc.compose(self.sigma(), qubits=data, inplace=True)
        qc.barrier()

        return qc

    def get_state_preparation(self):
        if self.__pad:
            return self.pad_state_preparation()
        else:
            return self.sqb_state_preparation()

    def pad_state_preparation(self):
        q = QuantumRegister(2)
        circuit = QuantumCircuit(q)
        circuit.ry(self.__x[0], 0)
        circuit.cx(0, 1)
        circuit.ry(self.__x[1], 1)
        circuit.cx(0, 1)
        circuit.ry(self.__x[2], 1)
        circuit.x(0)
        circuit.cx(0, 1)
        circuit.ry(self.__x[3], 1)
        circuit.cx(0, 1)
        circuit.ry(self.__x[4], 1)
        circuit.x(0)
        qc = QuantumCircuit(2)
        qc.append(circuit, range(2))
        return qc

    def sqb_state_preparation(self):
        qc = QuantumCircuit(1)
        qc.append(RawFeatureVector(2), [0])
        return qc

    def sigma(self):
        data = QuantumRegister(self.__n_data_qbit)
        qc = QuantumCircuit(data)
        # To this day we cannot yet implement non linear operations so we use a I gate as placeholder
        if self.__sigma is None:
            qc.id(data)
        else:
            qc.append(self.__sigma)
        return qc

    def get_interpreter(self):
        if self.__pad:
            return lambda x: 1 if x & 2**(self.__d + 1) else 0
        else:
            return lambda x: 1 if x & 2**(self.__d) else 0
    
    def get_full_circ(self):
        feature_map = self.get_state_preparation()
        ansatz = self.get_ansatz()

        circ = QuantumCircuit(self.n_qubit, 1)
        circ.append(feature_map, range(self.__d, self.n_qubit))
        circ.append(ansatz, range(self.n_qubit))
        circ.measure(self.n_qubit-1,0)
        return circ, feature_map.parameters, ansatz.parameters

    def __process_prob(self, probas):
        return (probas >= 0.5) * 1


    def __circuit_exec(self, inputs, points ,backend):
        t = [a for a in self.__circuit.parameters if a.name[:5] == "theta"]
        theta = {name: val for name,val in zip(t, self.get_full_par(points[-self.__num_theta:]))}
        b = [a for a in self.__circuit.parameters if a.name[:4] == "beta"]
        beta = {name: val for name,val in zip(b, points[:len(b)])}
        x = [a for a in self.__circuit.parameters if a.name[:1] == "x"]
        inputs = {name: val for name,val in zip(x, inputs)}
        theta.update(beta)
        theta.update(inputs)
        qc = self.__circuit.bind_parameters(theta)
        result = execute(qc, backend, shots=2047, seed_transpiler = self.__seed, seed_simulator=self.__seed).result()

        counts = result.get_counts(qc)
        result = np.zeros(2)
        for key in counts:
            result[int(key, 2)] = counts[key]
        result /= 2047
        return result[1]


    def train(self, X, y, optimizer, qantum_instance, warm_start = False, starting_points = None, loss = log_loss):
        if len(X[0]) != self.__n_sp_par:
            raise Exception("Missmatch between inputs shape and number of parameter in state preparation")
        if warm_start:
            if starting_points is None:
                raise Exception("Missing starting points, warm start cannot be executed")
            elif len(starting_points) != self.__num_ansatz_par:
                raise Exception(f"Size missmatch in starting_points: expected {self.__num_ansatz_par}, found {len(starting_points)}")
            else:
                points = starting_points
        elif self.__par_values is None: 
            points = rand(self.__num_ansatz_par)
        else: 
            points = self.__par_values
        call_back_hist =   {"accuracy": [],
                            "loss": []}
        self.__loss = loss
        random.seed(self.__seed)
        np.random.seed(self.__seed)

        def obj_function(params):
            # execute the quantum circuit for the current set of parameters for all the training set
            predictions = [self.__circuit_exec( x, params, qantum_instance.backend) for x in X]
            # return the log_loss
            random.seed(self.__seed)
            np.random.seed(self.__seed)

            a = accuracy_score(y,[self.__process_prob(p) for p in predictions])
            #   print(a, end = " ")
            b = self.__loss(y, predictions)
            call_back_hist["accuracy"].append(a)
            call_back_hist["loss"].append(b)
            return b
        points, _, _ = optimizer.optimize(self.__num_ansatz_par, obj_function, initial_point=points)
        self.__par_values = points
        return call_back_hist

    def predict(self, X, quantum_instance, starting_points = None):
        if starting_points is not None:
            if len(starting_points) != self.__num_ansatz_par:
                raise Exception(f"Size missmatch in starting_points: expected {self.__num_ansatz_par}, found {len(starting_points)}")
            else:
                points = starting_points
        elif self.__par_values is None: 
            raise Exception(f"Model not trained: expected starting point of size {self.__num_ansatz_par}, found None")
        else: 
            points = self.__par_values
        probs_test = [self.__circuit_exec(x, points, quantum_instance.backend) for x in X]

        return [self.__process_prob(p) for p in probs_test]
        
    def save_weights(self, output_file):
        if self.__par_values is None:
            raise Exception("Model not trained: cannot save an empty model")
        np.save(output_file, self.__par_values)
        return
    
    def load_weights(self, input_file):
        val = np.load(input_file)
        if len(val) != self.__num_ansatz_par:
            raise Exception(f"Weights not compatible with model: expected {self.__num_ansatz_par} parameters, found {len(val)}")
        else:
            self.__par_values = val

    def get_weights(self):
        return np.copy(self.__par_values)
        
    def set_weights(self, weights):
        if len(weights) != self.__num_ansatz_par:
            raise Exception(f"Weights not compatible with model: expected {self.__num_ansatz_par} parameters, found {len(weights)}")
        else:
            self.__par_values = weights

    def get_full_par(self, param):
        half_pi_indx = []
        if len(param) != self.__num_theta:
            raise Exception(f"Missmatch beetween number of paramter passed and number of parameter in the circuit. Expected {self.__num_theta} found {len(param)}")
        if self.__par_per_gate < 3:
            half_pi_indx = [a*self.__par_per_gate for a in range(int((len(param))/2))]
            param = np.insert(param, half_pi_indx, np.pi/2)
        
        return param

