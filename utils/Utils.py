# Copyright 2020 Antonio Macaluso
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
import random



from qiskit import *
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import BasicAer, execute, IBMQ, Aer
from qiskit.circuit import Gate
from qiskit.quantum_info.operators import Operator
from qiskit.aqua.components.optimizers import ADAM, CG, AQGD




def create_dir(path):
    if not os.path.exists(path):
        print('The directory', path, 'does not exist and will be created')
        os.makedirs(path)
    else:
        print('The directory', path, ' already exists')




def binary_crossentropy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        # print(l,p)
        loss = loss - l * np.log(np.max([p, 1e-8]))

    loss = loss / len(labels)
    return loss

def parity(x):
    return '{:b}'.format(x).count('1') % 2

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


def accuracy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss





def multivariateGrid(col_x, col_y, col_k, df, col_color=None,
                     scatter_alpha=0.5):
    import numpy as np
    import seaborn as sns
    from matplotlib import pyplot as plt


    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            plt.scatter(*args, **kwargs)

        return scatter

    g = sns.JointGrid(
        x=col_x,
        y=col_y,
        data=df
    )
    color = None
    legends = []
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        # if col_color:
        #     colors_data = np.unique(df[col_color])
        # else:
        #     colors_data = ["or_blue", "or_peru"]

        if col_color:
            color = df_group[col_color].tolist()[0]
        g.plot_joint(
            colored_scatter(df_group[col_x], df_group[col_y], color),
        )
        sns.distplot(
            df_group[col_x].values,
            ax=g.ax_marg_x,
            color=color,
        )
        sns.distplot(
            df_group[col_y].values,
            ax=g.ax_marg_y,
            color=color,
            vertical=True
        )
    # Do also global Hist:
    sns.distplot(
        df[col_x].values,
        ax=g.ax_marg_x,
        color='grey'
    )
    sns.distplot(
        df[col_y].values.ravel(),
        ax=g.ax_marg_y,
        color='grey',
        vertical=True
    )
    plt.tight_layout()
    plt.xlabel(r'$x_1$', fontsize=20)
    plt.ylabel(r'$x_2$', fontsize=20, rotation=0)
    plt.legend(legends, fontsize=18, loc='lower left')
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    colors_data = np.unique(df[col_color])
    plt.savefig('Data_{}_{}.png'.format(
        colors_data[0][:2], colors_data[1][:2]), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def get_params(model, dataset, target = "ending_points",  input_file = "results/training/file_result.txt"):
    df = pd.read_csv(input_file,header=0)
    proc = lambda x: eval(x.replace("\n", "").replace("[ ", "[").replace("\r", "").replace("   ", " " ).replace("  ", " ").replace(" ", ",").replace(",,", ","))
    idx = df.groupby(["dataset", "model"])['train_score'].idxmax()
    df = df.loc[idx]
    df["starting_points"] = df["starting_points"].apply(proc)   
    df["ending_points"] = df["ending_points"].apply(proc)   
    df = df.loc[df['dataset']==dataset]
    df = df.loc[df['model']==model]
    return df[target].values[0]
