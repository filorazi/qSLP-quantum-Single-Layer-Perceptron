
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
import random

#
# def load_iris(fraction=1, plot=True, type=1):
#
#     if type==1:
#         df = np.loadtxt("data/iris_classes1and2_scaled.txt")
#     else:
#         df = np.loadtxt("data/iris_classes2and3_scaled.txt")
#
#     df = pd.DataFrame(df)
#     df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'Y']
#
#     df = df.sample(frac=fraction)
#
#     X = df.iloc[:, 0:2].values
#     Y = (df.iloc[:, -1] + 1) / 2
#     Y = Y.to_numpy()
#
#     if plot:
#         #sns.set_theme(style="ticks")
#         sns.pairplot(df.iloc[:, [0,1,4]], hue="Y", diag_kind='hist')
#         plt.show()
#     return X,Y
def get_dataset(dataset):
    if dataset == "MNIST09":
        X_train, X_test, Y_train, Y_test = load_MNIST(n=200, seed=123, test_size=.2, class0=0, class1=9, n_comp = 2)
    elif dataset == "MNIST38":
        X_train, X_test, Y_train, Y_test = load_MNIST(n=200, seed=123, test_size=.2, class0=3, class1=8, n_comp = 2)
    elif dataset == "iris12":
        X,Y = load_iris(class_to_del=0)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=111, test_size=0.2)
    elif dataset == "iris01":
        X,Y = load_iris(class_to_del=2)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=111, test_size=0.2)
    elif dataset == "iris02":
        X,Y = load_iris(class_to_del=1)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=111, test_size=0.2)
    else:
        print("We don't have that dataset yet, come back later")
    return X_train, X_test, Y_train, Y_test 



def load_bivariate_gaussian(n_train=20, plot=True):
    X, Y = datasets.make_blobs(n_samples=n_train, centers=[[0.3, 0.7], [0.7, 0.3]], # centers=[[0.3, 0.8], [0.7, 0.3]]
                               n_features=2, center_box=(0, 1), cluster_std=0.1,
                               random_state=543)
    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df['Y'] = Y
    if plot:
        #sns.set_theme(style="ticks")
        sns.pairplot(df, hue="Y", diag_kind='hist')
        plt.show()
    return X,Y



def load_parity(plot=True):
    df = np.loadtxt("data/parity.txt")
    df = pd.DataFrame(df)
    df.columns = ['X' + str(i) for i in range(df.shape[1] - 1)] + ['Y']
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values
    #Y = Y * 2 - np.ones(len(Y))  # shift label from {0, 1} to {-1, 1}

    if plot:
        #sns.set_theme(style="ticks")
        sns.pairplot(df, hue="Y", diag_kind='hist')
        plt.show()
    return X,Y


def load_moon(fraction=1, plot=True, type=1):

    df = np.loadtxt("data/moons.txt")

    df = pd.DataFrame(df)
    df.columns = ['X' + str(i) for i in range(df.shape[1] - 1)] + ['Y']

    df = df.sample(frac=fraction)

    X = df.iloc[:, 0:2].values
    Y = df.iloc[:, 2].values

    if plot:
        #sns.set_theme(style="ticks")
        sns.pairplot(df, hue="Y", diag_kind='hist')
        plt.show()
    return X,Y


# load MNIST dataset
def load_MNIST(n=100, seed=123, test_size=.1, class0=0, class1=9, n_comp = 2):
    
    train_size = 1-test_size
    from tensorflow.keras.datasets import mnist
    (x_raw, y_raw), _ = mnist.load_data()


    ix = np.where((y_raw == class0) | (y_raw == class1)) 
    x_raw, y_raw =x_raw[ix], y_raw[ix]

    np.random.seed(seed)

    mask = np.hstack([np.random.choice(np.where(y_raw == l)[0], n, replace=False)
                          for l in np.unique(y_raw)])
    random.seed(seed)

    random.shuffle(mask)
    x_raw, y_raw =x_raw[mask], y_raw[mask]

    # Random splitting of dataset in training and test
    num_data = len(y_raw)
    num_train = int(train_size * num_data)
    np.random.seed(seed)

    index = np.random.permutation(range(num_data))

    # Training set
    X_train = x_raw[index[:num_train]]
    Y_train = y_raw[index[:num_train]]

    # Test set
    X_test = x_raw[index[num_train:]]
    Y_test = y_raw[index[num_train:]]

    ncol = x_raw.shape[1]*x_raw.shape[2]
    x_flat = X_train.reshape(-1,ncol)

    # Rename the columns
    feat_cols = ['pixel'+str(i) for i in range(x_flat.shape[1])]

    # construction of the pandas dataframe
    df_flat = pd.DataFrame(x_flat,columns=feat_cols)
    df_flat['Y'] = Y_train

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Two principal components are considered
    pca = PCA(n_components=n_comp) 

    # Application of the pca to the dataset
    principalComponents = pca.fit_transform(x_flat) 
    
    total_var = 100 * np.sum(pca.explained_variance_ratio_)
    print(f'{total_var:.3}% of total variance is explained by 2 principal components')

    # We create a new dataset where the images are represented by a two-dimensional vector
    # that corresponds to the first two principal components
    # cols=['Component '+str(i+1) for i in range(principalComponents.shape[1])]
    # data_pca = pd.DataFrame(data = principalComponents, 
    #                         columns = cols)

    # Append the target variable to the new dataset
    # data_pca['Y'] = df_flat.iloc[:,-1:].to_numpy()


    # Extract the new feature as numpy array
    x_pca = principalComponents # data_pca[cols].to_numpy()

    MAX=np.max(x_pca)
    MIN=np.min(x_pca)

    # Rescaleing of the values of the features
    X_train = (x_pca-MIN)/(MAX-MIN)
    Y_train = (Y_train-class0)/(class1-class0)


    # We define our training set, that will be the input of our QML model
    # X_train = X.copy()
    # Y_train = 

    X_test = pca.transform(X_test.reshape(-1,ncol))
    X_test = (X_test-MIN)/(MAX-MIN)
    Y_test = (Y_test-class0)/(class1-class0)
    
    return X_train, X_test, Y_train , Y_test


def load_breast(n=100, seed=123):
    
    from sklearn.datasets import load_breast_cancer
    from sklearn.decomposition import PCA

    # load dataset
    breast_cancer = load_breast_cancer()
    
    raw_data = breast_cancer.data
    
    # normalized data
    # normalized_data = StandardScaler().fit_transform(raw_data)
    
    
    # initialize pca with 2 components
    pca = PCA(n_components=2)

    # fit data
    pca_data = pca.fit_transform(raw_data)
    # Variance explained by principal components
    print(pca.explained_variance_ratio_)
    # [0.44272026 0.18971182]

    # Total Variance explained by principal components
    total_var = 100 * np.sum(pca.explained_variance_ratio_)
    print(f'{total_var:.3}% of total variance is explained by 2 principal components')
    # 63.2% of total variance is explained by 2 principal components


    # Create dataframe 
    pca_df = pd.DataFrame(np.vstack((pca_data.T, breast_cancer.target)).T,
                          columns = ['x1', 'x2', 'Y'])


    # Replace 0 with Malignant and 1 with Benign
    pca_df['Y'].replace(0.0, 'Malignant',inplace=True)
    pca_df['Y'].replace(1.0, 'Benign',inplace=True)
    
    df = pca_df
    
    g = df.groupby('Y')
    df = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
    
    x_raw=df.iloc[:,0:2].to_numpy()
    Y=df.iloc[:,2].to_numpy()
    y_raw=np.where(Y=='Benign', 0, 1)
    
    
    MAX=np.max(x_raw)
    MIN=np.min(x_raw)

    # Rescaleing of the values of the features
    x_raw = (x_raw-MIN)/(MAX-MIN)

    
    mask = np.hstack([np.random.choice(np.where(y_raw == l)[0], n, replace=False)
                          for l in np.unique(y_raw)])
    random.shuffle(mask)
    x_raw, y_raw =x_raw[mask], y_raw[mask]
    
    return x_raw, y_raw


def load_iris(class_to_del=0):

    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA

    # load dataset
    iris = load_iris()
    ixs = iris.target!=class_to_del

    raw_data = iris.data[ixs]
    raw_target = iris.target[ixs]

    # initialize pca with 2 components
    pca = PCA(n_components=2)

    # fit data
    pca_data = pca.fit_transform(raw_data)
    # Variance explained by principal components
    print(pca.explained_variance_ratio_)
    # [0.44272026 0.18971182]

    # Total Variance explained by principal components
    total_var = 100 * np.sum(pca.explained_variance_ratio_)
    print(f'{total_var:.3}% of total variance is explained by 2 principal components')
    # 63.2% of total variance is explained by 2 principal components


    # Create dataframe
    pca_df = pd.DataFrame(np.vstack((pca_data.T, raw_target)).T,
                          columns=['x1', 'x2', 'Y'])

    df = pca_df

    x_raw = df.iloc[:, 0:2].to_numpy()
    Y = df.iloc[:, 2].to_numpy()
    y_raw = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))

    MAX = np.max(x_raw)
    MIN = np.min(x_raw)

    # Rescaleing of the values of the features
    x_raw = (x_raw - MIN) / (MAX - MIN)
    return x_raw, y_raw

###################################################################
###################################################################


# def load_blood_transfusion(plot=True):
#     df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data')
#     df.columns = ['R', 'F', 'M', 'T', 'Y']

#     Y = df.Y
#     X = df.drop(labels='Y', axis=1).values #returns a numpy array
#     min_max_scaler = preprocessing.StandardScaler()
#     X_scaled = min_max_scaler.fit_transform(X)


#     pca = PCA(n_components=2)
#     pca.fit(X_scaled)
#     x_new = pca.fit_transform(X)
#     print('Explained variance of the 2 firts components:',
#           pca.explained_variance_ratio_)
#     df = pd.DataFrame(x_new, columns=['x1', 'x2'])
#     df['Y'] = Y

#     if plot:
#         sns.set_theme(style="ticks")
#         sns.pairplot(df, hue="Y")
#         plt.show()
#     return df


# def load_banknote_authentication(plot=True, fraction=.1):
#     df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt')
#     df.columns = ['X'+ str(i) for i in range(df.shape[1]-1)] + ['Y']

#     g = df.groupby('Y')
#     df = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
#     df = df.sample(frac=fraction)

#     Y = df.Y.to_numpy()
#     X = df.drop(labels='Y', axis=1).values #returns a numpy array
#     min_max_scaler = preprocessing.StandardScaler()
#     X_scaled = min_max_scaler.fit_transform(X)


#     pca = PCA(n_components=2)
#     pca.fit(X_scaled)
#     x_new = pca.fit_transform(X)
#     x_new_scaled = min_max_scaler.fit_transform(x_new)
#     print('Explained variance of the 2 firts components:',
#           pca.explained_variance_ratio_)
#     df = pd.DataFrame(x_new_scaled, columns=['x1', 'x2'])
#     df['Y'] = Y

#     if plot:
#         sns.set_theme(style="ticks")
#         sns.pairplot(df, hue="Y")
#         plt.show()
#     return x_new_scaled, Y


# def load_haberman(plot=True):
#     df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data')
#     df.columns = ['age', 'year', 'num_pos', 'Y']
#     df.Y.value_counts()

#     Y = df.Y - 1
#     X = df.drop(labels='Y', axis=1).values #returns a numpy array
#     min_max_scaler = preprocessing.StandardScaler()
#     X_scaled = min_max_scaler.fit_transform(X)


#     pca = PCA(n_components=2)
#     pca.fit(X_scaled)
#     x_new = pca.fit_transform(X)
#     print('Explained variance of the 2 firts components:',
#           pca.explained_variance_ratio_)
#     df = pd.DataFrame(x_new, columns=['x1', 'x2'])
#     df['Y'] = Y

#     if plot:
#         sns.set_theme(style="ticks")
#         sns.pairplot(df, hue="Y")
#         plt.show()
#     return df
