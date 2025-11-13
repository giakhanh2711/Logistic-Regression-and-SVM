import numpy as np
from sklearn import preprocessing


def load_data(data_file, train_idx_file, test_idx_file):
    """
    Load train and test data from data_file
    Returns:
        X_train, Y_train, X_test, Y_test
    """

    print("Load files:")
    print(f"\t{data_file}\n\t{train_idx_file}\n\t{test_idx_file}")

    # Load data
    def conv(item):
        return item.split(":")[1] if ":" in item else item

    data = np.loadtxt(data_file, converters=conv)
    print(f"\n Data shape: {data.shape}")
    
    # Load index
    train_idx = np.loadtxt(train_idx_file, dtype="int", delimiter=",")
    test_idx = np.loadtxt(test_idx_file, dtype="int", delimiter=",")
    
    X_train, Y_train = data[train_idx, 1:], data[train_idx, 0]
    X_test, Y_test = data[test_idx, 1:], data[test_idx, 0]
    
    print(f"\nX_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}, Y_train shape: {Y_test.shape}\n")
    return X_train, Y_train, X_test, Y_test


def load_covtype(filename, file_indices_train, file_indices_test):
    data = np.loadtxt(filename, delimiter=",")
    X = data[:, :-1]
    Y =data[:, -1]

    train_indices = np.loadtxt(file_indices_train, dtype="int", delimiter=",")
    test_indices = np.loadtxt(file_indices_test, dtype="int", delimiter=",")

    X_train, Y_train = X[train_indices, :], Y[train_indices]
    X_test, Y_test = X[test_indices, :], Y[test_indices]

    print("Load files:")
    print(f"\t{filename}\n\t{file_indices_train}\n\t{file_indices_test}")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    return X_train, Y_train, X_test, Y_test


def data_preprocessing(X, method:str):
    if method == "normalization":
        X_normalized = preprocessing.normalize(X, norm="l2", axis=1)
        return X_normalized
    
    if method == "rescale":
        minmax_scaler = preprocessing.MinMaxScaler(X) # default range (0, 1)
        X_scaled = minmax_scaler.fit_transform(X)
        return X_scaled

    if method == "standardization":
        standard_scaler = preprocessing.StandardScaler()
        X_standardized = standard_scaler.fit_transform(X=X)
        return X_standardized
    
    print("No available data preprocessing method!")
