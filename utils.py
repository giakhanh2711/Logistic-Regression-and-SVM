import numpy as np

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

