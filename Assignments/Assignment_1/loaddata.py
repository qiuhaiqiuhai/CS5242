import csv
import numpy as np
from sklearn import model_selection

root_path = './Question_2_1/'
x_train_path = root_path + 'x_train.csv'
x_test_path = root_path + 'x_test.csv'
y_train_path = root_path + 'y_train.csv'
y_test_path = root_path + 'y_test.csv'
root_path = './Question_2_2/'
w_path = root_path + 'c/w-'
b_path = root_path + 'c/b-'

def convertToOneHot(input):
    output = [0,0,0,0]
    output[int(input)] = 1
    return output

def read_file(file, isConvertToOneHot=False, withHeader=False):
    output = []
    with open(file) as csvfile:
        rows = csv.reader(csvfile)
        if isConvertToOneHot:
            for row in rows:
                output.append(convertToOneHot(row[0]))
        elif withHeader:
            for row in rows:
                output.append(row[1:])
        else:
            for row in rows:
                output.append(row)

    return output

def load_training_data(val_ratio = 0.4):

    X_train = np.asarray(read_file(x_train_path), dtype=int)
    X_test = np.asarray(read_file(x_test_path), dtype=int)
    T_train = np.asarray(read_file(y_train_path, isConvertToOneHot=True))
    T_test = np.asarray(read_file(y_test_path, isConvertToOneHot=True))

    X_validation, X_test, T_validation, T_test = model_selection.train_test_split(
        X_test, T_test, test_size=val_ratio)

    return (X_train, T_train, X_validation, T_validation, X_test, T_test)

def load_WB_data(module_name, layer_sizes):
    W_raw = read_file(w_path + module_name +'.csv', withHeader=True)
    B_raw = read_file(b_path + module_name +'.csv', withHeader=True)
    W, B = [], []

    cur = 0
    for i in range(len(layer_sizes)-1):
        W.append(np.asarray(W_raw[cur:cur+layer_sizes[i]], dtype=np.float32))
        B.append(np.asarray(B_raw[i], dtype=np.float32))
        cur+=layer_sizes[i]

    return (W, B)