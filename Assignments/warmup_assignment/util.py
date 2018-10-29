import csv
import numpy as np

def read_data(file_name):
    data_set = []
    with open(file_name, newline='') as csvfile:
        data_file = csv.reader(csvfile)
        for row in data_file:
            data_set.append(list(float(x) for x in list(row)))
    return np.array(data_set)
