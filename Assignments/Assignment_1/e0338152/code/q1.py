import csv
import numpy as np

folder_name = ['a', 'b', 'c', 'd', 'e']
weight_input_suffix = '_w.csv'
bias_input_suffix = '_b.csv'
layers = 3
nodes = 5
dt = np.float32

def read_file(name):
    l = list()
    with open(name) as f:
        reader = csv.reader(f)
        for row in reader:
            l.append(row[1:])
    return l

def read_weight(name):
    l = read_file(name)
    weight = list()
    for i in range(0, len(l), nodes):
        weight.append(np.array(l[i:i+nodes], dtype = dt))
    return weight


def read_bias(name):
    l = read_file(name)
    bias = list()

    for i in l:
        bias.append(np.array(i, dtype=dt))
    return bias

for name in folder_name:
    weight_input = read_weight('./' + name+ '/' + name + weight_input_suffix)
    bias_input = read_bias('./' + name+ '/' + name + bias_input_suffix)

    weight_output = np.identity(nodes, dtype=dt);
    bias_output = np.zeros(nodes, dtype=dt);

    for i in range(layers):
        weight_output = np.dot(weight_output, weight_input[i])
        bias_output = np.dot(bias_output, weight_input[i]) + bias_input[i];

    np.savetxt(name + "-w.csv", weight_output, delimiter=",", fmt='%.16f')
    np.savetxt(name + "-b.csv", bias_output[None], delimiter=",", fmt='%.16f')
