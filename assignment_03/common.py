import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
import pandas as pd

# training parameters passed to "model.fit(...)"
batch_size = 1
epochs = 10
optimizer = 'adam'

# The input sequence min and max length that the model is trained on for each output point
min_length = 1
max_length = 10

# load data from files
noisy_data = np.loadtxt('../filter_data/noisy_data.txt', delimiter='\t', dtype=np.float)
smooth_data = np.loadtxt('../filter_data/smooth_data.txt', delimiter='\t', dtype=np.float)

print('noisy_data shape:{}'.format(noisy_data.shape))
print('smooth_data shape:{}'.format(smooth_data.shape))
print('noisy_data first 5 data points:{}'.format(noisy_data[:5]))
print('smooth_data first 5 data points:{}'.format(smooth_data[:5]))

# Create model
def create_fc_model(length):
	##### YOUR MODEL GOES HERE #####
	model = Sequential()
	model.add(Dense(20, activation='relu', input_shape=(length,)))
	model.add(Dense(1))
	# model.add(Dense(1, input_shape=(length,)))
	model.compile(loss='mean_squared_error', optimizer=optimizer)
	return model

def create_rnn_model(length, stateful):
	##### YOUR MODEL GOES HERE #####
	model = Sequential()
	model.add(SimpleRNN(20, stateful=stateful, batch_input_shape=(batch_size, length, 1), activation='relu'))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer=optimizer)
	return model


def create_lstm_model(length, stateful):
	##### YOUR MODEL GOES HERE #####
	model = Sequential()
	model.add(LSTM(20, stateful=stateful, batch_input_shape=(batch_size, length, 1)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer=optimizer)
	return model


def split_data(x, y, ratio=0.8):
	to_train = int(len(x.index) * ratio)
	# tweak to match with batch_size
	to_train -= to_train % batch_size

	x_train = x[:to_train]
	y_train = y[:to_train]
	x_test = x[to_train:]
	y_test = y[to_train:]

	# tweak to match with batch_size
	to_drop = x.shape[0] % batch_size
	if to_drop > 0:
		x_test = x_test[:-1 * to_drop]
		y_test = y_test[:-1 * to_drop]

	# some reshaping
	##### RESHAPE YOUR DATA BASED ON YOUR MODEL #####
	x_train = np.expand_dims(x_train, axis=2)
	x_test = np.expand_dims(x_test, axis=2)

	print('x_train.shape: ', x_train.shape)
	print('y_train.shape: ', y_train.shape)
	print('x_test.shape: ', x_test.shape)
	print('y_test.shape: ', y_test.shape)

	return (x_train, y_train), (x_test, y_test)

def data_preprocess(length):
	# convert numpy arrays to pandas dataframe
	data_input = pd.DataFrame(noisy_data)
	expected_output = pd.DataFrame(smooth_data)

	# when length > 1, arrange input sequences
	if length > 1:
		##### ARRANGE YOUR DATA SEQUENCES #####
		data_input = pd.DataFrame([noisy_data[i:i+length][::-1] for i in range(len(noisy_data) - length+1)])
		expected_output = pd.DataFrame(expected_output[length-1:])

	print('Input shape:', data_input.shape)
	print('Output shape:', expected_output.shape)
	print('Input head: ')
	print(data_input.head())
	print('Output head: ')
	print(expected_output.head())
	print('Input tail: ')
	print(data_input.tail())
	print('Output tail: ')
	print(expected_output.tail())

	return data_input, expected_output