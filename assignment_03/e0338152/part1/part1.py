from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from keras import optimizers, losses

def plot_costs(history, title):
    nb_of_epochs = len(history['loss'])
    epochs_x = np.linspace(1, nb_of_epochs, num=nb_of_epochs)

    plt.plot(epochs_x, history['loss'], 'r-', linewidth=1.5, label='Train')
    # plt.plot(epochs_x, val_costs, 'b-', linewidth=1.5, label='Validation')
    plt.plot(epochs_x, history['val_loss'], 'g-', linewidth=1.5, label='Test')


    # best_val = np.min(val_costs);
    # best_epoch = np.argmin(val_costs) + 1;
    plt.xlabel('{} epochs'.format(nb_of_epochs))
    plt.ylabel('MSE')
    plt.title(title)
    plt.legend()

    # plt.axhline(y=best_val, color="blue", linestyle="--", linewidth=0.5)
    # plt.axvline(x=best_epoch, color="blue", linestyle="--", linewidth=0.5)

    ceil = np.amax([history['loss'], history['val_loss']])
    plt.xlim(0, 10)
    plt.ylim(0)
    plt.grid()
    plt.show()

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

# List to keep track of root mean square error for different length input sequences
fc_rmse_list=list()

for length in range(min_length,max_length+1):

	print("*" * 33)
	print("INPUT DIMENSION:{}".format(length))
	print("*" * 33)

	data_input, expected_output = data_preprocess(length)

	print('data_input length:{}'.format(len(data_input.index)) )

	# Split training and test data: use first 80% of data points as training and remaining as test
	(x_train, y_train), (x_test, y_test) = split_data(data_input, expected_output)

	# Create the model
	print('Creating Fully-Connected Model...')
	model_fc = create_fc_model(length)

	# Train the model
	print('Training')
	##### TRAIN YOUR MODEL #####
	# model_fc.load_weights('../trained_models/fc_model_weights_length_%d_trained.h5'%length, by_name=True)
	# model_fc.fit(batch_size=batch_size, x=x_train, y=y_train, epochs=epochs, verbose=2)
	history = model_fc.fit(x_train[:,:,0], y_train, epochs=epochs, batch_size=batch_size, verbose=2,
					 validation_data=(x_test[:,:,0], y_test), shuffle=False)

	# Plot and save loss curves of training and test set vs iteration in the same graph
	#### PLOT AND SAVE LOSS CURVES #####
	plot_costs(history.history, 'Loss of FC Model when length=%d'%length)

	# Save your model weights with following convention:
	# For example length 1 input sequences model filename
	# fc_model_weights_length_1.h5
	##### SAVE MODEL WEIGHTS #####
	model_fc.save_weights('fc_model_weights_length_%d.h5'%length)

	# Predict 
	print('Predicting')
	##### PREDICT #####
	predicted_fc = model_fc.predict(x_test[:,:,0], batch_size=None, verbose=0, steps=None)
	# plot.plot_data([y_test.values, predicted_fc])

	##### CALCULATE RMSE #####
	fc_rmse = np.sqrt(((predicted_fc - y_test) ** 2).mean())
	fc_rmse_list.append(fc_rmse)

	# print('tsteps:{}'.format(tsteps))
	print('length:{}'.format(length))
	print('Fully-Connected RMSE:{}'.format( fc_rmse ))

# save your rmse values for different length input sequence models:
filename = 'fc_model_rmse_values.txt'
np.savetxt(filename, np.array(fc_rmse_list), fmt='%.6f', delimiter='\t')

print("#" * 33)
print('Plotting Results')
print("#" * 33)

# Plot and save rmse vs Input Length
plt.figure()
plt.plot( np.arange(min_length,max_length+1), fc_rmse_list, c='black', label='FC')
plt.title('RMSE vs Input Length in Test Set')
plt.xlabel('length of input sequences')
plt.ylabel('rmse')
plt.legend()
plt.grid()
plt.show()


