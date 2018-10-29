from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from common import *


filenames = ['../trained_models/fc_model_weights_length_%d_trained.h5',
			 '../trained_models/rnn_stateful_model_weights_length_%d_trained.h5',
			 '../trained_models/rnn_stateless_model_weights_length_%d_trained.h5',
			 '../trained_models/lstm_stateful_model_weights_length_%d_trained.h5',
			 '../trained_models/lstm_stateless_model_weights_length_%d_trained.h5']

print('noisy_data shape:{}'.format(noisy_data.shape))
print('smooth_data shape:{}'.format(smooth_data.shape))
print('noisy_data first 5 data points:{}'.format(noisy_data[:5]))
print('smooth_data first 5 data points:{}'.format(smooth_data[:5]))


# List to keep track of root mean square error for different length input sequences
fc_rmse_list=list()
rnn_stateful_rmse_list=list()
rnn_stateless_rmse_list=list()
lstm_stateful_rmse_list=list()
lstm_stateless_rmse_list=list()

for length in range(min_length,max_length+1):

	print("*" * 33)
	print("INPUT DIMENSION:{}".format(length))
	print("*" * 33)

	data_input, expected_output = data_preprocess(length)


	print('data_input length:{}'.format(len(data_input.index)) )

	# Split training and test data: use first 80% of data points as training and remaining as test
	(x_train, y_train), (x_test, y_test) = split_data(data_input, expected_output)


	# Create the models and load trained weights
	print('Creating Fully-Connected Model and Loading Trained Weights...')
	##### LOAD MODEL WEIGHTS #####
	model_fc = create_fc_model(length)
	model_fc.load_weights(filenames[0] % length)

	print('Creating Stateful Vanilla RNN Model and Loading Trained Weights...')
	model_rnn_stateful = create_rnn_model(length, stateful=True)
	model_rnn_stateful.load_weights(filenames[1] % length)

	print('Creating stateless Vanilla RNN Model and Loading Trained Weights...')
	model_rnn_stateless = create_rnn_model(length, stateful=False)
	model_rnn_stateless.load_weights(filenames[2] % length)

	print('Creating Stateful LSTM Model and Loading Trained Weights...')
	model_lstm_stateful = create_lstm_model(length, stateful=True)
	model_lstm_stateful.load_weights(filenames[3] % length)

	print('Creating stateless LSTM Model and Loading Trained Weights...')
	model_lstm_stateless = create_lstm_model(length, stateful=False)
	model_lstm_stateless.load_weights(filenames[4] % length)

	# Predict 
	print('Predicting')
	##### PREDICT #####
	predicted_fc = model_fc.predict(x_test[:,:,0], batch_size=batch_size)
	fc_rmse = np.sqrt(((predicted_fc - y_test) ** 2).mean())
	fc_rmse_list.append(fc_rmse)

	predicted_rnn_stateful = model_rnn_stateful.predict(x_test, batch_size=batch_size)
	rnn_stateful_rmse = np.sqrt(((predicted_rnn_stateful - y_test) ** 2).mean())
	rnn_stateful_rmse_list.append(rnn_stateful_rmse)

	predicted_rnn_stateless = model_rnn_stateless.predict(x_test, batch_size=batch_size)
	rnn_stateless_rmse = np.sqrt(((predicted_rnn_stateless - y_test) ** 2).mean())
	rnn_stateless_rmse_list.append(rnn_stateless_rmse)

	predicted_lstm_stateful = model_lstm_stateful.predict(x_test, batch_size=batch_size)
	lstm_stateful_rmse = np.sqrt(((predicted_lstm_stateful - y_test) ** 2).mean())
	lstm_stateful_rmse_list.append(lstm_stateful_rmse)

	predicted_lstm_stateless = model_lstm_stateless.predict(x_test, batch_size=batch_size)
	lstm_stateless_rmse = np.sqrt(((predicted_lstm_stateless - y_test) ** 2).mean())
	lstm_stateless_rmse_list.append(lstm_stateless_rmse)

	# print('tsteps:{}'.format(tsteps))
	print('length:{}'.format(length))
	print('Fully-Connected RMSE:{}'.format( fc_rmse ))
	print('Stateful Vanilla RNN RMSE:{}'.format( rnn_stateful_rmse ))
	print('Stateless Vanilla RNN RMSE:{}'.format( rnn_stateless_rmse ))
	print('Stateful LSTM RMSE:{}'.format( lstm_stateful_rmse ))
	print('Stateless LSTM RMSE:{}'.format( lstm_stateless_rmse ))


# Save your rmse values for different length input sequence models:
# This file should have 5 rows (one row per model) and
# 10 columns (one column per input length).
# 1st row: fully-connected model
# 2nd row: vanilla rnn stateful
# 3rd row: vanilla rnn stateless
# 4th row: lstm stateful
# 5th row: lstm stateless
filename = 'all_models_rmse_values.txt'
##### PREPARE RMSE ARRAY THAT WILL BE WRITTEN INTO FILE #####
rmse_arr = np.array([fc_rmse_list, rnn_stateful_rmse_list, rnn_stateless_rmse_list, lstm_stateful_rmse_list, lstm_stateless_rmse_list])[:,:,0]
np.savetxt(filename, rmse_arr, fmt='%.6f', delimiter='\t')

print("#" * 33)
print('Plotting Results')
print("#" * 33)

plt.figure()
plt.plot(data_input[0][:100], '.')
plt.plot(expected_output[0][:100], '-')
plt.legend(['Input', 'Expected output'])
plt.title('Input - First 100 data points')

# Plot and save rmse vs Input Length
plt.figure()
plt.plot( np.arange(min_length,max_length+1), fc_rmse_list, c='black', label='FC')
plt.plot( np.arange(min_length,max_length+1), rnn_stateful_rmse_list, c='blue', label='Stateful RNN')
plt.plot( np.arange(min_length,max_length+1), rnn_stateless_rmse_list, c='cyan', label='Stateless RNN')
plt.plot( np.arange(min_length,max_length+1), lstm_stateful_rmse_list, c='red', label='Stateful LSTM')
plt.plot( np.arange(min_length,max_length+1), lstm_stateless_rmse_list, c='magenta', label='Stateless LSTM')
plt.title('RMSE vs Input Length in Test Set')
plt.xlabel('length of input sequences')
plt.ylabel('rmse')
plt.legend()
plt.grid()
plt.show()


