from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from common import *
import plot

# List to keep track of root mean square error for different length input sequences
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

	
	# Create the stateful model
	print('Creating Stateful LSTM Model...')
	model_lstm_stateful = create_lstm_model(length, stateful=True)

	# Train the model
	print('Training')
	history = {'loss':[], 'val_loss':[]}
	for i in range(epochs):
		print('Epoch', i + 1, '/', epochs)
		# Note that the last state for sample i in a batch will
		# be used as initial state for sample i in the next batch.
		
		##### TRAIN YOUR MODEL #####
		h = model_lstm_stateful.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=2,
										 validation_data=(x_test, y_test), shuffle=False)
		history['loss'].extend(h.history['loss'])
		history['val_loss'].extend(h.history['val_loss'])
		# reset states at the end of each epoch
		model_lstm_stateful.reset_states()

	plot.plot_costs(history, 'Loss of Stateful LSTM Model when length=%d' % length)


	# Plot and save loss curves of training and test set vs iteration in the same graph
	##### PLOT AND SAVE LOSS CURVES #####

	# Save your model weights with following convention:
	# For example length 1 input sequences model filename
	# lstm_stateful_model_weights_length_1.h5
	##### SAVE MODEL WEIGHTS #####
	model_lstm_stateful.save_weights('lstm_stateful_model_weights_length_%d.h5'%length)

	# Predict 
	print('Predicting')
	##### PREDICT #####
	predicted_lstm_stateful = model_lstm_stateful.predict(x_test, batch_size=batch_size)

	##### CALCULATE RMSE #####
	lstm_stateful_rmse = np.sqrt(((predicted_lstm_stateful - y_test) ** 2).mean())
	lstm_stateful_rmse_list.append(lstm_stateful_rmse)

	# print('tsteps:{}'.format(tsteps))
	print('length:{}'.format(length))
	print('Stateful LSTM RMSE:{}'.format( lstm_stateful_rmse ))



	# Create the stateless model
	print('Creating stateless LSTM Model...')
	model_lstm_stateless = create_lstm_model(length,stateful=False)

	# Train the model
	print('Training')
	##### TRAIN YOUR MODEL #####
	history = model_lstm_stateless.fit(x_train, y_train, epochs=10, batch_size=batch_size, verbose=2,
									  validation_data=(x_test, y_test), shuffle=False)

	# Plot and save loss curves of training and test set vs iteration in the same graph
	##### PLOT AND SAVE LOSS CURVES #####
	plot.plot_costs(history.history, 'Loss of Stateless LSTM Model when length=%d' % length)

	# Save your model weights with following convention:
	# For example length 1 input sequences model filename
	# lstm_stateless_model_weights_length_1.h5
	##### SAVE MODEL WEIGHTS #####
	model_lstm_stateful.save_weights('lstm_stateless_model_weights_length_%d.h5'%length)

	# Predict 
	print('Predicting')
	predicted_lstm_stateless = model_lstm_stateless.predict(x_test, batch_size=batch_size)
	lstm_stateless_rmse = np.sqrt(((predicted_lstm_stateless - y_test) ** 2).mean())
	lstm_stateless_rmse_list.append(lstm_stateless_rmse)

	# print('tsteps:{}'.format(tsteps))
	print('length:{}'.format(length))
	print('Stateless LSTM RMSE:{}'.format( lstm_stateless_rmse ))



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
plt.plot( np.arange(min_length,max_length+1), lstm_stateful_rmse_list, c='red', label='Stateful LSTM')
plt.plot( np.arange(min_length,max_length+1), lstm_stateless_rmse_list, c='magenta', label='Stateless LSTM')
plt.title('RMSE vs Input Length in Test Set')
plt.xlabel('length of input sequences')
plt.ylabel('rmse')
plt.legend()
plt.grid()
plt.show()

# save your rmse values for different length input sequence models - stateful rnn:
filename = 'lstm_stateful_model_rmse_values.txt'
np.savetxt(filename, np.array(lstm_stateful_rmse_list), fmt='%.6f', delimiter='\t')

# save your rmse values for different length input sequence models - stateless rnn:
filename = 'lstm_stateless_model_rmse_values.txt'
np.savetxt(filename, np.array(lstm_stateless_rmse_list), fmt='%.6f', delimiter='\t')







