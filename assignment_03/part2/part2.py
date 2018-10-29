from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from keras import optimizers, losses
from common import *
import plot


# List to keep track of root mean square error for different length input sequences
rnn_stateful_rmse_list=list()
rnn_stateless_rmse_list=list()

for length in range(10,10+1):

	print("*" * 33)
	print("INPUT DIMENSION:{}".format(length))
	print("*" * 33)

	data_input, expected_output = data_preprocess(length)

	print('data_input length:{}'.format(len(data_input.index)) )

	# Split training and test data: use first 80% of data points as training and remaining as test
	(x_train, y_train), (x_test, y_test) = split_data(data_input, expected_output)


	# Create the stateful model
	print('Creating Stateful Vanilla RNN Model...')
	model_rnn_stateful = create_rnn_model(length, stateful=True)
	# model_rnn_stateful.summary()
	# model_rnn_stateful.load_weights('../trained_models/rnn_stateful_model_weights_length_%d_trained.h5' % length, by_name=True)

	# Train the model
	print('Training')
	history = {'loss':[], 'val_loss':[]}
	for i in range(epochs):
		print('Epoch', i + 1, '/', epochs)
		# Note that the last state for sample i in a batch will
		# be used as initial state for sample i in the next batch.

		##### TRAIN YOUR MODEL #####
		h = model_rnn_stateful.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=2,
										 validation_data=(x_test, y_test), shuffle=False)
		history['loss'].extend(h.history['loss'])
		history['val_loss'].extend(h.history['val_loss'])
		# reset states at the end of each epoch
		model_rnn_stateful.reset_states()


	# Plot and save loss curves of training and test set vs iteration in the same graph
	##### PLOT AND SAVE LOSS CURVES #####
	plot.plot_costs(history, 'Loss of Stateful RNN Model when length=%d'%length)

	# Save your model weights with following convention:
	# For example length 1 input sequences model filename
	# rnn_stateful_model_weights_length_1.h5
	##### SAVE MODEL WEIGHTS #####
	model_rnn_stateful.save_weights('rnn_stateful_model_weights_length_%d.h5'%length)

	# Predict
	print('Predicting')
	##### PREDICT #####
	predicted_rnn_stateful = model_rnn_stateful.predict(x_test, batch_size=batch_size, verbose=0, steps=None)

	##### CALCULATE RMSE #####
	rnn_stateful_rmse = np.sqrt(((predicted_rnn_stateful - y_test) ** 2).mean())
	rnn_stateful_rmse_list.append(rnn_stateful_rmse)

	# print('tsteps:{}'.format(tsteps))
	print('length:{}'.format(length))
	print('Stateful Vanilla RNN RMSE:{}'.format( rnn_stateful_rmse ))



	# Create the stateless model
	print('Creating stateless Vanilla RNN Model...')
	model_rnn_stateless = create_rnn_model(length, stateful=False)
	# model_rnn_stateless.load_weights('../trained_models/rnn_stateless_model_weights_length_%d_trained.h5' % length, by_name=True)

	# Train the model
	print('Training')
	##### TRAIN YOUR MODEL #####
	history = model_rnn_stateless.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2,
									  validation_data=(x_test, y_test),shuffle=False)


	# Plot and save loss curves of training and test set vs iteration in the same graph
	##### PLOT AND SAVE LOSS CURVES #####
	plot.plot_costs(history.history, 'Loss of Stateless RNN Model when length=%d' % length)

	# Save your model weights with following convention:
	# For example length 1 input sequences model filename
	# rnn_stateless_model_weights_length_1.h5
	##### SAVE MODEL WEIGHTS #####
	model_rnn_stateless.save_weights('rnn_stateless_model_weights_length_%d.h5'%length)

	# Predict
	print('Predicting')
	##### PREDICT #####
	predicted_rnn_stateless = model_rnn_stateless.predict(x_test, batch_size=batch_size, verbose=0, steps=None)

	##### CALCULATE RMSE #####
	rnn_stateless_rmse = np.sqrt(((predicted_rnn_stateless - y_test) ** 2).mean())
	rnn_stateless_rmse_list.append(rnn_stateless_rmse)

	# print('tsteps:{}'.format(tsteps))
	print('length:{}'.format(length))
	print('Stateless Vanilla RNN RMSE:{}'.format( rnn_stateless_rmse ))


print("#" * 33)
print('Plotting Results')
print("#" * 33)

# Plot and save rmse vs Input Length
plt.figure()
plt.plot( np.arange(min_length,max_length+1), rnn_stateful_rmse_list, c='blue', label='Stateful RNN')
plt.plot( np.arange(min_length,max_length+1), rnn_stateless_rmse_list, c='cyan', label='Stateless RNN')
plt.title('RMSE vs Input Length in Test Set')
plt.xlabel('length of input sequences')
plt.ylabel('rmse')
plt.legend()
plt.grid()
plt.show()

# save your rmse values for different length input sequence models - stateful rnn:
filename = 'rnn_stateful_model_rmse_values.txt'
# np.savetxt(filename, np.array(rnn_stateful_rmse_list), fmt='%.6f', delimiter='\t')

# save your rmse values for different length input sequence models - stateless rnn:
filename = 'rnn_stateless_model_rmse_values.txt'
# np.savetxt(filename, np.array(rnn_stateless_rmse_list), fmt='%.6f', delimiter='\t')


