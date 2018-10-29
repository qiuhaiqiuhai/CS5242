import matplotlib.pyplot as plt
import numpy as np

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

def plot_data(data):

    plt.plot(data[0], 'C%d'%0, linewidth=1.5, label='Expected output')
    plt.plot(data[1], 'C%d' % 1, linewidth=1.5, label='Prediction')

    plt.xlabel('timestamp')
    plt.ylabel('value')
    plt.title('Output from most recent 5 timestamp\n(least recent 5 timestamp set to 0), Length = 10')
    plt.legend()

    # plt.axis((0, nb_of_epochs, 0, 0.001))
    plt.grid()
    plt.show()