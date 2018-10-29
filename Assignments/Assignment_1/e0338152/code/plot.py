import matplotlib.pyplot as plt
import numpy as np

def plot_costs(module_name, nb_of_epochs, train_costs, val_costs, test_costs):

    epochs_x = np.linspace(1, nb_of_epochs, num=nb_of_epochs)
    plt.plot(epochs_x, train_costs, 'r-', linewidth=1.5, label='Train')
    plt.plot(epochs_x, val_costs, 'b-', linewidth=1.5, label='Validation')
    plt.plot(epochs_x, test_costs, 'g-', linewidth=1.5, label='Test')


    best_val = np.min(val_costs);
    best_epoch = np.argmin(val_costs) + 1;
    plt.xlabel('{} epochs'.format(nb_of_epochs))
    plt.ylabel('Cross Entropy Cost')
    plt.title(module_name + ' Best Validation Performance is {:.4f} at epoch {}'.format(best_val, best_epoch))
    plt.legend()

    plt.axhline(y=best_val, color="blue", linestyle="--", linewidth=0.5)
    plt.axvline(x=best_epoch, color="blue", linestyle="--", linewidth=0.5)

    plt.axis((0, nb_of_epochs, 0, 2.5))
    plt.grid()
    plt.show()


def plot_accuracys(module_name, nb_of_epochs, train_acc, val_acc, test_acc):
    epochs_x = np.linspace(1, nb_of_epochs, num=nb_of_epochs)
    plt.plot(epochs_x, train_acc, 'r-', linewidth=1.5, label='Train')
    plt.plot(epochs_x, val_acc, 'b-', linewidth=1.5, label='Validation')
    plt.plot(epochs_x, test_acc, 'g-', linewidth=1.5, label='Test')

    best_val = np.max(val_acc);
    best_epoch = np.argmax(val_acc) + 1;
    plt.xlabel('{} epochs'.format(nb_of_epochs))
    plt.ylabel('Accuracy')
    plt.title(module_name + ' Best Validation Accuracy is {:.4f} at epoch {}'.format(best_val, best_epoch))
    plt.legend()

    plt.axhline(y=best_val, color="blue", linestyle="--", linewidth=0.5)
    plt.axvline(x=best_epoch, color="blue", linestyle="--", linewidth=0.5)

    plt.axis((0, nb_of_epochs, 0, 1.0))
    plt.grid()
    plt.show()


def plot_lrate(module_name, nb_of_epochs, lrate):
    epochs_x = np.linspace(1, nb_of_epochs, num=nb_of_epochs)
    plt.plot(epochs_x, lrate, 'g-', linewidth=1.5, label='Learning Rate')

    plt.xlabel('{} epochs'.format(nb_of_epochs))
    plt.ylabel('Learning Rate')
    plt.title(module_name + ' Step decay scheduled learning rate')
    plt.legend()

    plt.axis((0, nb_of_epochs, 0, lrate[0]))
    plt.grid()
    plt.show()