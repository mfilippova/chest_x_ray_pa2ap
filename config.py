import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output


MAPPER = {'PA': 0, 'AP': 1}


def plot_progress(train_losses, train_accs, test_losses, test_accs):
    clear_output(True)
    
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    f.set_figheight(6)
    f.set_figwidth(20)
    
    ax1.plot(train_losses, label='train loss')
    ax1.plot(test_losses, label='val loss')
    ax1.plot(np.zeros_like(train_losses), '--', label='zero')
    ax1.set_title('Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Batch number')
    ax1.legend()
    
    ax2.plot(train_accs, label='train acc')
    ax2.plot(test_accs, label='val acc')
    ax2.plot(np.ones_like(train_accs), '--', label='100% Accuracy')
    ax2.plot(np.zeros_like(train_accs), '--', label='0% Accuracy')
    ax2.set_ylim(-0.05)
    ax2.set_title('Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Batch number')
    ax2.legend()

    plt.show()
