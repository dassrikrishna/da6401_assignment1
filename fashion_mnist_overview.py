import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

for i in range(10):
    index = np.where(y_train == i)[0][0]
    plt.imshow(x_train[index])
    plt.show()