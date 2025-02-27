import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

index = np.where(y_train == 0)[0][0]
plt.imshow(x_train[index])
plt.show()