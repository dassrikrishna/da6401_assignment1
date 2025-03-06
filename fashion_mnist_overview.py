# Question 1:
import wandb
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist

# wandb login
wandb.login()
wandb.init(project = "MA24M025_DA6401-Assignment-1", mode = "online")

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# wandb
wandb_images = []

fig, axes = plt.subplots(2, 5, figsize = (10, 5))
fig.suptitle("Fashion-MNIST Images")
axes = axes.ravel()

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

for i in range(10):
    index = np.where(y_train == i)[0][0]
    axes[i].set_title(class_names[i])
    axes[i].imshow(x_train[index])
    wandb_image = wandb.Image(x_train[index], caption = class_names[i]) # wandb image
    wandb_images.append(wandb_image)

plt.show()
# log image 
wandb.log({"Fashion-MNIST Images": wandb_images})
# finish
wandb.finish()
