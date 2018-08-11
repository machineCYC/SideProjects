import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
import os


strProjectFolderPath = os.path.dirname(os.path.dirname(__file__))
strOutputFolderPath = os.path.join(strProjectFolderPath, "02-Output")

def visualize_reconstrict_images(autoencoder, save_image_name, images, vis_col=10):
    fig, axis = plt.subplots(2, vis_col, figsize=(vis_col, 2))
    for i in range(vis_col):
        img_train_org = images[i].reshape(28, 28)
        img_train = np.reshape(autoencoder.predict(images[i].reshape(-1, 784)), (28, 28))

        axis[0][i].imshow(img_train_org, cmap="gray")
        axis[1][i].imshow(img_train, cmap="gray")
    plt.savefig(os.path.join(strOutputFolderPath, save_image_name))


def visualize_2dim_code(autoencoder, save_image_name, images, labels):

    code = autoencoder.code(images)

    cm = plt.cm.get_cmap("RdYlBu")

    plt.figure(figsize=(10, 8))
    plt.scatter(code[:, 0], code[:, 1], c=labels, cmap=cm, alpha=0.5, s=35)
    plt.colorbar()
    plt.savefig(os.path.join(strOutputFolderPath, save_image_name))
