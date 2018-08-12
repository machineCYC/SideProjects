import matplotlib.pyplot as plt
import numpy as np
import os


strProjectFolderPath = os.path.dirname(os.path.dirname(__file__))
strOutputFolderPath = os.path.join(strProjectFolderPath, "02-Output")


def visualize_reconstruct_images(autoencoder, save_image_name, images, vis_col=10):
    fig, axis = plt.subplots(2, vis_col, figsize=(vis_col, 2))
    for i in range(vis_col):
        img_org = images[i].reshape(28, 28)
        img = np.reshape(autoencoder.predict(images[i].reshape(-1, 784)), (28, 28))

        axis[0][i].imshow(img_org, cmap="gray")
        axis[1][i].imshow(img, cmap="gray")
    plt.savefig(os.path.join(strOutputFolderPath, save_image_name))

def visualize_reconstruct_noiseimages(autoencoder, save_image_name, images, noise_images, vis_col=10):
    fig, axis = plt.subplots(3, vis_col, figsize=(vis_col, 3))
    for i in range(vis_col):
        img_org = images[i].reshape(28, 28)
        img_noise_org = noise_images[i].reshape(28, 28)
        img_recon = np.reshape(autoencoder.predict(noise_images[i].reshape(-1, 784)), (28, 28))

        axis[0][i].imshow(img_org, cmap="gray")
        axis[1][i].imshow(img_noise_org, cmap="gray")
        axis[2][i].imshow(img_recon, cmap="gray")
    plt.savefig(os.path.join(strOutputFolderPath, save_image_name))

def visualize_2dim_code(autoencoder, save_image_name, images, labels):

    code = autoencoder.code(images)

    cm = plt.cm.get_cmap("RdYlBu")

    plt.figure(figsize=(10, 8))
    plt.scatter(code[:, 0], code[:, 1], c=labels, cmap=cm, s=35, edgecolors="k")
    plt.colorbar()
    plt.savefig(os.path.join(strOutputFolderPath, save_image_name))

def add_noise(images, noise_factor):
    np.random.seed(1)
    return images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
