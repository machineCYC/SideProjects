import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.load_data import test


def save_sprite_image(embedding_dir_path, data_dir_path):
    with tf.Session() as sess:
        test_data = test(data_dir_path)
        test_img = test_data.map(lambda img, lab: img)
        test_img = test_img.batch(10000)
        test_img = test_img.make_one_shot_iterator().get_next()
        test_img = sess.run(test_img)

    to_visualise = test_img
    to_visualise = vector_to_matrix_mnist(to_visualise)
    to_visualise = invert_grayscale(to_visualise)

    sprite_image = create_sprite_image(to_visualise)

    plt.imsave(os.path.join(embedding_dir_path, "sprite.png"), sprite_image, cmap="gray")
    plt.imshow(sprite_image, cmap="gray")


def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    
    
    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))
    
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w] = this_img
    
    return spriteimage

def vector_to_matrix_mnist(mnist_digits):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits,(-1,28,28))

def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1-mnist_digits