import os
import tensorflow as tf
import numpy as np
import gzip
import shutil
from six.moves import urllib


def download(data_dir_path, filename):
    """Download (and unzip) a file from the MNIST dataset if not already done."""
    filepath = os.path.join(data_dir_path, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(data_dir_path):
        tf.gfile.MakeDirs(data_dir_path)
    
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    url = "https://storage.googleapis.com/cvdf-datasets/mnist/" + filename + ".gz"
    zipped_filepath = filepath + ".gz"
    print("Downloading %s to %s" % (url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    with gzip.open(zipped_filepath, "rb") as f_in, open(filepath, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(zipped_filepath)
    return filepath


def dataset(data_dir_path, images_file, labels_file):
    images_file_path = download(data_dir_path, images_file)
    labels_file_path = download(data_dir_path, labels_file)

    def decode_image(image):
        # Normalize from [0, 255] to [0.0, 1.0]
        image = tf.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [784])
        return image / 255.0

    def decode_label(label):
        label = tf.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
        label = tf.reshape(label, [])  # label is a scalar
        return tf.to_int32(label)

    images = tf.data.FixedLengthRecordDataset(images_file_path, 28 * 28, header_bytes=16)
    images = images.map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(labels_file_path, 1, header_bytes=8)
    labels = labels.map(decode_label)
    return tf.data.Dataset.zip((images, labels))


def train(data_dir_path):
    """tf.data.Dataset object for MNIST training data."""
    return dataset(data_dir_path, "train-images-idx3-ubyte", "train-labels-idx1-ubyte")


def test(data_dir_path):
    """tf.data.Dataset object for MNIST test data."""
    return dataset(data_dir_path, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")


def train_input_fn(data_dir_path, params):
    """Train input function for the MNIST dataset.

    Args:
        data_dir_path: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    with tf.name_scope("Data_Pipeline"):
        dataset = train(data_dir_path)
        dataset = dataset.shuffle(params["train_size"] + params["valid_size"])  # whole dataset into the buffer
        dataset = dataset.repeat(params["num_epochs"])
        dataset = dataset.batch(params["batch_size"])
        dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve

    # TODO:
    # train_dataset = dataset.take(params["train_size"])
    # valid_dataset = dataset.skip(params["train_size"])

    # train_dataset = train_dataset.batch(params["batch_size"])
    # train_dataset = train_dataset.shuffle(params["train_size"])
    # train_dataset = train_dataset.prefetch(1)  # make sure you always have one batch ready to serve

    # valid_dataset = valid_dataset.batch(params["batch_size"])
    # valid_dataset = valid_dataset.shuffle(params["valid_size"])
    # valid_dataset = valid_dataset.prefetch(1)  # make sure you always have one batch ready to serve
    # return train_dataset, valid_dataset
    return dataset


def test_input_fn(data_dir_path, params):
    """Test input function for the MNIST dataset.

    Args:
        data_dir_path: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = test(data_dir_path)
    dataset = dataset.batch(params["batch_size"])
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset
