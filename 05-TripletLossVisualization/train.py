import os
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from src.utils import Params
from model.model_fn import model_fn
# from model.input_fn import train_input_fn


def main(args):

    tf.logging.set_verbosity(tf.logging.INFO)

    json_path = os.path.join(args.model_dir, "params.json")
    if not json_path:
        tf.logging.error("No json configuration file found at {}".format(json_path))
    
    params = Params(json_path)

    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params["save_summary_steps"])
    estimator = tf.estimator.Estimator(model_fn=model_fn, params=params, config=config)

    tf.logging.info("Loading mnist datasets")
    mnist = input_data.read_data_sets(args.data_dir, one_hot=False)
    
    train_img = mnist.train.images.reshape([-1, 28, 28, 1])
    train_label = np.asarray(mnist.train.labels, dtype=np.int32)

    valid_img = mnist.validation.images.reshape([-1, 28, 28, 1])
    valid_label = np.asarray(mnist.validation.labels, dtype=np.int32)

    test_img = mnist.test.images.reshape([-1, 28, 28, 1])
    test_label = np.asarray(mnist.test.labels, dtype=np.int32)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_img},
                                                        y=train_label,
                                                        batch_size=100,
                                                        num_epochs=None,
                                                        shuffle=True)

    tf.logging.info("Starting training")
    estimator.train(input_fn=train_input_fn, steps=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/MNIST_data",
        help="Experiment directory containing params.json"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="experiments/naive_cnn",
        help="Experiment directory containing params.json"
    )
    main(parser.parse_args())