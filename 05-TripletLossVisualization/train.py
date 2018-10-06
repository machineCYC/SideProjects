import os
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from src.utils import get_params
from model.model_fn import model_fn


def main(args):

    tf.logging.set_verbosity(tf.logging.INFO)

    json_path = os.path.join(args.MODEL_DIR_PATH, "params.json")
    if not json_path:
        tf.logging.error("No json configuration file found at {}".format(json_path))
    
    params = get_params(json_path)

    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.MODEL_DIR_PATH,
                                    save_summary_steps=params["save_summary_steps"])
    estimator = tf.estimator.Estimator(model_fn=model_fn, params=params, config=config)

    tf.logging.info("Loading mnist datasets")
    mnist = input_data.read_data_sets(args.DATA_DIR_PATH, one_hot=False)
    
    train_img = mnist.train.images.reshape([-1, 28, 28, 1]) # (55000, 28, 28, 1)
    train_label = np.asarray(mnist.train.labels, dtype=np.int32)

    valid_img = mnist.validation.images.reshape([-1, 28, 28, 1]) # (5000, 28, 28, 1)
    valid_label = np.asarray(mnist.validation.labels, dtype=np.int32)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_img},
                                                        y=train_label,
                                                        batch_size=64,
                                                        num_epochs=1,
                                                        shuffle=True)
    
    valid_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": valid_img},
                                                        y=valid_label,
                                                        batch_size=64,
                                                        num_epochs=1,
                                                        shuffle=False)

    tf.logging.info("Starting training")

    for e in range(params["num_epochs"]):
        tf.logging.info("Training on epochs {}.".format(e))
        estimator.train(input_fn=train_input_fn)

        tf.logging.info("Validation on epochs {}.".format(e))
        estimator.evaluate(input_fn=valid_input_fn)

if __name__ == "__main__":
    PROJECT_DIR_PATH = os.path.dirname(__file__)
    DATA_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "data/MNIST_data")
    MODEL_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "experiments/naive_cnn")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--DATA_DIR_PATH",
        type=str,
        default=DATA_DIR_PATH,
        help="Experiment directory containing params.json"
    )
    parser.add_argument(
        "--MODEL_DIR_PATH",
        type=str,
        default=MODEL_DIR_PATH,
        help="Experiment directory containing params.json"
    )
    main(parser.parse_args())