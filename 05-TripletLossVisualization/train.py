import os
import time
import argparse
import tensorflow as tf
import numpy as np

from src.utils import get_params
from src.load_data import train_input_fn
from src.load_data import test_input_fn
from model.model_fn import model_fn


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)

    # confirm model dir path is exists
    if not os.path.exists(args.MODEL_DIR_PATH):
        os.makedirs(args.MODEL_DIR_PATH)

    # confirm jason file is exists
    json_path = os.path.join(args.MODEL_DIR_PATH, "params.json")
    if not json_path:
        tf.logging.error("No json configuration file found at {}".format(json_path))
    
    # get training parameters
    params = get_params(json_path)

    # confirm training model saving place exists
    time_now = time.strftime("%Y-%m-%d-%H%M%S")
    chpt_dir_path = os.path.join(args.MODEL_DIR_PATH, time_now)
    if not os.path.exists(chpt_dir_path):
        os.makedirs(chpt_dir_path)

    # estmator config setting and create estimator
    session_config = tf.ConfigProto(log_device_placement=False)
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=chpt_dir_path,
                                    save_checkpoints_steps=params["save_checkpoints_steps"],
                                    keep_checkpoint_max=params["keep_checkpoint_max"],
                                    session_config=session_config)
    estimator = tf.estimator.Estimator(model_fn=model_fn, params=params, config=config)

    # training and validation
    tf.logging.info("Starting training")
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(args.DATA_DIR_PATH, params))
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: test_input_fn(args.DATA_DIR_PATH, params), throttle_secs=10)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

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