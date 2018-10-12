import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from src.utils import get_params
from src.create_metadata_tsv import write_matadata_tsv
from src.create_sprite_image import save_sprite_image
from model.model_fn import model_fn
from src.load_data import test_input_fn


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    
    embedding_dir_path = os.path.join(args.MODEL_DIR_PATH, "embedding")
    if not os.path.exists(embedding_dir_path):
        os.makedirs(embedding_dir_path)

    json_path = os.path.join(args.MODEL_DIR_PATH, "params.json")
    if not json_path:
        tf.logging.error("No json configuration file found at {}".format(json_path))
    
    params = get_params(json_path)
    
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.MODEL_CHPT_DIR_PATH,
                                    save_summary_steps=params["save_summary_steps"])

    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    #### prepare sprite.png and metadata.tsv
    sprite_image_path = os.path.join(embedding_dir_path, "sprite.png")
    if not os.path.exists(sprite_image_path):
        save_sprite_image(embedding_dir_path, args.DATA_DIR_PATH)
    metadata_path = os.path.join(embedding_dir_path, "metadata.tsv")
    if not os.path.exists(metadata_path):
        write_matadata_tsv(embedding_dir_path, args.DATA_DIR_PATH)

    # Compute embeddings on the test set
    tf.logging.info("Predicting")
    predictions = estimator.predict(lambda: test_input_fn(args.DATA_DIR_PATH, params))

    embeddings = np.zeros((params["test_size"], params["embedding_size"]))
    for i, p in enumerate(predictions):
        embeddings[i] = p["embeddings"]

    # Visualize test embeddings
    embedding_var = tf.Variable(embeddings, name="mnist_embedding")
    summary_writer = tf.summary.FileWriter(embedding_dir_path)

    config1 = projector.ProjectorConfig()
    embedding = config1.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # Specify where you find the metadata
    embedding.metadata_path = "metadata.tsv" #'metadata.tsv'
    # Specify where you find the sprite (we will create this later)
    embedding.sprite.image_path = "sprite.png"
    embedding.sprite.single_image_dim.extend([28, 28])
    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(summary_writer, config1)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(embedding_var.initializer)
        saver.save(sess, os.path.join(embedding_dir_path, "embeddings.ckpt"))

if __name__ == "__main__":
    PROJECT_DIR_PATH = os.path.dirname(__file__)
    DATA_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "data/MNIST_data")
    MODEL_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "experiments/naive_cnn")
    MODEL_CHPT_DIR_PATH = os.path.join(MODEL_DIR_PATH, "2018-10-12-002416")

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
    parser.add_argument(
        "--MODEL_CHPT_DIR_PATH",
        type=str,
        default=MODEL_CHPT_DIR_PATH,
        help="Experiment directory containing params.json"
    )
    main(parser.parse_args())