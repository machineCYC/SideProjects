import tensorflow as tf

from model.triplet_loss import batch_hard_triplet_loss
from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_semi_hard_triplet_loss


def modify_structure(images, params):
    """
    Define NN structure, map the image to Euclidean space with embedding_size dim.

     Args:
        images: tensor, shape (batch_size, img_size, img_size, img_channel)

    Returns:
        embeddings: tensor, shape (batch_size, embedding_size)
    """
    out = images
    with tf.name_scope("conv_block_1"):
        # Convolutional block
        out = _conv_block(input_layer=out, filters=32, kernel_size=3)
    with tf.name_scope("conv_block_2"):
        # Convolutional block
        out = _conv_block(input_layer=out, filters=64, kernel_size=3)
    with tf.name_scope("conv_block_3"):
        # Convolutional block
        out = _conv_block(input_layer=out, filters=128, kernel_size=3)

    with tf.name_scope("flatten"):
        out = tf.layers.flatten(out)
    with tf.name_scope("dense_1"):
        out = tf.layers.dense(out, params["embedding_size"])
    with tf.name_scope("L2_normalize_embeddings"):
        embeddings = tf.nn.l2_normalize(out, axis=1, epsilon=1e-10, name="embeddings")
    return embeddings


def model_fn(features, labels, params, mode):
    """
    Args:
        features:dict

    Returns:
        model_spec: tf.estimator.EstimatorSpec object

    # 1. Configure the model via TensorFlow operations
    # 2. Define the loss function for training/evaluation
    # 3. Define the training operation/optimizer
    # 4. Generate predictions
    # 5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object
   """

    images = features

    images = tf.reshape(images, [-1, params["image_size"], params["image_size"], params["image_channel"]])

    if not images.shape[1:] == [params["image_size"], params["image_size"], params["image_channel"]]:
        tf.logging.error("Image shape do not equal to the config setting")
    
    with tf.name_scope("Embedding_Model"):
        embeddings = modify_structure(images, params)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"embeddings": embeddings}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    with tf.name_scope("Loss"):
        if params["triplet_strategy"] == "batch_all":
            triplet_loss, fraction_positive_triplets = batch_all_triplet_loss(labels, embeddings, params["margin"])
        elif params["triplet_strategy"] == "batch_hard":
            triplet_loss = batch_hard_triplet_loss(labels, embeddings, params["margin"])
        elif params["triplet_strategy"] == "batch_semi_hard":
            triplet_loss = batch_semi_hard_triplet_loss(labels, embeddings, params["margin"])
        else:
            raise ValueError("Triplet strategy not recognized: {}".format(params["triplet_strategy"]))
    
    # Summaries for training
    tf.summary.scalar("Triplet_Loss", triplet_loss)
    if params["triplet_strategy"] == "batch_all":
        tf.summary.scalar("fraction_positive_triplets", fraction_positive_triplets)

    tf.summary.image("train_image", images, max_outputs=10)

    # Define training step that minimizes the loss with the Adam optimizer
    optimizer = tf.train.AdamOptimizer(params["learning_rate"])
    global_step = tf.train.get_global_step()
    train_op = optimizer.minimize(loss=triplet_loss, global_step=global_step)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode, loss=triplet_loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=triplet_loss)


def _conv_block(input_layer, filters, kernel_size):
        # Convolutional Layer
        conv = tf.layers.conv2d(inputs=input_layer,
                                filters=filters,
                                kernel_size=kernel_size,
                                padding="same",
                                activation=tf.nn.relu)
        # Pooling Layer
        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=2, strides=2)
        return pool




