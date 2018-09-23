import os, sys
import tensorflow as tf
import tarfile
from six.moves import urllib


def create_model_info(architecture):
    """
    Given the name of a model architecture, returns information about it.
    There are different base image recognition pretrained models that can be
    retrained using transfer learning, and this function translates from the name
    of a model to the attributes that are needed to download and train with it.
    
    Args:
        architecture: Name of a model architecture.
    
    Returns:
        Dictionary of information about the model, or None if the name isn"t
        recognized
    Raises:
        ValueError: If architecture name is unknown.
    """
    architecture = architecture.lower()
    if architecture == "inception_v3":
        data_url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"
        bottleneck_tensor_name = "pool_3/_reshape:0"
        bottleneck_tensor_size = 2048
        input_width = 299
        input_height = 299
        input_depth = 3
        resized_input_tensor_name = "Mul:0"
        model_file_name = "classify_image_graph_def.pb"
        input_mean = 128
        input_std = 128
    else:
        tf.logging.error("Couldn't understand architecture name {}".format(architecture))
        raise ValueError("Unknown architecture", architecture)

    return {
        "data_url": data_url,
        "bottleneck_tensor_name": bottleneck_tensor_name,
        "bottleneck_tensor_size": bottleneck_tensor_size,
        "input_width": input_width,
        "input_height": input_height,
        "input_depth": input_depth,
        "resized_input_tensor_name": resized_input_tensor_name,
        "model_file_name": model_file_name,
        "input_mean": input_mean,
        "input_std": input_std,
    }


def maybe_download_and_extract(data_url, model_dir):
    """
    Download and extract model tar file.
    If the pretrained model we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a directory.
    
    Args:
        data_url: Web location of the tar file containing the pretrained model.
    """
    dest_directory = model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write("\r>> Downloading %s %.1f%%" % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        tf.logging.info("Successfully downloaded", filename, statinfo.st_size, "bytes.")
    tarfile.open(filepath, "r:gz").extractall(dest_directory)


def create_model_graph(model_info, model_dir):
    """"
    Creates a graph from saved GraphDef file and returns a Graph object.
    
    Args:
        model_info: Dictionary containing information about the model architecture.
    
    Returns:
        Graph holding the trained Inception network, and various tensors we'll be
        manipulating.
    """
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(model_dir, model_info["model_file_name"])
        with tf.gfile.FastGFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
            
            bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(graph_def, name="", return_elements=[model_info["bottleneck_tensor_name"], model_info["resized_input_tensor_name"],]))

    return graph, bottleneck_tensor, resized_input_tensor


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor, bottleneck_tensor_size, learning_rate_init):
    """
    Adds a new softmax and fully-connected layer for training.
    We need to retrain the top layer to identify our new classes, so this function
    adds the right operations to the graph, along with some variables to hold the
    weights, and then sets up all the gradients for the backward pass.
    The set up for the softmax and fully-connected layers is based on:
    https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
    
    Args:
        class_count: Integer of how many categories of things we're trying to
        recognize.
        final_tensor_name: Name string for the new final node that produces results.
        bottleneck_tensor: The output of the main CNN graph.
        bottleneck_tensor_size: How many entries in the bottleneck vector.
    
    Returns:
        The tensors for the training and cross entropy results, and tensors for the
        bottleneck input and ground truth input.
    """
    with tf.name_scope("input"):
        bottleneck_input_placeholder = tf.placeholder_with_default(bottleneck_tensor,
            shape=[None, bottleneck_tensor_size],
            name="BottleneckInputPlaceholder")

        ground_truth_input_placeholder = tf.placeholder(tf.float32, [None, class_count], name="GroundTruthInput")

    # Organizing the following ops as `final_training_ops` so they're easier
    # to see in TensorBoard
    layer_name = "final_training_ops"
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            initial_value = tf.truncated_normal([bottleneck_tensor_size, class_count], stddev=0.001)

            layer_weights = tf.Variable(initial_value, name="final_weights")
            _variable_summaries(layer_weights)
        
        with tf.name_scope("biases"):
            layer_biases = tf.Variable(tf.zeros([class_count]), name="final_biases")
            _variable_summaries(layer_biases)
        
        with tf.name_scope("Wx_plus_b"):
            logits = tf.matmul(bottleneck_input_placeholder, layer_weights) + layer_biases
            tf.summary.histogram("pre_activations", logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    tf.summary.histogram("activations", final_tensor)

    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_input_placeholder, logits=logits)
        with tf.name_scope("total"):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar("cross_entropy", cross_entropy_mean)

    with tf.name_scope("train"):
        lr = learning_rate_init
        step_rate = 80
        decay = 0.95

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(lr, global_step, step_rate, decay, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean, global_step=global_step)

    return (train_step, cross_entropy_mean, bottleneck_input_placeholder, ground_truth_input_placeholder, final_tensor, learning_rate)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """
    Inserts the operations we need to evaluate the accuracy of our results.
    
    Args:
        result_tensor: The new final node that produces results.
        ground_truth_tensor: The node we feed ground truth data
        into.
    
    Returns:
        Tuple of (evaluation step, prediction).
    """
    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_prediction"):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(prediction, tf.argmax(ground_truth_tensor, 1))
        with tf.name_scope("accuracy"):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", evaluation_step)
    return evaluation_step, prediction


def _variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)