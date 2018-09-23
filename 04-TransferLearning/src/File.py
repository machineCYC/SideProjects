import tensorflow as tf


def prepare_file_system(summaries_dir):
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(summaries_dir):
        tf.gfile.DeleteRecursively(summaries_dir)
    tf.gfile.MakeDirs(summaries_dir)
    return
