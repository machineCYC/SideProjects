import os
import tensorflow as tf

from src.load_data import test


def write_matadata_tsv(embedding_dir_path, data_dir_path):
    with tf.Session() as sess:
        test_data = test(data_dir_path)
        test_label = test_data.map(lambda img, lab: lab)
        test_label = test_label.batch(10000)
        test_label = test_label.make_one_shot_iterator().get_next()
        test_label = sess.run(test_label)


    with open(os.path.join(embedding_dir_path, "metadata.tsv"), "w") as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(test_label):
            f.write("%d\t%d\n" % (index, label))