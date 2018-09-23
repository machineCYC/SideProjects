import os, sys
import argparse
import tensorflow as tf
from datetime import datetime

from src.File import prepare_file_system

from src.model import create_model_info
from src.model import maybe_download_and_extract
from src.model import create_model_graph

from src.common import create_image_lists
from src.common import add_jpeg_decoding
from src.common import cache_bottlenecks

from src.model import add_final_training_ops
from src.model import add_evaluation_step

from src.common import  get_random_cached_bottlenecks


def main(args):
    # Needed to make sure the logging output is visible.
    # See https://github.com/tensorflow/tensorflow/issues/3047
    tf.logging.set_verbosity(tf.logging.INFO)

    # Prepare necessary directories that can be used during training
    prepare_file_system(args.summaries_dir)

    # Gather information about the model architecture we'll be using.
    model_info = create_model_info(args.architecture)
    if not model_info:
        tf.logging.error("Did not recognize architecture flag")
        return -1
    
    # Set up the pre-trained graph.
    maybe_download_and_extract(model_info["data_url"], args.model_dir)
    graph, bottleneck_tensor, resized_image_tensor = (create_model_graph(model_info, args.model_dir))
    
    image_lists = create_image_lists(args.image_dir, args.testing_percentage, args.validation_percentage)

    with tf.Session(graph=graph) as sess:
        jpeg_data_placeholder, decoded_image_tensor = add_jpeg_decoding(model_info["input_width"], model_info["input_height"], model_info["input_depth"], model_info["input_mean"], model_info["input_std"])

        # We'll make sure we've calculated the "bottleneck" image summaries and
        # cached them on disk.
        # cache_bottlenecks(sess, image_lists, args.image_dir, args.bottleneck_dir, jpeg_data_placeholder, decoded_image_tensor, resized_image_tensor, bottleneck_tensor, args.architecture)

        # Add the new layer that we'll be training.
        (train_step, cross_entropy, bottleneck_input_placeholder, ground_truth_input_placeholder, final_tensor, learning_rate) = add_final_training_ops(len(image_lists.keys()), args.final_tensor_name, bottleneck_tensor, model_info["bottleneck_tensor_size"], args.learning_rate_init)

        # Create the operations we need to evaluate the accuracy of our new layer.
        evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input_placeholder)

        # Merge all the summaries and write them out to the summaries_dir
        merged = tf.summary.merge_all()
        
        train_writer = tf.summary.FileWriter(args.summaries_dir + "/train", sess.graph)
        validation_writer = tf.summary.FileWriter(args.summaries_dir + "/validation")

        # Set up all our weights to their initial default values.
        init = tf.global_variables_initializer()
        sess.run(init)

        # Run the training for as many cycles as requested on the command line.
        for i in range(args.how_many_training_iteration):
            # Get a batch of input bottleneck values from the cache stored on disk.
            (train_bottlenecks, train_ground_truth, _) = get_random_cached_bottlenecks(sess, image_lists, args.train_batch_size, "training", args.bottleneck_dir, args.image_dir, jpeg_data_placeholder, decoded_image_tensor, resized_image_tensor, bottleneck_tensor, args.architecture)

            # Feed the bottlenecks and ground truth into the graph, and run a training iteration.
            # Capture training summaries for TensorBoard with the `merged` op.
            train_summary, _ = sess.run([merged, train_step],
                                        feed_dict={bottleneck_input_placeholder: train_bottlenecks,
                                                   ground_truth_input_placeholder: train_ground_truth})
            train_writer.add_summary(train_summary, i)

            # Every so often, print out how well the graph is training.
            is_last_iteration = (i + 1 == args.how_many_training_iteration)
            if (i % args.eval_step_interval) == 0 or is_last_iteration:
                train_accuracy, cross_entropy_value = sess.run([evaluation_step, cross_entropy], 
                                                                feed_dict={bottleneck_input_placeholder: train_bottlenecks,
                                                                           ground_truth_input_placeholder: train_ground_truth})
                tf.logging.info("%s: Iteration %d: Learning rate: %.6f" % (datetime.now(), i, sess.run(learning_rate)))
                tf.logging.info("%s: Iteration %d: Train accuracy = %.1f%%" % (datetime.now(), i, train_accuracy * 100))
                tf.logging.info("%s: Iteration %d: Cross entropy = %f" % (datetime.now(), i, cross_entropy_value))
                
                (validation_bottlenecks, validation_ground_truth, _) = get_random_cached_bottlenecks(sess, image_lists, args.validation_batch_size, "validation", args.bottleneck_dir, args.image_dir, jpeg_data_placeholder, decoded_image_tensor, resized_image_tensor, bottleneck_tensor, args.architecture)
                
                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                validation_summary, validation_accuracy = sess.run([merged, evaluation_step], 
                                                                    feed_dict={bottleneck_input_placeholder: validation_bottlenecks,
                                                                               ground_truth_input_placeholder: validation_ground_truth})
                validation_writer.add_summary(validation_summary, i)

                tf.logging.info("%s: Iteration %d: Validation accuracy = %.1f%% (N=%d)" % (datetime.now(), i, validation_accuracy * 100, len(validation_bottlenecks)))

        # We've completed all our training, so run a final test evaluation on
        # some new images we haven't used before.
        test_bottlenecks, test_ground_truth, test_filenames = (get_random_cached_bottlenecks(sess, image_lists, args.test_batch_size, "testing", args.bottleneck_dir, args.image_dir, jpeg_data_placeholder, decoded_image_tensor, resized_image_tensor, bottleneck_tensor, args.architecture))
        test_accuracy, predictions = sess.run([evaluation_step, prediction], 
                                              feed_dict={bottleneck_input_placeholder: test_bottlenecks, 
                                                         ground_truth_input_placeholder: test_ground_truth})
        tf.logging.info("Final test accuracy = %.1f%% (N=%d)" % (test_accuracy * 100, len(test_bottlenecks)))

        if args.print_misclassified_test_images:
            tf.logging.info("=== MISCLASSIFIED TEST IMAGES ===")
            for i, test_filename in enumerate(test_filenames):
                if predictions[i] != test_ground_truth[i].argmax():
                    tf.logging.info("%70s  %s" % (test_filename, list(image_lists.keys())[predictions[i]]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summaries_dir",
        type=str,
        default=os.path.join(os.path.dirname(sys.argv[0]), "tmp/retrain_logs"),
        help="Where to save summary logs for TensorBoard."
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="inception_v3",
        help="""\
        Which model architecture to use. "inception_v3" is the most accurate, but
        also the slowest. For faster or smaller models, chose a MobileNet with the
        form "mobilenet_<parameter size>_<input_size>[_quantized]". For example,
        "mobilenet_1.0_224" will pick a model that is 17 MB in size and takes 224
        pixel input images, while "mobilenet_0.25_128_quantized" will choose a much
        less accurate, but smaller and faster network that's 920 KB on disk and
        takes 128x128 images. See https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
        for more information on Mobilenet.\
        """)
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join(os.path.dirname(sys.argv[0]), "tmp/imagenet"),
        help="""\
        Path to classify_image_graph_def.pb,
        imagenet_synset_to_human_label_map.txt, and
        imagenet_2012_challenge_label_map_proto.pbtxt.\
        """)
    parser.add_argument(
        "--bottleneck_dir",
        type=str,
        default=os.path.join(os.path.dirname(sys.argv[0]), "tmp/bottleneck"),
        help="Path to cache bottleneck layer values as files."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=os.path.join(os.path.dirname(sys.argv[0]), "data/train2"),
        help="Path to folders of labeled images."
    )
    parser.add_argument(
        "--how_many_training_iteration",
        type=int,
        default=2000,
        help="How many training steps to run before ending."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=150,
        help="How many images to train on at a time."
    )
    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=150,
        help="How many images to validation on at a time."
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=8000,
        help="How many images to test on at a time."
    )
    parser.add_argument(
        "--learning_rate_init",
        type=float,
        default=0.01,
        help="How large a learning rate to use when training."
    )
    parser.add_argument(
        "--eval_step_interval",
        type=int,
        default=10,
        help="How often to evaluate the training results."
    )
    parser.add_argument(
        "--testing_percentage",
        type=int,
        default=15,
        help="What percentage of images to use as a test set."
    )
    parser.add_argument(
        "--validation_percentage",
        type=int,
        default=10,
        help="What percentage of images to use as a validation set."
    )
    parser.add_argument(
        "--final_tensor_name",
        type=str,
        default="final_result",
        help="""\
        The name of the output classification layer in the retrained graph.\
        """
    )
    parser.add_argument(
        "--print_misclassified_test_images",
        default=True,
        help="""\
        Whether to print out a list of all misclassified test images.\
        """,
        action="store_true"
    )
    main(parser.parse_args())