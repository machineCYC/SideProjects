import tensorflow as tf
import time
import numpy as np


class AutoEncoder(object):
    """
    A L-layer AutoEncoder network with the following architecture:
    
    input - (hidden-relu) x L - output
    
    The network operates on minibatches of data that have shape (N, D)
    consisting of N images, each with D dim.
    """
    
    def __init__(self, n_input=28*28, n_hidden=[256, 64, 2],  float_Learning_rate=0.01):
        """
        Initialize a new network.

        Inputs:
        - n_input: An lintegers giving the size of input layer.
        - n_hidden: A list of integers giving the size of each hidden layer.
        - float_Learning_rate: Scalar learning rate
        """
        self.n_input = n_input
        self.build(n_input, n_hidden, float_Learning_rate) 
        self.sess = tf.Session()


    def build(self, n_input, n_hidden, float_Learning_rate):
        """
        This funciton build the graph

        Inputs:
        - n_hidden: The same as above.
        - float_Learning_rate: The same as above.
        """

        # define Input
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, n_input])

        # build autoencoder
        self.Loss, self.Decoder, self.Encoder = self.structure(X=self.X, Y=self.Y, n_hidden=n_hidden)

        # optimization
        self.Optimizer = tf.train.AdamOptimizer(float_Learning_rate).minimize(self.Loss)

        # initialization
        self.init = tf.global_variables_initializer()
        
        self.saver = tf.train.Saver()


    def structure(self, X, Y, n_hidden):
        """
        This funciton define the model graph

        Inputs:
        - X: Must be a placeholder.
        - n_hidden: The same as above.
        """

        # Define variable
        self.weights = {}
        self.biases = {}

        # Encoder variable
        n_Encoder = [self.n_input] + n_hidden
        for layer, layer_size in enumerate(n_Encoder[:-1]):
            self.weights["encoder_h{}".format(layer + 1)] = tf.Variable(tf.truncated_normal(shape=[layer_size, n_Encoder[layer + 1]], stddev=0.1))
            self.biases["encoder_b{}".format(layer + 1)] = tf.Variable(tf.constant(0.1, shape=[n_Encoder[layer + 1]]))
        
        # Decoder variable
        n_Decoder = list(reversed(n_hidden)) + [self.n_input]
        for layer, layer_size in enumerate(n_Decoder[:-1]):
            self.weights["decoder_h{}".format(layer + 1)] = tf.Variable(tf.truncated_normal(shape=[layer_size, n_Decoder[layer + 1]], stddev=0.1))
            self.biases["decoder_b{}".format(layer + 1)] = tf.Variable(tf.constant(0.1, shape=[n_Decoder[layer + 1]]))


        # Encoder
        Encoder = tf.nn.relu(tf.add(tf.matmul(X, self.weights["encoder_h1"]), self.biases["encoder_b1"]))
        for layer in range(1, len(n_hidden)):
            Encoder = tf.nn.relu(tf.add(tf.matmul(Encoder, self.weights["encoder_h{}".format(layer + 1)]), self.biases["encoder_b{}".format(layer + 1)]))

        # Decoder
        Decoder = tf.nn.relu(tf.add(tf.matmul(Encoder, self.weights["decoder_h1"]), self.biases["decoder_b1"]))
        for layer in range(1, len(n_hidden)-1):
            Decoder = tf.nn.relu(tf.add(tf.matmul(Decoder, self.weights["decoder_h{}".format(layer + 1)]), self.biases["decoder_b{}".format(layer + 1)]))

        Decoder = tf.nn.relu(tf.add(tf.matmul(Decoder, self.weights["decoder_h{}".format(len(n_hidden))]), self.biases["decoder_b{}".format(len(n_hidden))]))

        # Loss
        Loss = tf.reduce_mean(tf.pow(Decoder - Y, 2))
        
        return Loss, Decoder, Encoder 


    def fit(self, X, Y, int_Epochs, int_Batch_size):
        """
        Run optimization to train the model.

        Inputs:
        - X: Array of data, of shape (N, D)
        - int_Epochs: The number of epochs to run for during training.
        - int_Batch_size: Size of minibatches used to compute loss and gradient during training.
        """

        self.sess.run(self.init)

        N = X.shape[0]
        list_Loss = []
        for epoch in range(int_Epochs):
            start_time = time.time()

            index = [i for i in range(N)]
            np.random.shuffle(index)
            while len(index) > int_Batch_size:
                batch_index = [index.pop() for _ in range(int_Batch_size)]

                _, L = self.sess.run([self.Optimizer, self.Loss], feed_dict = {self.X:X[batch_index],
                                                                               self.Y:Y[batch_index]})

            list_Loss.append(np.round(L, 5))
            print("Epoch:{}".format(epoch + 1), "{:.2f}s".format(time.time()-start_time), "Loss:{:.5f}".format(L))

        return list_Loss

    def predict(self, X):
        return self.sess.run(self.Decoder, feed_dict={self.X: X})

    def Code(self, X):
        return self.sess.run(self.Encoder, feed_dict={self.X: X})

    def save(self, path):
        save_path = self.saver.save(self.sess, path)
        print("Save to path: ", save_path)
    
    def reload(self, path):
        self.saver.restore(self.sess, path)
        print("reload AutoEncoder: ", path)


