import tensorflow as tf
import time, os
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
        self.saver = tf.train.Saver()
        
        self.merge = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(os.path.join(os.path.dirname(os.path.dirname(__file__)), "02-Output/"), self.sess.graph)


    def build(self, n_input, n_hidden, float_Learning_rate):
        """
        This funciton build the graph

        Inputs:
        - n_hidden: The same as above.
        - float_Learning_rate: The same as above.
        """
        # define Input
        with tf.name_scope("Inputs"):
            self.X = tf.placeholder(dtype=tf.float32, shape=[None, n_input], name="X_input")
            self.Y = tf.placeholder(dtype=tf.float32, shape=[None, n_input], name="Y_input")

        with tf.name_scope("input_reshape"):  
            input_image = tf.reshape(self.X, [-1, 28, 28, 1])  
            tf.summary.image("input", input_image, 10) 

        # build autoencoder
        self.Loss, self.Decoder, self.Encoder = self.structure(X=self.X, Y=self.Y, n_hidden=n_hidden)

        # optimization
        with tf.name_scope("Train"):
            self.Optimizer = tf.train.AdamOptimizer(float_Learning_rate).minimize(self.Loss)

        # initialization
        self.init = tf.global_variables_initializer()


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
            
            with tf.name_scope("En_Weights{}".format(layer + 1)):
                self.weights["encoder_w{}".format(layer + 1)] = tf.Variable(tf.truncated_normal(shape=[layer_size, n_Encoder[layer + 1]], stddev=0.1), name="en_w{}".format(layer + 1))
                tf.summary.histogram("layer{}/weights".format(layer + 1), self.weights["encoder_w{}".format(layer + 1)])
            
            with tf.name_scope("En_Biase{}".format(layer + 1)):
                self.biases["encoder_b{}".format(layer + 1)] = tf.Variable(tf.constant(0.1, shape=[n_Encoder[layer + 1]]), name="en_b{}".format(layer + 1))
                tf.summary.histogram("layer{}/biases".format(layer + 1), self.biases["encoder_b{}".format(layer + 1)])

        # Decoder variable
        n_Decoder = list(reversed(n_hidden)) + [self.n_input]
        for layer, layer_size in enumerate(n_Decoder[:-1]):

            with tf.name_scope("De_Weights{}".format(layer + 1)):
                self.weights["decoder_w{}".format(layer + 1)] = tf.Variable(tf.truncated_normal(shape=[layer_size, n_Decoder[layer + 1]], stddev=0.1), name="de_w{}".format(layer + 1))
                tf.summary.histogram("layer{}/weights".format(layer + 1), self.weights["decoder_w{}".format(layer + 1)])

            with tf.name_scope("De_Biases{}".format(layer + 1)):
                self.biases["decoder_b{}".format(layer + 1)] = tf.Variable(tf.constant(0.1, shape=[n_Decoder[layer + 1]]), name="de_b{}".format(layer + 1))
                tf.summary.histogram("layer{}/biases".format(layer + 1), self.biases["decoder_b{}".format(layer + 1)])


        # Encoder
        Encoder = self.fc_layer(hidden_layer=X, weights=self.weights["encoder_w1"], biases=self.biases["encoder_b1"], name="Encoder_Layer{}".format(1))
        tf.summary.histogram("Encoder_layer{}".format(1), Encoder)

        for layer in range(1, len(n_hidden)):
            Encoder = self.fc_layer(hidden_layer=Encoder, weights=self.weights["encoder_w{}".format(layer + 1)], biases=self.biases["encoder_b{}".format(layer + 1)], name="Encoder_Layer{}".format(layer + 1))
            tf.summary.histogram("Encoder_layer{}".format(layer + 1), Encoder)

        # Decoder
        Decoder = self.fc_layer(hidden_layer=Encoder, weights=self.weights["decoder_w1"], biases=self.biases["decoder_b1"], name="Decoder_Layer{}".format(1))
        tf.summary.histogram("Decoder_layer{}".format(1), Decoder)

        for layer in range(1, len(n_hidden)):
            Decoder = self.fc_layer(hidden_layer=Decoder, weights=self.weights["decoder_w{}".format(layer + 1)], biases=self.biases["decoder_b{}".format(layer + 1)], name="Decoder_Layer{}".format(layer + 1))
            tf.summary.histogram("Decoder_layer{}".format(layer + 1), Decoder)

        # with tf.name_scope("Decoder_Layer{}".format(len(n_hidden))):
        #     Decoder = tf.nn.relu(tf.add(tf.matmul(Decoder, self.weights["decoder_w{}".format(len(n_hidden))]), self.biases["decoder_b{}".format(len(n_hidden))]))
        #     tf.summary.histogram("layer{}/Decoder".format(len(n_hidden)), Decoder)

        # Loss
        with tf.name_scope("Loss"):
            Loss = tf.reduce_mean(tf.pow(Decoder - Y, 2))
            tf.summary.scalar("Loss", Loss) # In tensorboard event
        
        return Loss, Decoder, Encoder 

    def fc_layer(self, hidden_layer, weights, biases, name):
        with tf.name_scope(name):
            return tf.nn.relu(tf.add(tf.matmul(hidden_layer, weights), biases))

    def fit(self, X, Y, int_Epochs, int_Batch_size, validation_data):
        """
        Run optimization to train the model.

        Inputs:
        - X: Array of data, of shape (N, D)
        - int_Epochs: The number of epochs to run for during training.
        - int_Batch_size: Size of minibatches used to compute loss and gradient during training.
        """

        self.sess.run(self.init)

        N = X.shape[0]
        for epoch in range(int_Epochs):
            start_time = time.time()

            index = [i for i in range(N)]
            np.random.shuffle(index)
            while len(index) > int_Batch_size:
                batch_index = [index.pop() for _ in range(int_Batch_size)]

                _, L = self.sess.run([self.Optimizer, self.Loss], feed_dict = {self.X:X[batch_index],
                                                                               self.Y:Y[batch_index]})
            
            if validation_data is not None:
                Valid_Loss = self.evaluate(validation_data[0], validation_data[1])
                tf.summary.scalar("Valid_Loss", Valid_Loss) # In tensorboard event

             # visulization wieght, biases and loss in tensorboard
            variable_process = self.sess.run(self.merge, feed_dict={self.X:X[batch_index], self.Y:Y[batch_index]})
            self.writer.add_summary(variable_process, epoch)

            print("Epoch:{}".format(epoch + 1), "{:.2f}s".format(time.time()-start_time), "Train Loss:{:.5f}".format(L), ", Valid Loss:{:.5}".format(Valid_Loss))

        return


    def predict(self, X):
        return self.sess.run(self.Decoder, feed_dict={self.X: X})

    def code(self, X):
        return self.sess.run(self.Encoder, feed_dict={self.X: X})

    def evaluate(self, X, Y):
        return self.sess.run(self.Loss, feed_dict={self.X: X, self.Y: Y})

    def save(self, path):
        save_path = self.saver.save(self.sess, path)
        print("Save to path: ", save_path)
    
    def reload(self, path):
        self.saver.restore(self.sess, path)
        print("reload AutoEncoder: ", path)
