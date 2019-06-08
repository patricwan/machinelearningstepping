import glob
import random
import time
import numpy as np
import os.path as op
import tensorflow as tf

from model import Model
from utils.general_utils import get_minibatches



class Config(object):
    """Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_samples = 1024
    n_features = 100
    n_classes = 5
    batch_size = 64
    n_epochs = 500
    lr = 1e-4
    
    
class SoftmaxModel(Model):
    """Implements a Softmax classifier with cross-entropy loss."""
    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors.
        """
        self.input_placeholder = tf.placeholder(tf.float32, [self.config.batch_size, self.config.n_features])
        self.labels_placeholder = tf.placeholder(tf.float32, [self.config.batch_size, self.config.n_classes])
        
    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """Creates the feed_dict for training the given step.
        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = {self.input_placeholder: inputs_batch, 
                     self.labels_placeholder: labels_batch}
        return feed_dict
    
    def add_prediction_op(self):
        """Adds the core transformation for this model which transforms a batch of input
        data into a batch of predictions. In this case, the transformation is a linear layer plus a
        softmax transformation:
        y = softmax(Wx + b)
        Args:
            input_data: A tensor of shape (batch_size, n_features).
        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        with tf.variable_scope("transformation"):
            bias = tf.Variable(tf.random_uniform([self.config.n_classes]))
            W = tf.Variable(tf.random_uniform([self.config.n_features, self.config.n_classes]))
            z = tf.matmul(self.input_placeholder, W) + bias
        #pred = softmax(z)
        pred= tf.nn.softmax(z)
        return pred
    
    def add_loss_op(self, pred):
        """Adds cross_entropy_loss ops to the computational graph.
        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar)
        """
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels_placeholder,logits=pred)
        #loss = cross_entropy_loss(self.labels_placeholder, pred)
        return loss
    
    def add_training_op(self, loss):
        """Sets up the training Ops.
        Creates an optimizer and applies the gradients to all trainable variables.
        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        train_op = tf.train.GradientDescentOptimizer(self.config.lr).minimize(loss)
        return train_op
    
    def run_epoch(self, sess, inputs, labels):
        """Runs an epoch of training.
        Args:
            sess: tf.Session() object
            inputs: np.ndarray of shape (n_samples, n_features)
            labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        n_minibatches, total_loss = 0, 0
        for input_batch, labels_batch in get_minibatches([inputs, labels], self.config.batch_size):
            n_minibatches += 1
            total_loss += self.train_on_batch(sess, input_batch, labels_batch)
        return total_loss / n_minibatches
    
    def fit(self, sess, inputs, labels):
        """Fit model on provided data.
        Args:
            sess: tf.Session()
            inputs: np.ndarray of shape (n_samples, n_features)
            labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            losses: list of loss per epoch
        """
        losses = []
        for epoch in range(self.config.n_epochs):
            start_time = time.time()
            average_loss = self.run_epoch(sess, inputs, labels)
            duration = time.time() - start_time
            print('Epoch {}: loss = {} ({:.3f} sec)'.format(epoch, average_loss, duration))
            losses.append(average_loss)
        return losses
 
    def __init__(self, config):
        """Initializes the model.
        Args:
            config: A model configuration object of type Config
        """
        self.config = config
        self.build()
        
from utils.general_utils import test_all_close


def softmax(x):
    """
    Compute the softmax function in tensorflow
    Args:
        x:   tf.Tensor with shape (n_samples, n_features). Note feature vectors are
                  represented by row-vectors. (For simplicity, no need to handle 1-d
                  input as in the previous homework)
    Returns:
        out: tf.Tensor with shape (n_sample, n_features). You need to construct this
                  tensor in this problem.
    """

    ### YOUR CODE HERE
    x_max = tf.reduce_max(x,1,keep_dims=True)          # find row-wise maximums
    x_sub = tf.subtract(x,x_max)                       # subtract maximums
    x_exp = tf.exp(x_sub)                              # exponentiation
    sum_exp = tf.reduce_sum(x_exp,1,keep_dims=True)    # row-wise sums
    out = tf.div(x_exp,sum_exp)                        # divide
    ### END YOUR CODE

    return out


def cross_entropy_loss(y, yhat):
    """
    Compute the cross entropy loss in tensorflow
    Args:
        y:    tf.Tensor with shape (n_samples, n_classes). One-hot encoded.
        yhat: tf.Tensorwith shape (n_sample, n_classes). Each row encodes a
                    probability distribution and should sum to 1.
    Returns:
        out:  tf.Tensor with shape (1,) (Scalar output). You need to construct this
                    tensor in the problem.
    """
    l_yhat = tf.log(yhat)                           # log yhat
    product = tf.multiply(tf.to_float(y), l_yhat)   # multiply element-wise
    out = tf.negative(tf.reduce_sum(product))       # negative summation to scalar

    return out

def test_softmax_basic():
    """
    Some simple tests of softmax to get you started.
    Warning: these are not exhaustive.
    """

    test1 = softmax(tf.constant(np.array([[1001, 1002], [3, 4]]), dtype=tf.float32))
    with tf.Session() as sess:
            test1 = sess.run(test1)
    test_all_close("Softmax test 1", test1, np.array([[0.26894142,  0.73105858],
                                                      [0.26894142,  0.73105858]]))

    test2 = softmax(tf.constant(np.array([[-1001, -1002]]), dtype=tf.float32))
    with tf.Session() as sess:
            test2 = sess.run(test2)
    test_all_close("Softmax test 2", test2, np.array([[0.73105858, 0.26894142]]))

    print("Basic (non-exhaustive) softmax tests pass\n")


def test_cross_entropy_loss_basic():
    """
    Some simple tests of cross_entropy_loss to get you started.
    Warning: these are not exhaustive.
    """
    y = np.array([[0, 1], [1, 0], [1, 0]])
    yhat = np.array([[.5, .5], [.5, .5], [.5, .5]])

    test1 = cross_entropy_loss(
            tf.constant(y, dtype=tf.int32),
            tf.constant(yhat, dtype=tf.float32))
    with tf.Session() as sess:
        test1 = sess.run(test1)
    expected = -3 * np.log(.5)
    test_all_close("Cross-entropy test 1", test1, expected)

    print("Basic (non-exhaustive) cross-entropy tests pass")
    
def test_softmax_model():
    """Train softmax model for a number of steps."""
    config = Config()

    # Generate random data to train the model on
    np.random.seed(1234)
    inputs = np.random.rand(config.n_samples, config.n_features)
    labels = np.zeros((config.n_samples, config.n_classes), dtype=np.int32)
    labels[:, 0] = 1

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Build the model and add the variable initializer Op
        model = SoftmaxModel(config)
        init = tf.global_variables_initializer()

        # Create a session for running Ops in the Graph
        with tf.Session() as sess:
            # Run the Op to initialize the variables.
            sess.run(init)
            # Fit the model
            losses = model.fit(sess, inputs, labels)


if __name__ == "__main__":
    test_softmax_model()
    test_softmax_basic()
    test_cross_entropy_loss_basic()

    