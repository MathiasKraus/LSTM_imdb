#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 15:33:27 2016

@author: mkraus
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
#import imdb
from keras.datasets import imdb              #loading imdb data
#from keras.preprocessing import sequence     #padding data

def to_one_hot(y):
    '''class 0 -> vector [1.0, 0.0]
       class 1 -> vector [0.0, 1.0]'''
    y_one_hot = []
    for row in y:
        if row == 0:
            y_one_hot.append([1.0, 0.0])
        else:
            y_one_hot.append([0.0, 1.0])
    return np.array([np.array(row) for row in y_one_hot])

class RNN(object):
    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.n_classes = n_classes = config.n_classes
        size = config.hidden_size
        vocab_size = config.vocab_size
    
        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._target = tf.placeholder(tf.float32, [batch_size, n_classes])
    
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
    
        self._initial_state = cell.zero_state(batch_size, tf.float32)
    
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)
    
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
    
        output, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=self._initial_state)
    
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)
        softmax_w = tf.get_variable("softmax_w", [size, n_classes], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [n_classes], dtype=tf.float32)
        logits = tf.matmul(last, softmax_w) + softmax_b
        
        self._cost = cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self._target))
        self._final_state = state
    
        correct_pred = tf.equal(tf.argmax(self._target, 1), tf.argmax(logits, 1))
        self._accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        if not is_training:
            return
    
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),config.max_grad_norm)
#        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = tf.train.AdamOptimizer().minimize(cost)
#        self._train_op = optimizer.apply_gradients(zip(grads, tvars))
    
        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def target(self):
        return self._target

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost
    
    @property
    def accuracy(self):
        return self._accuracy

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class Config(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 1
    num_steps = 80
    hidden_size = 128
    max_epoch = 10      #keep learning rate at initial value for max_epoch iterations
    max_max_epoch = 10
    keep_prob = 0.5
    lr_decay = 0.9
    batch_size = 32
    vocab_size = 20000
    n_classes = 2

def fill_feed_dict(data_X, data_Y, batch_size):
    '''Generator to yield batches'''
    #Shuffle data first.
    perm = np.random.permutation(data_X.shape[0])
    data_X = data_X[perm]
    data_Y = data_Y[perm]
    for idx in range(data_X.shape[0] // batch_size):
        x_batch = data_X[batch_size * idx : batch_size * (idx + 1)]
        y_batch = data_Y[batch_size * idx : batch_size * (idx + 1)]
        yield x_batch, y_batch

def run_epoch(session, model, x_data, y_data, eval_op, verbose=False):
    """
    Runs the model on the given data.
    We do not update the state of the modell but have a fixed sized (padded) input. To feed in sequences
    of arbitrary length, we have to use truncated backpropagation. After each of these truncated inputs,
    we update the state of the LSTM.
    
    A B C D E            F G H I J K
    ---------            -----------
            |            ^
            |____________|
                 state
    """
    step = 0
    acc_accuracy = 0
    for x_batch, y_batch in fill_feed_dict(x_data, y_data, model.batch_size):
        
        fetches = [model.accuracy, model.cost, eval_op]
        feed_dict = {}
        feed_dict[model.input_data] = x_batch
        feed_dict[model.target] = y_batch
        accuracy, cost, _ = session.run(fetches, feed_dict)
        step += 1
        acc_accuracy += accuracy
        if verbose:
            accuracy = session.run(model.accuracy, feed_dict)
            print("step %d \t cost: %.3f \t accuracy: %.3f" % (step, cost, accuracy))
    return acc_accuracy / step

def filter_and_cut_by_length(X_data, y_data, length, max_length):
    X_filtered = []
    y_filtered = []
    for x, y in zip(X_data, y_data):
        if len(x) < length or len(x) > max_length:
            continue
        X_filtered.append(x[-length:])
        y_filtered.append(y)
    return np.array([np.array(x) for x in X_filtered]), np.array(y_filtered)
        
if __name__ == "__main__":
    config = Config()
    maxlen = 80
    print('Loading data...')
    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=config.vocab_size)
    y_train = to_one_hot(y_train)
    y_test = to_one_hot(y_test)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

#    print('Pad sequences (samples x time)')
#    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
#    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    X_train, y_train = filter_and_cut_by_length(X_train,y_train,80,np.inf)
    X_test, y_test = filter_and_cut_by_length(X_test,y_test,80,np.inf)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
        
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = RNN(is_training=True, config=config)
            
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mtest = RNN(is_training=False, config=config)

        tf.initialize_all_variables().run()

        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_accuracy = run_epoch(session, m, X_train, y_train, m.train_op,verbose=False)
            val_accuracy = run_epoch(session, mtest, X_test, y_test, tf.no_op())
            print("Epoch: %d \t Train Accuracy: %.3f \t Valid Accuracy: %.3f" % (i + 1, train_accuracy, val_accuracy))          

#        test_accuracy = run_epoch(session, mtest, x_val, y_val, tf.no_op())
#        print("Test Accuracy: %.3f" % test_accuracy)