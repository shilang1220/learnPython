import os
import sys
import numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#from alexnet import AlexNet

def main(_)
    #1. Get Input Vector or Batch
    mnist = input_data.read_data_sets('data/mnist/',one_hot=True)
    print(mnist.train.images.shape, mnist.train.labels.shape)
    print(mnist.test.images.shape, mnist.test.labels.shape)
    print(mnist.validation.images.shape, mnist.validation.labels.shape)

    #2. Construct Compu. Graph
    sess = tf.InteractiveSession()

    # 输入和输出张量
    x = tf.placeholder(tf.float32,[None,784])
    y_ = tf.placeholder(tf.float32,[None,10])

    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x,W)+b)

    #3. Execute Compu. Graph.
    sess = tf.Session()
    # Initialize all variables.
    init_op = tf.initialize_all_variables()
    sess.run(w1.init_op)

    print(sess.run(y))

    #4. Get verifing results.

    #5. Inference or predict.

    tf.logging.info('Loading Graph...')

    tf.logging.info('Graph loaded.')

    sess1 = tf.Session()
    sess2 = tf.Session()

    with sess1.as_default():
        print('hello sess1')

    with sess2.as_default():
        print('hello sess2'       )
    #利用计算图training

    #利用计算图verifing

    #保存权重参数

# Run as main module.
if __name__ == '__main__':
    tf.app.run()