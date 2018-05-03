import os
import sys
import numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#from alexnet import AlexNet

def main(_):
    #1. Get Input Vector or Batch
    mnist = input_data.read_data_sets('data/mnist/',one_hot=True)
    print(mnist.train.images.shape, mnist.train.labels.shape)
    print(mnist.test.images.shape, mnist.test.labels.shape)
    print(mnist.validation.images.shape, mnist.validation.labels.shape)

    #2. Construct Compu. Graph
    # 输入和输出张量
    x = tf.placeholder(tf.float32,[None,784])
    y_ = tf.placeholder(tf.float32,[None,10])

    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x,W)+b)

    # 定义损失函数
    cross_entroy = tf.reduce_mean(-tf.reduce_mean(y_ * tf.log(y), reduction_indices=[1]))

    # 定义优化算法
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entroy)

    #3. Training Compu. Graph.
    sess = tf.InteractiveSession()
    #初始化权重数据
    tf.global_variables_initializer().run()

    for i in range(1000):
        batch_xs,batch_ys = mnist.train.next_batch(100)
        train_step.run({x:batch_xs,y_:batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))



# Run as main module.
if __name__ == '__main__':
    tf.app.run()