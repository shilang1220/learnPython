import  tensorflow as tf
from numpy.random import RandomState

'''两层网络范例，无激活、批量归一化等操作
'''

#0.定义计算图
x = tf.placeholder(tf.float32,[None,2])
y_ = tf.placeholder(tf.float32,[None,1])

w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#定义损失函数
loss_cross_entropy= - tf.reduce_mean(y_*tf.log(y))
learning_rate = 0.01
train_step= tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_cross_entropy)

#1. 训练
batch_size = 10
dataset_size = 1280
epochs = 300

#1.1 input data
rdm = RandomState(1)
input_x = rdm.rand(dataset_size,2)
input_y = [[int(x1+x2<1)] for (x1,x2) in input_x]

with tf.Session() as session:
#变量初始化
    init_ops = tf.global_variables_initializer()
    session.run(init_ops)

    for i in range(epochs):
        start = 0

        for j in range( dataset_size // batch_size):
            #1. 定义小批次数据
            start = (j * batch_size) % dataset_size
            end = min(start + batch_size,dataset_size)
            #2. 训练
            session.run(train_step,feed_dict={x:input_x[start:end],y_:input_y[start:end]})

        #每一轮输出一次交叉熵
        total_cross_entroy = session.run(loss_cross_entropy,feed_dict={x:input_x,y_:input_y})
        print('After %d epochs, the cross_entroy is %g' % (i,total_cross_entroy))

    print(session.run(w1))
    print(session.run(w2))


