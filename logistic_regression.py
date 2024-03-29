import tensorflow as tf
import numpy as np

N = 10000000
NUM_STEPS = 100

def sigmoid(x):
    return 1/(1 + np.exp(-x))

x_data = np.random.randn(N,3)
w_real = [0.3, 0.5, 0.1]
b_real = -0.2
wxb = np.matmul(w_real, x_data.T) + b_real

y_data_pre_noise = sigmoid(wxb)
y_data = np.random.binomial(1, y_data_pre_noise)

#loss function--cross entropy binary version
g = tf.Graph()
wb_ = []

with g.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 3])
    y_true = tf.placeholder(tf.float32, shape=None)

    with tf.name_scope("inference") as scope:
        w = tf.Variable([[0,0,0]], dtype=tf.float32, name="weights")
        b = tf.Variable(0, dtype=tf.float32, name="bias")
        y_pred = tf.matmul(w, tf.transpose(x)) + b

    #sigmoid with logits manual
    # y_pred = tf.sigmoid(y_pred)
    # loss = y_true*tf.log(y_pred) - (1-y_true)*tf.log(1-y_pred)
    # loss = tf.reduce_mean(loss)

    # Loss Function
    with tf.name_scope("loss") as scope:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        loss = tf.reduce_mean(loss)

    # Minimizing the loss function using Gradient Descent Optimization
    with tf.name_scope("train") as scope:
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)

        for step in range(NUM_STEPS):
            sess.run(train, {x: x_data, y_true: y_data})
            
            if(step % 5 == 0):
                res = sess.run([w,b])
                print(step, res)
                wb_.append(res)

        print(NUM_STEPS, sess.run([w,b]))

