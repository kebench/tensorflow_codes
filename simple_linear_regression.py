import tensorflow as tf
import numpy as np

# #define the variables and placeholders
# x = tf.placeholder(tf.float32, shape=[None,3]) #input
# y_true = tf.placeholder(tf.float32, shape=None) #true labels
# w = tf.Variable([[0,0,0,0]], dtype=tf.float32, name="weights")
# b = tf.Variable(0, dtype=tf.float32, name="bias")

# #simple multivariate linear regression
# y_pred = tf.matmul(w, tf.transpose(x)) + b


# #define a loss function for evaluattion of the model's performance
# #cross entropy
# loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
# loss = tf.reduce_mean(loss)

# #minimize loss function using gradient descent


#exercise
# ============= Simple Linear Regression ===============
#create data
x_data = np.random.randn(20000,3)
w_real = [0.7, 0.3, 0.4]
b_real = -0.2

noise = np.random.randn(1,20000) * 0.1
y_data = np.matmul(w_real, x_data.T) + b_real + noise

NUM_STEPS = 10

g = tf.Graph()
wb_ = []
with g.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 3])
    y_true = tf.placeholder(tf.float32, shape=None)

    with tf.name_scope("inference") as scope:
        w = tf.Variable([[0,0,0]], dtype=tf.float32, name="weights")
        b = tf.Variable(0, dtype=tf.float32, name="bias")
        y_pred = tf.matmul(w, tf.transpose(x)) + b

    #Mean Square Error Loss Function
    with tf.name_scope("loss") as scope:
        loss = tf.reduce_mean(tf.square(y_true-y_pred))

    with tf.name_scope("train") as scope:
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    #INITIALIZE THE VARIABLES!!!
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)

        for step in range(NUM_STEPS):
            sess.run(train, {x: x_data, y_true: y_data})

            if(step % 5 == 0):
                # res = sess.run([w,b])
                print(step, sess.run([w,b]))
                wb_.append(sess.run([w,b]))
        
        #it should be close to the real weights
        print(10, sess.run([w,b]))