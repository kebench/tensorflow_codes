import tensorflow as tf
import numpy as np

#tensor variables can maintain a fixed state in the graph.
#variables can also be used as input to the operations

# init_val = tf.random_normal((1,5),0,1)
# var = tf.Variable(init_val, name='var')
# print("Pre run: {}\n".format(var))

# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     post_var = sess.run(var)

# print("Post Run: {}\n".format(post_var))

#placeholders are built-in structure for feeding input values
#think of it as an empty variable that will be filled with data later on.
#ph = tf.placeholder(tf.float32, shape=(None, 10))

x_data = np.random.randn(5,10)
w_data = np.random.randn(10,1)

with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, shape=(5,10)) #matrix
    w = tf.placeholder(tf.float32, shape=(10,1)) #vector
    b = tf.fill((5,1), -1.) #constant vector filled with -1 values
    xw = tf.matmul(x,w)

    xwb = xw + b
    s = tf.reduce_max(xwb)
    
    with tf.Session() as sess:
        outs = sess.run(s, feed_dict={x: x_data, w: w_data}) # returns the max of the matrix result from matrix addition
        outs2 = sess.run(xwb, feed_dict={x: x_data, w: w_data}) #returns a matrix

print("outs = {}".format(outs))
print("outs2 = {}".format(outs2))