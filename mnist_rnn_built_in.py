import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# FUNCTIONS
def get_linear_layer(vector):
    return tf.matmul(vector, Wl) + bl

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

element_size = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128

_inputs = tf.placeholder(tf.float32, shape=[None, time_steps, element_size], name="inputs")
y = tf.placeholder(tf.float32, shape=[None, num_classes], name="labels")

rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_layer_size)
outputs, _ = tf.nn.dynamic_rnn(rnn_cell, _inputs, dtype=tf.float32)

Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes], mean = 0, stddev = 0.01))

bl = tf.Variable(tf.truncated_normal([num_classes], mean = 0, stddev=0.01))

last_rnn_output = outputs[:, -1, :]
final_output = get_linear_layer(last_rnn_output)

softmax = tf.nn.softmax_cross_entropy_with_logits(logits = final_output, labels = y)
cross_entropy = tf.reduce_mean(softmax)
train_step = tf.train.RMSPropOptimizer(1e-3, 0.9).minimize(cross_entropy)
 
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(final_output, 1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
 
test_data = mnist.test.images[:batch_size].reshape((-1, time_steps, element_size))
test_label = mnist.test.labels[:batch_size]

for i in range(3000):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape((batch_size, time_steps, element_size))
    feed_dict = {
        _inputs: batch_x,
        y: batch_y
    }

    sess.run(train_step, feed_dict=feed_dict)

    if i % 1000 == 0:
        acc = sess.run(accuracy, feed_dict=feed_dict)
        loss = sess.run(cross_entropy, feed_dict=feed_dict)

        print("Iter "+ str(i) +", Minibatch Loss: {:.6f}".format(loss) + ", Training Accuracy: {:0.5f}".format(acc))

testing_accuracy = sess.run(accuracy, feed_dict={
    _inputs: test_data,
    y: test_label
})
print("Testing Accuracy: {:.6f}".format(testing_accuracy))