import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

# this helper function, taken from official tensorflow documentation
# simply adds some ops that take care of logging summaries
def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)

def rnn_step(previous_hidden_state, x):
    current_hidden_state = tf.tanh(
        tf.matmul(previous_hidden_state, Wh) + 
        tf.matmul(x, Wx) + b_rnn
    )

    return current_hidden_state

# applies linear layer to the state vector
def get_linear_layer(hidden_state):
    return tf.matmul(hidden_state, Wl) + bl


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

element_size = 28 #dimension of each vector in the sequence
time_steps = 28 #number of elements in a sequence
num_classes = 10 #
batch_size = 128
hidden_layer_size = 128

LOG_DIR = os.path.join(os.getcwd(), "models/mnsit_rnn")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

_inputs = tf.placeholder(tf.float32, shape=[None, time_steps, element_size], name="inputs")
y = tf.placeholder(tf.float32, shape=[None, num_classes], name="labels")

with tf.name_scope("rnn_weights"):
    with tf.name_scope("W_x"):
        Wx = tf.Variable(tf.zeros([element_size, hidden_layer_size]))
        variable_summaries(Wx)
    with tf.name_scope("W_h"):
        Wh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
        variable_summaries(Wh)
    with tf.name_scope("Bias"):
        b_rnn = tf.Variable(tf.zeros([hidden_layer_size]))
        variable_summaries(b_rnn)

#processing input to work with the scan function
#current input shape: (batch_size, time_steps, element_size)
processed_input = tf.transpose(_inputs, perm=[1, 0, 2])
#current input shape now: (time steps, batch_size, element_size)

initial_hidden = tf.zeros([batch_size, hidden_layer_size])
#getting all state vectors accross time
all_hidden_states = tf.scan(rnn_step, processed_input, initializer=initial_hidden, name="States")

# Weight for output layers
with tf.name_scope("linear_layer_weights"):
    with tf.name_scope("W_linear"):
        Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes], mean=0, stddev=.01))
        variable_summaries(Wl)
    with tf.name_scope("Bias_linear"):
        bl = tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=.01))
        variable_summaries(bl)

with tf.name_scope("linear_layer_weights"):
    #Iterate accross time and apply linear layer to all RNN outputs
    all_outputs = tf.map_fn(get_linear_layer, all_hidden_states)
    # Get last output
    output = all_outputs[-1]
    tf.summary.histogram("outputs", output)

with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    tf.summary.scalar("cross_entropy", cross_entropy)

with tf.name_scope("train"):
    # Uses RMSPropOptimizer
    # Minimize the error
    train_step = tf.train.RMSPropOptimizer(1e-3, 0.9).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100
    tf.summary.scalar("accuracy", accuracy)

# merge all summary
merged = tf.summary.merge_all()

########### TRAIN AND TEST ###################
test_data = mnist.test.images[:batch_size].reshape((-1, time_steps, element_size))
test_label = mnist.test.labels[:batch_size]

with tf.Session() as sess:
    #Write Summaries to LOG_DIR
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test', graph=tf.get_default_graph())

    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        batch_x, batch_y = mnist.train.next_batch(128)
        #reshape data to get 28 sequence of 28 pixels
        batch_x = batch_x.reshape((batch_size, time_steps, element_size))

        summary, _ = sess.run([merged, train_step], feed_dict={_inputs: batch_x, y: batch_y})

        train_writer.add_summary(summary, i)

        if(i % 1000 == 0):
            acc, loss = sess.run([accuracy, cross_entropy], feed_dict={_inputs: batch_x, y: batch_y})
            print("Iter " + str(i) + ", Minibatch Loss {:.6f}".format(loss) + ", Training Accuracy {:.5f}".format(acc))

        if(i % 10):
            #calculate accuracy for MNIST test images and add to summary
            summary, acc = sess.run([merged, accuracy],  feed_dict={_inputs: test_data, y: test_label})
            test_writer.add_summary(summary, i)

    test_acc = sess.run(accuracy,  feed_dict={_inputs: test_data, y: test_label})

print("Test Accuracy: ", test_acc)
