import tensorflow as tf
#I'm a lazy man so I let tensorflow fo the downloading and partitioning of the data
from tensorflow.examples.tutorials.mnist import input_data

#constants
DATA_DIR = '/tmp/data' #location to save the data
NUM_STEPS = 1000 
MINIBACTH_SIZE = 100

#download dataset and save it locally
data = input_data.read_data_sets(DATA_DIR, one_hot=True)

# Placeholder has to be supplied while triggering the computation. This is the image
# A single image of size 784 (28x28 pixels) rolled into a vector
# None is an indicator that we are not currently specifying how many of these images we will use at once
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10])) #element manipulated by the computation

#True label -- this should be the right label for images
y_true = tf.placeholder(tf.float32, [None, 10])
#predicted label -- this is the predicted label by the model
y_pred = tf.matmul(x, W)

#we choose this since the model outputs class probabilities --- One of the loss functions
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

# train the model through Gradient Descent with a learning rate of 0.5 while minimizing the losses
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Define the evaluation procedure to test the accuracy of the model
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

# use session to use the computation graph
with tf.Session() as sess:
    #train
    sess.run(tf.global_variables_initializer())

    #loop the steps
    for _ in range(NUM_STEPS):
        #ask fo the next batch of 100s for the next step
        batch_xs, batch_ys = data.train.next_batch(MINIBACTH_SIZE)
        #
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

    # supply test images to the model
    ans = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans * 100))
