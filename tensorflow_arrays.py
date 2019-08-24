import tensorflow as tf
import numpy as np

c = tf.constant([[1,2,3],
                [4,5,6]])

print("Python list input: {}".format(c.get_shape()))

c = tf.constant([[[1,2,3],[4,5,6]], 
                [[1,1,1], [2,2,2]]])

print("3d NumPy array Input: {}".format(c.get_shape()))

# One method to avoid constantly refering to the session var is to use InteractiveSession
sess = tf.InteractiveSession()
c = tf.lin_space(0.0, 4.0, 5)
print("The content of 'c':\n {}\n".format(c.eval()))
sess.close()