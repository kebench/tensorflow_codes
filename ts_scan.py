import tensorflow as tf
import numpy as np

elems = np.array(["T","e","n","s","o","r", " ", "F","l","o","w"])
print(elems) 
scan_sum = tf.scan(lambda a, x: a + x, elems)
sess=tf.InteractiveSession() 
output = sess.run(scan_sum)

print(output)