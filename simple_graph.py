import tensorflow as tf

#output a constant value
a = tf.to_float(tf.constant(5))
b = tf.to_float(tf.constant(2))
# c = tf.constant(3)

#perform simple arithmetic
#this will create a dependency graph
# d = tf.multiply(a,b)
# e = tf.add(c,b)
# f = tf.subtract(d,e) #printing this will only print a node

#new graph exercise
# d = tf.add(a,b)
# c = tf.multiply(a,b)
# f = tf.add(d,c)
# e = tf.subtract(d,c)
# g = tf.divide(f,e)

c = tf.multiply(a,b)
d = tf.sin(c)
e = tf.divide(b,d)

#session communicates with the python objects and data and the actual computation
sess = tf.Session()
outs = sess.run(e)
nodes = [a,b,c,d,e] #pass the nodes that will be fetched
outs2 = sess.run(nodes)
sess.close()
print("outs = {}".format(outs))
print("outs = {}".format(outs2))

# print(tf.get_default_graph())
#Create new Graph.
#operations will not be associated with the new graph unless it is set to the default graph
# g = tf.Graph()
# print(g)

# a = tf.constant(6)

# print(a.graph is g)
# print(a.graph is tf.get_default_graph())

# g1 = tf.get_default_graph()
# g2 = tf.Graph()

# print(g1 is tf.get_default_graph())

# #using with keyword and as_default will set the graph to g2 temporarily.
# #All of the code within the scope of with-as_default() will be assigned to the new graph
# with g2.as_default():
#     print(g1 is tf.get_default_graph())
#     a = tf.constant(4)
#     print(a.graph is g2)

# print(g1 is tf.get_default_graph)
# print(a.graph is g2)


