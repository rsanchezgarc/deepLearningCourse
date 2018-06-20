import os
import tensorflow as tf
import numpy as np

#Placeholder(interface variables) definition
x= tf.placeholder(dtype=tf.float32, shape=[3, 3], name="x")
y= tf.placeholder(dtype=tf.float32, shape=[3, 3], name="y")

#Graph computations definition
myResult1= tf.add( x, y)  #same as x + y
myResult2= tf.matmul(myResult1, tf.transpose(x))


#Create session and initialize variables
session = tf.Session()
session.run(tf.global_variables_initializer())

#actual data
x_input= np.array([[1,2,3],[1,2,3],[1,2,3]])
y_input= np.array([[1,1,1],[2,2,2],[3,3,3]])


#Do actual computations
feed_dict= {x: x_input, y:y_input}
res1, res2= session.run([myResult1, myResult2], feed_dict=feed_dict)
print(res1)
print(res2)
session.close()

