import os
import tensorflow as tf
import numpy as np

#Placeholder(interface variables) definition
x= tf.placeholder(dtype=tf.float32, shape=[None, 3], name="x")
y= tf.placeholder(dtype=tf.float32, shape=[None, 3], name="y")

#Graph computations definition
def simpleModel(x0,y0):
  results1= tf.add( x0, y0)  #same as x0 + y0
  results2= tf.matmul(results1, tf.transpose(x0))
  return results1, results2
myResult1, myResult2= simpleModel(x,y)

#Create session and initialize variables
session = tf.Session()
session.run(tf.global_variables_initializer())

#Do actual computations
x_input= np.array([[1,2,3],[1,2,3],[1,2,3]])
y_input= np.array([[1,1,1],[2,2,2],[3,3,3]])
feed_dict= {x: x_input, y:y_input}
res1, res2= session.run([myResult1, myResult2], feed_dict=feed_dict)
print(res1)
print(res2)
session.close()

