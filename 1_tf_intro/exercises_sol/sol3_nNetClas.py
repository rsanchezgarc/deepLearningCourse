import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

N_EPOCHS= 10
BATCH_SIZE= 32
LEARNING_RATE= 1e-2  #PLAY with learning rate. try 1e-1, 1e-2 ...
N_HIDDEN= 64

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


#Rescale data.
x_train= x_train /255.0
x_test=  x_test  /255.0

print(x_train.shape, y_train.shape, x_train.max(), x_train.min() )
#One-hot-encode labels
oh_encoder= OneHotEncoder(n_values=10, sparse=False)
y_train= oh_encoder.fit_transform(y_train)
y_test= oh_encoder.transform(y_test)
print(x_train.shape, y_train.shape)

x_train, x_validation, y_train, y_validation = cross_validation.train_test_split(x_train, y_train, test_size=0.1, random_state=121)


#Model definition

inputPh= tf.placeholder(dtype=tf.float32, shape=[None ]+ list(x_train.shape[1:]) , name="inputData")
labelsPh= tf.placeholder(dtype=tf.float32, shape=[None, y_train.shape[-1]], name="labelsData")
tf.summary.image('inputImage', inputPh)
with tf.variable_scope("flatten_layer"):
  input_flatten= tf.reshape(inputPh, [-1, np.prod(x_train.shape[1:])])

with tf.variable_scope("hidden_layer"):
  w= tf.get_variable(name="weights", shape=[input_flatten.get_shape().as_list()[1], N_HIDDEN], dtype=tf.float32, 
                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32, seed=None),
                     regularizer=None, trainable=True)
              
  b= tf.get_variable(name="bias", shape=[N_HIDDEN], dtype=tf.float32, 
                     initializer=tf.constant_initializer(value=0.01, dtype=tf.float32),
                     regularizer=None, trainable=True)
  h1_out= tf.nn.relu( tf.matmul(input_flatten,w) + b)

with tf.variable_scope("output_layer"):
  w= tf.get_variable(name="weights", shape=[h1_out.get_shape().as_list()[1], y_train.shape[-1]], dtype=tf.float32, 
                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32, seed=None),
                     regularizer=None, trainable=True)
              
  b= tf.get_variable(name="bias", shape=[ y_train.shape[-1] ], dtype=tf.float32, 
                     initializer=tf.constant_initializer(value=0.01, dtype=tf.float32),
                     regularizer=None, trainable=True)
  logits= tf.matmul(h1_out,w) + b
  y_pred= tf.nn.softmax(logits)
tf.summary.histogram('weights', w)
#error = -tf.reduce_sum(labelsPh * tf.log(y_pred+ 1e-10), 1)
error = tf.losses.softmax_cross_entropy(labelsPh, logits) #Equivalent but prefered for numerical estability

tf.summary.scalar('loss', error)

optimizer= tf.train.GradientDescentOptimizer(learning_rate= LEARNING_RATE)
#optimizer= tf.train.AdamOptimizer(learning_rate= LEARNING_RATE) #Smarter optimizer

global_step = tf.Variable(0, name='global_step', trainable=False)
train_step = optimizer.minimize(error, global_step=global_step)

session = tf.Session()
session.run(tf.global_variables_initializer())

#Writer to use tensorboard
merged_summary= tf.summary.merge_all()
train_writer = tf.summary.FileWriter("tboard/train",  session.graph)
val_writer = tf.summary.FileWriter("tboard/val",  session.graph)
    
nStep=0
for nEpoch in range( N_EPOCHS ):
  x_train, y_train = shuffle(x_train, y_train, random_state=121)
  labels_train= []
  preds_train= []
  for i in range(0, x_train.shape[0], BATCH_SIZE):
    feed_dict= {inputPh: x_train[i:i + BATCH_SIZE, ...], labelsPh: y_train[i:i + BATCH_SIZE, ...]}
    __, y_pred_train, errorExample, m_summary= session.run([train_step, y_pred, error, merged_summary], feed_dict=feed_dict)
    train_writer.add_summary(m_summary, nStep)
    nStep+=1
    labels_train.append( y_train[i:i + BATCH_SIZE, ...])
    preds_train.append( y_pred_train)

  #EVALUATE VALIDATION DATA
  labels_val= []
  preds_val= []
  for i in range(0, x_validation.shape[0], BATCH_SIZE):
    feed_dict= {inputPh: x_validation[i:i + BATCH_SIZE, ...], labelsPh: y_validation[i:i + BATCH_SIZE, ...]}
    y_pred_val, errorVal, m_summary= session.run([y_pred, error, merged_summary], feed_dict=feed_dict)  
    labels_val.append( y_validation[i:i + BATCH_SIZE, ...])
    preds_val.append(y_pred_val)
  val_writer.add_summary(m_summary, nStep)
  preds_train= np.concatenate(preds_train)
  labels_train= np.concatenate(labels_train)
  train_roc_auc= roc_auc_score(labels_train, preds_train)
  
  preds_val= np.concatenate(preds_val)
  labels_val= np.concatenate(labels_val)
  val_roc_auc= roc_auc_score(labels_val, preds_val)
  print("Epoch %d. train_r2 %f  val_r2 %f"%(nEpoch, train_roc_auc, val_roc_auc))
 

#REPORT PERFORMANCE ON TEST SET
labels_test= []
preds_test= []
for i in range(0, x_test.shape[0], BATCH_SIZE):
  feed_dict= {inputPh: x_test[i:i + BATCH_SIZE, ...], labelsPh: y_test[i:i + BATCH_SIZE, ...]}
  y_pred_test, errorTest= session.run([y_pred, error], feed_dict=feed_dict)  
  labels_test.append( y_test[i:i + BATCH_SIZE, ...])
  preds_test.append(y_pred_test)
preds_test= np.concatenate(preds_test)
labels_test= np.concatenate(labels_test)
test_roc_auc= roc_auc_score(labels_test, preds_test)

print("END.     test_r2 %f"%(test_roc_auc))

train_writer.close()
session.close()

                  
