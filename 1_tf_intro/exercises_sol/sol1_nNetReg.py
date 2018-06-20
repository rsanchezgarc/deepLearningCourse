import tensorflow as tf
import numpy as np
from keras.datasets import boston_housing
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

N_EPOCHS= 2
N_HIDDEN= 64
BATCH_SIZE= 32
LEARNING_RATE= 1e-10  #PLAY with learning rate. try 1e-1, 1e-2 ...

#load data
def generateData1(size=1024):
  x= np.random.rand(size, 3)*10
  y= np.expand_dims( np.sum(x, axis=1) + np.random.rand(size)*.1, axis=-1)
  x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.1, random_state=121)  
  return (x_train, y_train), (x_test, y_test)
  
def generateData2():
  (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
  y_train= np.expand_dims(y_train, axis= -1)
  y_test= np.expand_dims(y_test, axis= -1)
  return (x_train, y_train), (x_test, y_test)
 
(x_train, y_train), (x_test, y_test) = generateData1()



#Normalize data. ( (x-mean)/std )
normalizer= StandardScaler()
x_train=normalizer.fit_transform(x_train)
x_test=normalizer.transform(x_test)

#split train and validation
x_train, x_validation, y_train, y_validation = cross_validation.train_test_split(x_train, y_train, test_size=0.1, random_state=121)
print(x_train.shape, y_train.shape)

#Model definition

inputPh= tf.placeholder(dtype=tf.float32, shape=[None, x_train.shape[1]], name="inputData") #shape= N_Examples x nFeats
labelsPh= tf.placeholder(dtype=tf.float32, shape=[None, 1], name="labelsData")

with tf.variable_scope("hidden_layer"):
  w= tf.get_variable(name="weights", shape=[x_train.shape[1],N_HIDDEN], dtype=tf.float32, #shape= nFeats x N_HIDDEN
                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32, seed=None),
                     regularizer=None, trainable=True)
              
  b= tf.get_variable(name="bias", shape=[N_HIDDEN], dtype=tf.float32,  #shape= N_HIDDEN
                     initializer=tf.constant_initializer(value=0.01, dtype=tf.float32),
                     regularizer=None, trainable=True)
  h1_out= tf.nn.relu( tf.matmul(inputPh,w) + b)
  
with tf.variable_scope("output_layer"):
  w= tf.get_variable(name="weights", shape=[N_HIDDEN,1], dtype=tf.float32, #shape=  N_HIDDEN x 1
                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32, seed=None),
                     regularizer=None, trainable=True)
              
  b= tf.get_variable(name="bias", shape=[1], dtype=tf.float32, 
                     initializer=tf.constant_initializer(value=0.01, dtype=tf.float32),
                     regularizer=None, trainable=True)
  y_pred= tf.matmul(h1_out,w) + b


error = tf.reduce_mean(( tf.square(labelsPh -y_pred) ) ) #shape= N_Examples x 1
#error = tf.losses.mean_squared_error(labelsPh, y_pred) #Equivalent but prefered

optimizer= tf.train.GradientDescentOptimizer(learning_rate= LEARNING_RATE)
#optimizer= tf.train.AdamOptimizer(learning_rate= LEARNING_RATE) #Smarter optimizer

global_step = tf.Variable(0, name='global_step', trainable=False)
train_step = optimizer.minimize(error, global_step=global_step)



session = tf.Session()
session.run(tf.global_variables_initializer())

#FUNCTION TO EVALUATE
def coefficient_of_determination(y_true,y_pred):
  def squared_error(y_true,y_pred):
    return np.sum((y_pred - y_true) * (y_pred - y_true))
  y_mean_pred = [np.mean(y_true) for y in y_true]
  squared_error_regr = squared_error(y_true, y_pred)
  squared_error_y_mean = squared_error(y_true, y_mean_pred)
  return 1 - (squared_error_regr/squared_error_y_mean)
    
nStep=0
for nEpoch in range( N_EPOCHS ):
  x_train, y_train = shuffle(x_train, y_train, random_state=121)
  labels_train= []
  preds_train= []
  for i in range(0, x_train.shape[0], BATCH_SIZE):
    feed_dict= {inputPh: x_train[i:i + BATCH_SIZE, ...], labelsPh: y_train[i:i + BATCH_SIZE]}
    __, y_pred_train, errorExample= session.run([train_step, y_pred, error], feed_dict=feed_dict)
    nStep+=1
    labels_train.append( y_train[i:i + BATCH_SIZE])
    preds_train.append( y_pred_train)

  #EVALUATE VALIDATION DATA
  labels_val= []
  preds_val= []
  for i in range(0, x_validation.shape[0], BATCH_SIZE):
    feed_dict= {inputPh: x_validation[i:i + BATCH_SIZE, ...], labelsPh: y_validation[i:i + BATCH_SIZE]}
    y_pred_val, errorVal= session.run([y_pred, error], feed_dict=feed_dict)  
    labels_val.append( y_validation[i:i + BATCH_SIZE])
    preds_val.append(y_pred_val)
    
  preds_train= np.concatenate(preds_train)
  labels_train= np.concatenate(labels_train)
  train_r2= coefficient_of_determination(labels_train, preds_train)
  
  preds_val= np.concatenate(preds_val)
  labels_val= np.concatenate(labels_val)
  val_r2= coefficient_of_determination(labels_val, preds_val)
  print("Epoch %d. train_r2 %f  val_r2 %f"%(nEpoch, train_r2, val_r2))
 

#REPORT PERFORMANCE ON TEST SET
labels_test= []
preds_test= []
for i in range(0, x_test.shape[0], BATCH_SIZE):
  feed_dict= {inputPh: x_test[i:i + BATCH_SIZE, ...], labelsPh: y_test[i:i + BATCH_SIZE]}
  y_pred_test, errorTest= session.run([y_pred, error], feed_dict=feed_dict)  
  labels_test.append( y_test[i:i + BATCH_SIZE])
  preds_test.append(y_pred_test)
preds_test= np.concatenate(preds_test)
labels_test= np.concatenate(labels_test)
test_r2= coefficient_of_determination(labels_test, preds_test)

print("END.     test_r2 %f"%(test_r2)) 
session.close()

                  
