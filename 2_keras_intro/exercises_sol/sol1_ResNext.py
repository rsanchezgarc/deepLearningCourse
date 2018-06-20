import tensorflow as tf
import numpy as np
import keras
from keras.datasets import cifar10
from keras import layers
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from keras.regularizers import l2 as l2_regularizer

N_EPOCHS= 20
BATCH_SIZE= 32
LEARNING_RATE= 1e-3
N_LAYERS_BLOCKS= 8
REDUCE_DIM_AT = 4
INIT_N_CHANN=16

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


#Rescale data and substract mean
x_train_mean= np.mean(x_train, axis=0)

x_train= (x_train- x_train_mean) /255.0
x_test=  (x_test-x_train_mean)  /255.0

print(x_train.shape, y_train.shape, x_train.max(), x_train.min() )
#One-hot-encode labels
oh_encoder= OneHotEncoder(n_values=10, sparse=False)
y_train= oh_encoder.fit_transform(y_train)
y_test= oh_encoder.transform(y_test)
print(x_train.shape, y_train.shape)

#No longer needed as keras can split validation automatically
#x_train, x_validation, y_train, y_validation = cross_validation.train_test_split(x_train, y_train, test_size=0.1, random_state=121)

#Advanced topic: use tensorflow directly inside keras we won't discuss now
def roc_auc(y_true, y_pred):
  auc = tf.metrics.auc(y_true, y_pred)[1]
  keras.backend.get_session().run(tf.local_variables_initializer())
  return auc
print("data ready")

#Model definition

def createStreamBlock( outPrevBlock, nChanConv, nChanRatio, reduceDim=False, useBatchNorm=True, kernel_regularizer=None):
  conv_stride= 2 if reduceDim else 1

  #conv1d
  net_out= layers.Conv2D(filters=nChanConv// nChanRatio, kernel_size= 1, strides= conv_stride, padding='same',
                             activation='linear', kernel_regularizer=kernel_regularizer  )(outPrevBlock)
  if useBatchNorm:
    net_out= keras.layers.BatchNormalization()(net_out)
  net_out= keras.layers.Activation("relu")(net_out)

  #conv3d
  net_out= layers.Conv2D(filters=nChanConv// nChanRatio, kernel_size= 3, strides= 1, padding='same',
                             activation='linear', kernel_regularizer=kernel_regularizer  )(net_out)
  if useBatchNorm:
    net_out= keras.layers.BatchNormalization()(net_out)
  net_out= keras.layers.Activation("relu")(net_out)

  #conv1d_2
  net_out= layers.Conv2D(filters= nChanConv, kernel_size= 1, strides=1,padding='same',
                             activation='linear',kernel_regularizer=kernel_regularizer  )(net_out)
  if useBatchNorm:
    net_out= keras.layers.BatchNormalization()(net_out)

  return net_out
   
def createResNextBlock( outPrevBlock, nChanPrevBlock, nStreams= 4, nChanRatio= 8, reduceDim=False, useBatchNorm=True, kernel_regularizer=None):
  assert nChanPrevBlock>nChanRatio, "error, nChanPrevBlock>nChanRatio"
  streamsOut=[]
  nChanConv= 2*nChanPrevBlock if reduceDim else nChanPrevBlock
  for nStream in range(nStreams):
    net_out= createStreamBlock( outPrevBlock, nChanConv, nChanRatio, reduceDim, useBatchNorm, kernel_regularizer)
    streamsOut.append(net_out)
  streamsOut= keras.layers.Add()( streamsOut )
  
  if reduceDim:
     #Reduce image_size and increase channels. can be used 0 padding also, but not implemented directly in keras.
    outPrevBlock= layers.Conv2D(filters=nChanConv, kernel_size= 1, strides=2,padding='same',
                       activation='linear',kernel_regularizer=kernel_regularizer  )(outPrevBlock)
    if useBatchNorm:
      outPrevBlock= keras.layers.BatchNormalization()(outPrevBlock)

  net_out= keras.layers.Add()([streamsOut, outPrevBlock])
  net_out= keras.layers.Activation("relu")(net_out)
  return net_out, nChanConv



net_input= layers.Input( shape= x_train.shape[1:])
net_out= layers.Conv2D(filters=INIT_N_CHANN, kernel_size= 5, strides=1,padding='same',
                           activation='relu'  )(net_input)
net_out= layers.MaxPooling2D(pool_size=4, strides=2 )(net_out)

currNChan= INIT_N_CHANN
for i in range(N_LAYERS_BLOCKS):
  if (i +1) % REDUCE_DIM_AT == 0:
    net_out, new_nChann = createResNextBlock( net_out, nChanPrevBlock= currNChan, reduceDim=True, kernel_regularizer= l2_regularizer(1e-5))
  else:
    net_out, new_nChann = createResNextBlock( net_out, nChanPrevBlock= currNChan, reduceDim=False, kernel_regularizer= l2_regularizer(1e-5))
  currNChan= new_nChann

#net_out= layers.GlobalAveragePooling2D(data_format=None)(net_out) #Used in paper instead FC. Needed deeper network to be effective
net_out= layers.Flatten()(net_out)
net_out= layers.Dense(1024, activation='relu')(net_out)
net_out= layers.Dropout(0.5) (net_out)
net_out= layers.Dense(y_train.shape[-1], activation='softmax')(net_out)

model = keras.models.Model( inputs=net_input, outputs= net_out )

sgd= keras.optimizers.Adam(lr=LEARNING_RATE)
   
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy', roc_auc])
print("network created")
print("training begin")
model.fit(x_train, y_train,epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)
score = tuple(model.evaluate(x_test, y_test, batch_size=BATCH_SIZE))
print("\nTesting evaluation loss %f acc %f roc_auc %f"%score)

