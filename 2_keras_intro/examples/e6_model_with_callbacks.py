import tensorflow as tf
import numpy as np
import keras
from keras.datasets import cifar10
from keras import layers
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from keras.regularizers import l2 as l2_regularizer

N_EPOCHS= 20
BATCH_SIZE= 32
LEARNING_RATE= 1e-4
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




#Model definition

def createResidualBlock( outPrevBlock, nChanPrevBlock, reduceDim=False, useBatchNorm=True, kernel_regularizer=None):

  conv_stride= 2 if reduceDim else 1
  nChanConv= 2*nChanPrevBlock if reduceDim else nChanPrevBlock
  net_out= layers.Conv2D(filters=nChanConv, kernel_size= 3, strides= conv_stride, padding='same',
                             activation='linear', kernel_regularizer=kernel_regularizer  )(outPrevBlock)
  if useBatchNorm:
    net_out= keras.layers.BatchNormalization()(net_out)
  net_out= keras.layers.Activation("relu")(net_out)
  net_out= layers.Conv2D(filters= nChanConv, kernel_size= 3, strides=1,padding='same',
                             activation='linear',kernel_regularizer=kernel_regularizer  )(net_out)
  if useBatchNorm:
    net_out= keras.layers.BatchNormalization()(net_out)
  if reduceDim:
     #Reduce image_size and increase channels. can be used 0 padding also, but not implemented directly in keras.
    outPrevBlock= layers.Conv2D(filters=nChanConv, kernel_size= 1, strides=2,padding='same',
                       activation='linear',kernel_regularizer=kernel_regularizer  )(outPrevBlock)
    if useBatchNorm:
      outPrevBlock= keras.layers.BatchNormalization()(outPrevBlock)

  net_out= keras.layers.Add()([net_out, outPrevBlock])
  net_out= keras.layers.Activation("relu")(net_out)
  return net_out, nChanConv



net_input= layers.Input( shape= x_train.shape[1:])
net_out= layers.Conv2D(filters=INIT_N_CHANN, kernel_size= 5, strides=1,padding='same',
                           activation='relu', name= "initial_conv")(net_input)
net_out= layers.MaxPooling2D(pool_size=4, strides=2 )(net_out)

currNChan= INIT_N_CHANN
for i in range(N_LAYERS_BLOCKS):
  if (i +1) % REDUCE_DIM_AT == 0:
    net_out, new_nChann = createResidualBlock( net_out, nChanPrevBlock= currNChan, reduceDim=True, kernel_regularizer= l2_regularizer(1e-5))
  else:
    net_out, new_nChann = createResidualBlock( net_out, nChanPrevBlock= currNChan, reduceDim=False, kernel_regularizer= l2_regularizer(1e-5))
  currNChan= new_nChann

#net_out= layers.GlobalAveragePooling2D(data_format=None)(net_out) #Used in paper instead FC. Needed deeper network to be effective
net_out= layers.Flatten()(net_out)
net_out= layers.Dense(1024, activation='relu')(net_out)
net_out= layers.Dropout(0.5) (net_out)
net_out= layers.Dense(y_train.shape[-1], activation='softmax')(net_out)

model = keras.models.Model( inputs=net_input, outputs= net_out )

sgd= keras.optimizers.Adam(lr=LEARNING_RATE)

model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
print("network created")
print("training begin")

callbacksList=  [keras.callbacks.ModelCheckpoint("myBestModel.hdf5", monitor='val_loss', verbose=1, save_best_only=True, period=1)]
callbacksList+= [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=0.0001, cooldown=1, min_lr=1e-10)]
callbacksList+= [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)]

tf.summary.image("input images", model.get_layer("initial_conv").input)
tf.summary.image("first layer output filter0", model.get_layer("initial_conv").output[...,0:1]) #input must be a 4-D tensor with last dimension 1 or 3

callbacksList+= [keras.callbacks.TensorBoard(log_dir='./logs', batch_size=BATCH_SIZE, histogram_freq=1, write_graph=False, write_grads=False, write_images=False)]

model.fit(x_train, y_train,epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks= callbacksList)

del model
#Dont use custom metric if you are going to checkpoint full model. Save weights instead
model= keras.models.load_model("myBestModel.hdf5")
score = tuple(model.evaluate(x_test, y_test, batch_size=BATCH_SIZE))
print("\nTesting evaluation loss %f acc %f "%score)

