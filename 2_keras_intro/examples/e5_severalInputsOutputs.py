import tensorflow as tf
import numpy as np
import keras
from keras.datasets import cifar10
from keras import layers
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

N_EPOCHS= 20
BATCH_SIZE= 32
LEARNING_RATE= 1e-2  #PLAY with learning rate. try 1e-1, 1e-2 ...

def makeFakeMetadata(data):
  '''
  This could be date, country...
  '''
  return np.stack( [np.mean(data, axis=(1,2,3)), 
                    np.std( data, axis=(1,2,3) ) ], axis=1)
  
def makeFakeLabel(label):
  return label % 2

(x1_train, y1_train), (x1_test, y1_test) = cifar10.load_data()
x2_train= makeFakeMetadata( x1_train)
x2_test= makeFakeMetadata( x1_test)
print(x2_train.shape)
#Rescale data.
x1_train= x1_train /255.0
x1_test=  x1_test  /255.0
x2_train= x2_train /255.0
x2_test=  x2_test  /255.0

y2_train= makeFakeLabel(y1_train)
y2_test= makeFakeLabel(y1_test)
#One-hot-encode labels
oh1_encoder= OneHotEncoder(n_values=10, sparse=False)
y1_train= oh1_encoder.fit_transform(y1_train)
y1_test= oh1_encoder.transform(y1_test)
oh2_encoder= OneHotEncoder(n_values=5, sparse=False)
y2_train= oh2_encoder.fit_transform(y2_train)
y2_test= oh2_encoder.transform(y2_test)

#Model definition

net_input_1= layers.Input( shape= x1_train.shape[1:])
net_input_2= layers.Input( shape= x2_train.shape[1:])
net_out= layers.Conv2D(filters=32, kernel_size= 5, strides=1,padding='same',
                           activation='relu'  )(net_input_1)
net_out= layers.MaxPooling2D(pool_size=4, strides=2 )(net_out)

#branch1
net_out1= layers.Conv2D(filters=64, kernel_size= 3, strides=1,padding='same',
                           activation='relu'  )(net_out)
net_out1= layers.MaxPooling2D(pool_size=4, strides=2 )(net_out1)
net_out1= layers.Flatten()(net_out1)
net_out1= layers.Concatenate()( [net_out1, net_input_2] ) #additional input inserted here
net_out1= layers.Dense(256, activation='relu')(net_out1)
net_out1= layers.Dense(y1_train.shape[-1], activation='softmax', name="softmax_out1")(net_out1) #name is mandatory if several outpus

#branch2
net_out2= layers.Flatten()(net_out)
net_out2= layers.Dense(128, activation='relu')(net_out2)
net_out2= layers.Dense(y2_train.shape[-1], activation='softmax', name="softmax_out2")(net_out2) #name is mandatory if several outpus


model = keras.models.Model( inputs=[net_input_1, net_input_2], outputs= [net_out1, net_out2] )

sgd= keras.optimizers.SGD(lr=LEARNING_RATE, decay=0., momentum=0., nesterov=False)

losses = {
	"softmax_out1": "categorical_crossentropy",
	"softmax_out2": "categorical_crossentropy",
}
lossWeights = {"softmax_out1": 2.0, "softmax_out2": 1.0}
   
model.compile(loss=losses, loss_weights=lossWeights, optimizer=sgd, metrics=['accuracy'])
print("network created")
print("training begin")
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)
score = tuple(model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=BATCH_SIZE))
print(score)
print("\nTesting evaluation loss1 %f loss2 %f acc1 %f acc2 %f"%score[1:])

