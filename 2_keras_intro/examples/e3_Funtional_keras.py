import tensorflow as tf
import numpy as np
import keras
from keras.datasets import cifar10
from keras import layers
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

N_EPOCHS= 20
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

#No longer needed as keras can split validation automatically
#x_train, x_validation, y_train, y_validation = cross_validation.train_test_split(x_train, y_train, test_size=0.1, random_state=121)

#Advanced topic: use tensorflow directly inside keras we won't discuss now
def roc_auc(y_true, y_pred):
  auc = tf.metrics.auc(y_true, y_pred)[1]
  keras.backend.get_session().run(tf.local_variables_initializer())
  return auc
print("data ready")

#Model definition

net_input= layers.Input( shape= x_train.shape[1:])
net_out= layers.Conv2D(filters=32, kernel_size= 5, strides=1,padding='same',
                           activation='relu'  )(net_input)
net_out= layers.MaxPooling2D(pool_size=4, strides=2 )(net_out)
net_out= layers.Flatten()(net_out)
net_out= layers.Dense(N_HIDDEN, activation='relu')(net_out)
net_out= layers.Dense(y_train.shape[-1], activation='softmax')(net_out)

model = keras.models.Model( inputs=net_input, outputs= net_out )

sgd= keras.optimizers.SGD(lr=LEARNING_RATE, decay=0., momentum=0., nesterov=False)
   
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy', roc_auc])
print("network created")
print("training begin")
model.fit(x_train, y_train,epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)
score = tuple(model.evaluate(x_test, y_test, batch_size=BATCH_SIZE))
print("\nTesting evaluation loss %f acc %f roc_auc %f"%score)

