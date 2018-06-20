import keras
from keras.datasets import cifar10
from keras import layers
from sklearn import cross_validation
from sklearn.utils import shuffle
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

print("data ready")
#Model definition

model = keras.models.Sequential()
model.add( layers.Conv2D(filters=32, kernel_size= 5, strides=1,padding='same', 
                         activation='relu',  input_shape= x_train.shape[1:] ))
model.add( layers.MaxPooling2D(pool_size=4, strides=2 ) )
model.add( layers.Flatten() )
model.add( layers.Dense(N_HIDDEN, activation='relu'))
model.add( layers.Dense(y_train.shape[-1], activation='softmax'))
sgd= keras.optimizers.SGD(lr=LEARNING_RATE, decay=0., momentum=0., nesterov=False)
#Model compilation or loading
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
print("network created")
print("training begin")
#Model training
model.fit(x_train, y_train,epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)
#Model evaluation
score = tuple(model.evaluate(x_test, y_test, batch_size=BATCH_SIZE))
print("Testing evaluation loss %f acc %f"%score)

