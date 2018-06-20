import keras
from keras.datasets import cifar10
from keras import layers
from keras.utils import Sequence
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np

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

x_train, x_validation, y_train, y_validation = cross_validation.train_test_split(
                                                  x_train, y_train, test_size=0.1, random_state=121)

DATA_MANAGER_OPTION=1 #Can be 1 or 2

#OPTION 1
def generateBatches(data, labels):
  '''
  This is a dummy example, here you will be loading files,
  applying transformations, etc.
  '''
  while True:
    data, labels= shuffle(data, labels)
    for i in range(0, data.shape[0], BATCH_SIZE):
      batch_x= data[i:i+BATCH_SIZE, ...]
      #transform batch_x or apply data augmentation here
      batch_y= labels[i:i+BATCH_SIZE, ...]
      yield batch_x, batch_y
      #The following line is a possible real example in which data are filenames
      #return np.array([ resize(imread(file_name), (200, 200))
      #          for file_name in batch_x]), np.array(batch_y)
      
trainGenerator= generateBatches(x_train, y_train)
nBatchesPerEpochTrain= x_train.shape[0]//BATCH_SIZE
validationGenerator= generateBatches(x_validation, y_validation)
nBatchesPerEpochValid= x_validation.shape[0]//BATCH_SIZE
testGenerator= generateBatches(x_test, y_test)
nBatchesPerEpochTest= x_test.shape[0]//BATCH_SIZE

#OPTION 2
class myDataManger(Sequence):
  def __init__(self, x_set, y_set, batch_size= BATCH_SIZE):
    self.x, self.y = x_set, y_set
    self.batch_size = batch_size

  def __len__(self):
    return int(np.ceil(len(self.x) / float(self.batch_size)))

  def __getitem__(self, idx):
    '''
      idx is the index of a given example
    '''
    batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
    batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
    return batch_x, batch_y
    #The following line is a possible real example in which self.x are filenames
    #return np.array([ resize(imread(file_name), (200, 200)) for file_name in batch_x]), np.array(batch_y)
           

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
if DATA_MANAGER_OPTION==1:
  #OPTION 1
  model.fit_generator(trainGenerator, steps_per_epoch=nBatchesPerEpochTrain, epochs=N_EPOCHS,
                      validation_data= validationGenerator, validation_steps=nBatchesPerEpochValid, 
                      workers=2, max_queue_size=10, use_multiprocessing=True )
if DATA_MANAGER_OPTION==2:
  #OPTION 2
  model.fit_generator(myDataManger(x_train, y_train), epochs=N_EPOCHS,
                      validation_data= myDataManger(x_validation, y_validation), 
                      workers=2, max_queue_size=10 )
                                      
#Model evaluation
score = tuple(model.evaluate_generator(testGenerator, steps=nBatchesPerEpochTest) )
print("Testing evaluation loss %f acc %f "%score)

