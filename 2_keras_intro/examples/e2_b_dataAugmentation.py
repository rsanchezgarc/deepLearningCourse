import keras
from keras.datasets import cifar10
from keras import layers
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import random
#from matplotlib import pyplot as plt #To visualize images

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
AUGMENTATION_OPTION=2 #Can be 1 or 2

if AUGMENTATION_OPTION==1:
  #OPTION 1: AUTOMATIC AUGMENTATION. Data need to fit in memory or being stored in one directory per class
  #Image augmentation
  dataAugmentator = ImageDataGenerator(rotation_range=10.0, horizontal_flip=True, fill_mode='reflect')

else:
  #OPTION 2: MANUAL AUGMENTATION. If you want to use generators
  from skimage.transform import rotate
  class myDataManger(Sequence):
    def __init__(self, x_set, y_set, batch_size= BATCH_SIZE, augment=True):
      self.x, self.y = x_set, y_set
      self.batch_size = batch_size
      if augment:
        self.augmentBatch= self._augmentBatch
      else:
        self.augmentBatch= lambda arg: arg
    def __len__(self):
      return int(np.ceil(len(self.x) / float(self.batch_size)))

    def _random_flip_leftright(self, batch):
      for i in range(len(batch)):
        if bool(random.getrandbits(1)):
          batch[i] = np.fliplr(batch[i])
      return batch

    def _random_flip_updown(self, batch):
      for i in range(len(batch)):
        if bool(random.getrandbits(1)):
          batch[i] = np.flipud(batch[i])
      return batch

    def _random_90degrees_rotation(self, batch, rotations=[0, 1, 2, 3]):
      for i in range(len(batch)):
        num_rotations = random.choice(rotations)
        batch[i] = np.rot90(batch[i], num_rotations)
      return batch

    def _random_rotation(self, batch, max_angle=15.0):
      for i in range(len(batch)):
        if bool(random.getrandbits(1)):
          # Random angle
          angle = random.uniform(-max_angle, max_angle)
#          fig=plt.figure(figsize=(2,1))
#          fig.add_subplot(2,1,1)
#          plt.imshow(batch[i])          
          batch[i] = rotate(batch[i], angle, mode="reflect")
#          fig.add_subplot(2,1,2)          
#          plt.imshow(batch[i])
#          plt.show()
      return batch

    def _augmentBatch(self, batch):
      if bool(random.getrandbits(1)):
        batch= self._random_flip_leftright(batch)
      if bool(random.getrandbits(1)):
        batch= self._random_rotation(batch, 10.0)
      return batch
      
    def __getitem__(self, idx):
      '''
        idx is the index of a given example
      '''
      batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
      batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
      return self.augmentBatch(batch_x), batch_y
      
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

if AUGMENTATION_OPTION==1:
  #When data is located as png, jpg, bmp, files ImageDataGenerator can use then directly with 
  #dataAugmentator.flow_from_directory
  model.fit_generator(dataAugmentator.flow(x_train, y_train), epochs=N_EPOCHS,
                      validation_data= (x_validation, y_validation), 
                      workers=2, max_queue_size=10, use_multiprocessing=True )
else:
  model.fit_generator(myDataManger(x_train, y_train, augment=True), epochs=N_EPOCHS,
                      validation_data= (x_validation, y_validation), 
                      workers=2, max_queue_size=10, use_multiprocessing=True )
#Model evaluation
score = tuple(model.evaluate(x_test, y_test) )
print("Testing evaluation loss %f acc %f"%score)

