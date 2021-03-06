import os, sys
import gzip
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from sklearn import cross_validation
import keras
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
LEARNING_RATE= 1e-3
BATCH_SIZE=8
N_EPOCHS= 700

MODEL_TYPE= 2  #1 for encoder-decoder; 2 for Unet-like

seed_ = 42 #MANDATORY to fix seed as we will want image and mask to be equally transformed
random.seed(seed_)

#Load data
with gzip.GzipFile("segmentationData.json.gz", 'r') as f: 
  json_bytes = f.read()                    
  json_str = json_bytes.decode('utf-8')           
  data = json.loads(json_str)       

x_train= np.array(data["X"])
x_train= x_train /255.
y_train= np.array(data["Y"]).astype(np.int32)

print(x_train.shape, y_train.shape)

x_train, x_test, y_train, y_test = cross_validation.train_test_split(
                                                  x_train, y_train, test_size=0.1, random_state=seed_)
                                                  

x_train, x_val, y_train, y_val = cross_validation.train_test_split(
                                                  x_train, y_train, test_size=0.1, random_state=seed_)                                                  

def myCustomImageAugm(one_image):
  '''
  input is np.array of shape (n,m,3)
  '''                            
  if one_image.shape[-1]==3:
    if bool(random.getrandbits(1)):
      #switch colors
      one_image= one_image[:,:, random.sample([0,1,2],3)]
  return one_image
  

data_gen_params = { 'rotation_range':0, 'width_shift_range':0.0,'height_shift_range':0.05,
                    'zoom_range':0.0, 'horizontal_flip':True, 'vertical_flip':True, 'fill_mode':'reflect',
                    'preprocessing_function':myCustomImageAugm }
x_train_datagen = ImageDataGenerator(**data_gen_params)
y_train_datagen = ImageDataGenerator(**data_gen_params)

x_train_datagen.fit(x_train, augment=True, seed= seed_) #same seed needed
y_train_datagen.fit(y_train, augment=True, seed= seed_)
                                                  
#Check if training data looks ok
ix = random.randint(0, len(x_train))
fig= plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(x_train[ix])
fig.add_subplot(1, 2, 2)
plt.imshow(np.squeeze(y_train[ix]))
plt.show()


# Define mean_IoU metric for evaluation
def mean_iou(y_true, y_pred):
  prec = []
  for t in np.arange(0.5, 1.0, 0.05):
    y_pred_ = tf.to_int32(y_pred > t)
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    prec.append(score)
  return K.mean(K.stack(prec), axis=0)
    
if MODEL_TYPE==1:
  inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
  encoder_out = Conv2D(1, (3, 3), activation='relu', padding='same')(inputs)
  encoder_out = MaxPooling2D((2, 2), padding='same')(encoder_out)
  encoder_out = Conv2D(16, (3, 3), activation='relu', padding='same')(encoder_out)
  encoder_out = MaxPooling2D((2, 2), padding='same')(encoder_out)
  encoder_out = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_out)
  encoder_out = MaxPooling2D((2, 2), padding='same', name="encoded_out")(encoder_out)

  decoder_out = Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_out)
  decoder_out = UpSampling2D((2, 2))(decoder_out)
  decoder_out = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_out)
  decoder_out = UpSampling2D((2, 2))(decoder_out)
  decoder_out = Conv2D(8, (3, 3), activation='relu', padding='same')(decoder_out)
  decoder_out = UpSampling2D((2, 2))(decoder_out)
  Y = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name="regular_out")(decoder_out)

  model = Model(inputs= inputs, outputs= Y)



else:   
  # Build U-Net model
  inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

  c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
  c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
  p1 = MaxPooling2D((2, 2)) (c1)

  c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
  c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
  p2 = MaxPooling2D((2, 2)) (c2)

  c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
  c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
  p3 = MaxPooling2D((2, 2)) (c3)

  c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
  c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
  p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

  c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
  c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

  u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
  u6 = concatenate([u6, c4])
  c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
  c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

  u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
  u7 = concatenate([u7, c3])
  c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
  c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

  u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
  u8 = concatenate([u8, c2])
  c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
  c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

  u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
  u9 = concatenate([u9, c1], axis=3)
  c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
  c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

  outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
  model = Model(inputs=[inputs], outputs=[outputs])

adam= keras.optimizers.Adam(lr=LEARNING_RATE)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()


# Fit model
earlystopper = keras.callbacks.EarlyStopping(patience=10, verbose=1)
checkpointer = keras.callbacks.ModelCheckpoint('model-nucle1.hdf5', verbose=1, save_best_only=True)
reduLR = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, cooldown=1, min_lr=0)
#no data augmentation
#results= model.fit(x_train, y_train, validation_data=(x_val, y_val),  batch_size=BATCH_SIZE, 
#                   epochs=N_EPOCHS, callbacks=[earlystopper, checkpointer, reduLR])
#use data augmentation
results= model.fit_generator( zip(x_train_datagen.flow(x_train, batch_size=BATCH_SIZE, seed=seed_), 
                                  y_train_datagen.flow(y_train, batch_size=BATCH_SIZE, seed=seed_)) ,
                              steps_per_epoch=len(x_train) / BATCH_SIZE,
                              validation_data=(x_val, y_val), 
                              epochs=N_EPOCHS, callbacks=[earlystopper, checkpointer, reduLR])
                    
scores= model.evaluate(x_test, y_test)
print("\n++++++++++\nTest loss %f mean_iou %f\n\n"%tuple(scores))

preds_test = model.predict(x_test, verbose=1)
preds_test_t = (preds_test > 0.5).astype(np.uint8)
#Check if training data looks all right
print("show predicted example")
ix = random.randint(0, len(x_test))
fig= plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(x_test[ix])
fig.add_subplot(1, 2, 2)
plt.imshow(np.squeeze(preds_test_t[ix]))
plt.show()

