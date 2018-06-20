
import numpy as np
import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Model
from keras.datasets import mnist
from keras import layers

N_EPOCHS= 1
BATCH_SIZE= 32
LEARNING_RATE= 1e-3  #PLAY with learning rate. try 1e-1, 1e-2 ...
N_CLASSES= 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train= np.expand_dims(x_train,-1)
x_test= np.expand_dims(x_test,-1)

#Rescale data.
x_train= x_train /255.0
x_test=  x_test  /255.0

print("data ready")
#Model definition

input_img = Input(shape=(28, 28, 1))

encoder_out = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
encoder_out = MaxPooling2D((2, 2), padding='same')(encoder_out)
encoder_out = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder_out)
encoder_out = MaxPooling2D((2, 2), padding='same')(encoder_out)
encoder_out = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder_out)
encoder_out = MaxPooling2D((2, 2), padding='same', name="encoded_out")(encoder_out)
Z= Flatten()( encoder_out )


decoder_out = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder_out)
decoder_out = UpSampling2D((2, 2))(decoder_out)
decoder_out = Conv2D(8, (3, 3), activation='relu', padding='same')(decoder_out)
decoder_out = UpSampling2D((2, 2))(decoder_out)
#decoder_out = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_out)
decoder_out = Conv2D(16, (3, 3), activation='relu', padding='valid')(decoder_out) #To make it 28x28 at the end
decoder_out = UpSampling2D((2, 2))(decoder_out)
Y = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name="regular_out")(decoder_out)

autoencoder = Model(inputs= input_img, outputs= [Y, Z])
losses = {"regular_out": "binary_crossentropy"} #as we have mormalized input to 0-1 range

adam= keras.optimizers.Adam(lr=LEARNING_RATE)
autoencoder.compile(optimizer= adam, loss=losses)
print(autoencoder.summary())
autoencoder.fit(x_train, x_train,
                epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                shuffle=True,validation_split=0.1 )

score = tuple(autoencoder.evaluate(x_test, x_test, batch_size=BATCH_SIZE))
print("Testing evaluation,", score)

predictions= autoencoder.predict(x_test)
# reconstruction  / compressed representation
print(predictions[0].shape, predictions[1].shape)


#VISUALIZE X and Y
from matplotlib import pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(predictions[0][i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#VISUALIZE clusters
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.cm as cm

encoded_states= predictions[1]
pred_labels= KMeans(n_clusters=N_CLASSES, n_jobs=-1).fit_predict(encoded_states)
data_smallDim= PCA(n_components=2).fit_transform(encoded_states)

colors = cm.rainbow(np.linspace(0, 1, N_CLASSES))
colorsList= np.array([ colors[elem] for elem in pred_labels])

print(data_smallDim.shape)

def generateOnPick3(x):
  def onpick3(event):
      ind = event.ind
      ind=ind[0]
      print('onpick3 scatter:', ind)
      plt.figure(2)
      imFig= plt.axes()
      imFig.imshow(x[ind,...].squeeze(), cmap='gray')
      plt.draw()
      print("press enter to continue")
  return onpick3
x,y= zip(*data_smallDim)
fig, ax = plt.subplots()
plt.figure(2)
col = ax.scatter(x, y, c=colorsList, lw=0, picker=True)
fig.canvas.mpl_connect('pick_event', generateOnPick3(x_test))
plt.show()

a= input("press enter to leave plot\n")
for i in range(N_CLASSES):
  a= input("press enter to visualize single class elements\n")
  toTakeIds= pred_labels==i
  images_small= x_test[toTakeIds]
  smallData= data_smallDim[toTakeIds]
  x,y= zip(*smallData)
  if not y_test is None:
    y_test_small= y_test[toTakeIds]
    colorsList= np.array([ colors[elem] for elem in y_test_small])
  else: 
    colorsList= None
  fig, ax = plt.subplots()
  ax.set_title("pred class: %d. #%d"%(i, np.sum(toTakeIds)))
  plt.figure(2)
  col = ax.scatter(x, y, c=colorsList, lw=0, picker=True)
  fig.canvas.mpl_connect('pick_event', generateOnPick3(images_small))
  plt.show()
a= input("press enter to leave plot\n")
plt.close('all') 
