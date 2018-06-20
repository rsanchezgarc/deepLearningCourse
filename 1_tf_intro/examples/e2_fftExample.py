import os
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

from skimage import data as dataExamples
import time
import matplotlib.pyplot as plt

print(dataExamples.moon().shape, dataExamples.camera().shape )
IN_IMGS= np.stack( [dataExamples.moon(), dataExamples.camera()] )
imgs_size= IN_IMGS[0].shape[1]
print(IN_IMGS.shape)


def correctScale(imgsRaw):
  for i in range(imgsRaw.shape[0]):
    imgsRaw[i]=  imgsRaw[i]-np.min( imgsRaw[i]) / (np.max( imgsRaw[i])-np.min( imgsRaw[i]))
  return imgsRaw

IN_IMGS= correctScale(IN_IMGS)
IN_IMGS= IN_IMGS.astype(np.complex64)



#NUMPY VERSION


def npLPFilter(imgs_in):
  stime= time.time()  
  imgs=  np.fft.ifftshift(imgs_in)
  fftOut= np.fft.fft2(imgs)
  fftOut0Real, fftOut0Imag = np.real(fftOut), np.imag(fftOut)
  fft0Mod= np.sqrt(np.square(fftOut0Real)+np.square(fftOut0Imag) )

  #fftshift
  fftOutShiftedFinal= np.fft.fftshift(fftOut)

  fftOutReal, fftOutImag = np.real(fftOutShiftedFinal), np.imag(fftOutShiftedFinal)
  fftShifMod= np.sqrt(np.square(fftOutReal)+np.square(fftOutImag))

  SIZE=64
  npMask= np.array(np.zeros( (imgs_size, imgs_size) ), dtype=np.complex64)
  npMask[ npMask.shape[0]//2- npMask.shape[0]//SIZE:  npMask.shape[0]//2+ npMask.shape[0]//SIZE ,
          npMask.shape[1]//2- npMask.shape[1]//SIZE:  npMask.shape[1]//2+ npMask.shape[1]//SIZE ]=1+0j
  npMaskReal= np.real(npMask)     

  fftOutWithMask= np.multiply(fftOutShiftedFinal, npMask)

  fftOutRealM, fftOutImagM = np.real(fftOutWithMask), np.imag(fftOutWithMask)
  fftOutWithMaskM= np.sqrt(np.square(fftOutRealM)+np.square(fftOutImagM))

  #ifftshift # from central pixel to top rigth corner
  ifftOutShiftedFinal= np.fft.ifftshift(fftOutWithMask)

  filteredImagesTemp= np.fft.ifft2(ifftOutShiftedFinal)
  filteredImagesTemp= np.fft.fftshift(filteredImagesTemp)
  filteredImagesReal, filteredImagesImag = np.real(filteredImagesTemp), np.imag(filteredImagesTemp)
  filteredImages=filteredImagesReal
  return np.real(imgs_in), filteredImages

x__, filteredImages__= npLPFilter(IN_IMGS)

N_IMAG= x__.shape[0]
fig, axs = plt.subplots(2, N_IMAG, figsize=(10, 5))
for example_i in range(N_IMAG):
  axs[0][example_i].imshow(x__[example_i, ...], cmap='Greys')
  axs[0][example_i].set_title("original data")
  axs[1][example_i].imshow(filteredImages__[example_i, ...], cmap='Greys')
  axs[1][example_i].set_title("out")
plt.show()


#TF version

def tf_ffshift2(x):
  n= x.get_shape().as_list()[1] #Number of shifts
  p2= (n+1)//2
  fftOutShiftedDim1= tf.concat([x[:,p2:n,:], x[:,0:p2,:]], 1)
  fftOutShiftedFinal= tf.concat([fftOutShiftedDim1[:,:, p2:n], fftOutShiftedDim1[:,:, 0:p2]], 2)
  return fftOutShiftedFinal

def tf_iffshift2(x):
  n= x.get_shape().as_list()[1]
  p2= n - (n+1)//2
  ifftOutShiftedDim1= tf.concat([x[:,p2:n,:], x[:,0:p2,:]], 1)
  ifftOutShiftedFinal= tf.concat([ifftOutShiftedDim1[:,:, p2:n], ifftOutShiftedDim1[:,:, 0:p2]], 2)
  return ifftOutShiftedFinal
  
def tfLPFilter(imgs):
  #Placeholder(interface variables) definition
  x= tf.placeholder(dtype=tf.complex64, shape=[None, imgs.shape[1], imgs.shape[2]], name="x")
  x_Real, x_Imag = tf.real(x), tf.imag(x)
  xMod= tf.sqrt(tf.square(x_Real)+tf.square(x_Imag))
  #Graph computations definition
  i_x= tf_iffshift2(x)
  fftOut= tf.fft2d(i_x)
  fftOut0Real, fftOut0Imag = tf.real(fftOut), tf.imag(fftOut)
  fft0Mod= tf.sqrt(tf.square(fftOut0Real)+tf.square(fftOut0Imag))

  #fftshift
  fftOutShiftedFinal= tf_ffshift2(fftOut)
  
  fftOutReal, fftOutImag = tf.real(fftOutShiftedFinal), tf.imag(fftOutShiftedFinal)
  fft1Mod= tf.sqrt(tf.square(fftOutReal)+tf.square(fftOutImag))

  FILTER_SIZE=64
  npMask= np.array(np.zeros( (imgs_size, imgs_size) ), dtype=np.complex64)
  npMask[ npMask.shape[0]//2- npMask.shape[0]//FILTER_SIZE:  npMask.shape[0]//2+ npMask.shape[0]//FILTER_SIZE ,
          npMask.shape[1]//2- npMask.shape[1]//FILTER_SIZE:  npMask.shape[1]//2+ npMask.shape[1]//FILTER_SIZE ]=1+0j
  mask= tf.constant(npMask)
  fftOutWithMask= tf.multiply(fftOutShiftedFinal, mask)

  fftOutRealM, fftOutImagM = tf.real(fftOutWithMask), tf.imag(fftOutWithMask)
  fftOutWithMaskM= tf.sqrt(tf.square(fftOutRealM)+tf.square(fftOutImagM))

  #ifftshift
  ifftOutShiftedFinal= tf_iffshift2(fftOutWithMask)

  filteredImagesTemp= tf.ifft2d(ifftOutShiftedFinal)
  filteredImages_i= tf_ffshift2(filteredImagesTemp)
  filteredImagesReal, filteredImagesImag = tf.real(filteredImages_i), tf.imag(filteredImages_i)

  filteredImages=filteredImagesReal


  #Create session and initialize variables
  session = tf.Session()
  session.run(tf.global_variables_initializer())
  #Do actual computations
  stime= time.time()  
  feed_dict= {x: imgs}
  x_, fft0Mod_, fft1Mod_, filteredImages_, outMask = session.run([xMod, fft0Mod, fft1Mod, filteredImages, fftOutWithMaskM], feed_dict=feed_dict)
  stime= time.time()
  
  session.close()

  
  return x_, fft0Mod_, fft1Mod_, filteredImages_, outMask
    
x__, fft0Mod__, fft1Mod__, filteredImages__, outMask__= tfLPFilter(IN_IMGS)

N_IMAG= x__.shape[0]
fig, axs = plt.subplots(2, N_IMAG, figsize=(10, 5))
for example_i in range(N_IMAG):
  axs[0][example_i].imshow(x__[example_i, ...], cmap='Greys')
  axs[0][example_i].set_title("original data")
  axs[1][example_i].imshow(filteredImages__[example_i, ...], cmap='Greys')
  axs[1][example_i].set_title("out")
plt.show()

