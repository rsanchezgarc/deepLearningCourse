# deepLearningCourse

Prerequisites.
To make the most of this micro-course, basic knowledge of the following topics is required:
Python numpy.
Python functional programing: lambda functions, map, iterators.
Neural networks fundamentals (see http://cs231n.github.io/ if needed). 

Installation.

During this micro-course we will be using python (3.6 preferred) and the following packages and its dependencies:
Tensorflow >1.4
Keras >2.0
scikit-learn
scikit-image
hdf5
All them can be installed using pip. If GPU version of tensorflow desired, cuda and cudnn installation have to be done manually. 
Installation is detailed in https://www.tensorflow.org/install/
However, the easiest way to proceed is to install Anaconda, a python virtual environments system that will allow for automatic 
installation of all this packages included cuda. The next paragraphs will describe how to set up your machine using Anaconda.

1. Go to https://www.anaconda.com/download/#linux and download Ananconda-XXXXX.sh
2. Execute “ bash Ananconda-XXXXX.sh”
  1.1. Read license (press space bar several times to go faster) and accept terms.
  1.2. Indicate the path were Anaconda will be installed.
  1.3. If you want to set Anaconda as your default Python, let Anaconda add it to your $PATH in .bashrc file. Otherwise, you will need to use 
        /path/to/ananconda/bin/activate to load environments
3. Create an environment  “conda create -n myEnvironment” or “/path/to/ananconda/bin/conda create -n myEnvironment”
4. Activate environment “source activate  myEnvironment” or “source /path/to/ananconda/bin/activate myEnvironment”
5. Install packages:
  5.1. “conda install -c anaconda tensorflow” or if GPU available, “ conda install -c anaconda tensorflow-gpu” NOTE: make sure you have nvidia drivers 
        installed if you want to use tensorflow-gpu. For ubuntu sudo “apt-get install nvidia-XXX”
  5.2. “conda install -c anaconda keras”
  5.3. “conda install scikit-learn”
  5.4. “conda install scikit-image”
  5.5. “conda install hdf5”
6. Check if installation was successful.
  6.1. Launch python: “python”
  6.2. “import tensorflow as tf”
  6.3. “s=tf.Session()”
7. If no errors, installation has been completed.


