# Save and Restore Models

## Overview

When publishing models and techniques, most machine learning practitioners share:

- code to create the model, and
- the trained weights, or parameters for the model

## Setup

### Installs and imports

Install and import TensorFlow and dependencies

```bash
$ pip install h5py pyyaml
```

```python
import os
import tensorflow as tf

from tensorflow import keras
```

## Get an example dataset

```python
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

Loading the dataset returns four NumPy arrays:

- The `train_images` and `train_labels` arrays are the training set (the data the model use to learn)

- The model is tested against the test set, the `test_images` and `test_labels` arrays

```python
train_images = train_images[:1000].reshape(-1, 28 * 28)
test_images = test_images[:1000].reshape(-1, 28 * 28)

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
```

### Preprocess the data

The data must be preprocessed before training the network. We scale the pixel values to a range of 0 to 1 before feeding to the network model. It's important that the training set and the testing set are preprocessed in the same way:

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
```

## Define the model

Building the network model requires configuring the layers of the model, then compiling the model

### Setup the layers

The basic building block of a neural network is the _layer_

```python
model = keras.Sequential([
    keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784, 1)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.keras.activations.softmax)
])
```

### Compile the model

Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:

- Loss function

- Optimizer

- Metrics

```python
model.compile(optimizer=tf.keras.optimzers.Adam(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
```

## Save checkpoints during training the model

### Checkpoint callback usage

Train the model and pass it the `ModelCheckpoint` callback

```python
checkpoint_path = "training/mnist-cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit(train_images, train_labels,  epochs=10, 
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # pass callback to training
```

Load the weights from the checkpoint, and evaluate:

```python
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
```

### Checkpoint callback options

The callback provides several options to give the resulting checkpoints unique names, and adjust the checkpointing frequency.

```python
checkpoint_path = "training/mnist-cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                save_weights_only=True,
                                                verbose=1, 
                                                # Save weights, every 5-epochs
                                                period=5)

model.save_weights(checkpoint_path.format(epoch=0))
model.fit(train_images, train_labels, epochs=50,
          validation_data=(test_images,test_labels),
          callbacks=[cp_callback], verbose=0)
```

## Reference

- https://www.tensorflow.org/tutorials/keras/save_and_restore_models
